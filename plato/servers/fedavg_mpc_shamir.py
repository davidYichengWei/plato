"""
A simple federated learning server using federated averaging.
"""

import asyncio
import logging
import os
import random
import pickle
import numpy as np
import torch
import math

from plato.algorithms import registry as algorithms_registry
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import all_inclusive
from plato.servers import base
from plato.trainers import registry as trainers_registry
from plato.utils import csv_processor, fonts
from plato.utils import s3

class fraction:
    n = 0 #numerator
    d = 0 #denominator

    def __init__(self, n, d):
        self.n = n
        self.d = d
    
    def mult(self, f):
        temp_frac = fraction(self.n * f.n, self.d * f.d)
        return temp_frac
    
    def add(self, f):
        temp_frac = fraction(self.n * f.d + self.d * f.n, self.d * f.d)
        return temp_frac

class Server(base.Server):
    """Federated learning server using federated averaging."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(callbacks=callbacks)

        self.custom_model = model
        self.model = None

        self.custom_algorithm = algorithm
        self.algorithm = None

        self.custom_trainer = trainer
        self.trainer = None

        self.custom_datasource = datasource
        self.datasource = None

        self.testset = None
        self.testset_sampler = None
        self.total_samples = 0

        self.total_clients = Config().clients.total_clients
        self.clients_per_round = Config().clients.per_round

        self.s3_client = None

        logging.info(
            "[Server #%d] Started training on %d clients with %d per round.",
            os.getpid(),
            self.total_clients,
            self.clients_per_round,
        )

    def configure(self) -> None:
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        super().configure()

        total_rounds = Config().trainer.rounds
        target_accuracy = None
        target_perplexity = None

        if hasattr(Config().trainer, "target_accuracy"):
            target_accuracy = Config().trainer.target_accuracy
        elif hasattr(Config().trainer, "target_perplexity"):
            target_perplexity = Config().trainer.target_perplexity

        if target_accuracy:
            logging.info(
                "Training: %s rounds or accuracy above %.1f%%\n",
                total_rounds,
                100 * target_accuracy,
            )
        elif target_perplexity:
            logging.info(
                "Training: %s rounds or perplexity below %.1f\n",
                total_rounds,
                target_perplexity,
            )
        else:
            logging.info("Training: %s rounds\n", total_rounds)

        self.init_trainer()

        # Prepares this server for processors that processes outbound and inbound
        # data payloads
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Server", server_id=os.getpid(), trainer=self.trainer
        )

        if not (hasattr(Config().server, "do_test") and not Config().server.do_test):
            if self.datasource is None and self.custom_datasource is None:
                self.datasource = datasources_registry.get(client_id=0)
            elif self.datasource is None and self.custom_datasource is not None:
                self.datasource = self.custom_datasource()

            self.testset = self.datasource.get_test_set()
            if hasattr(Config().data, "testset_size"):
                self.testset_sampler = all_inclusive.Sampler(
                    self.datasource, testing=True
                )

        # Initialize the test accuracy csv file if clients compute locally
        if hasattr(Config().clients, "do_test") and Config().clients.do_test:
            accuracy_csv_file = (
                f"{Config().params['result_path']}/{os.getpid()}_accuracy.csv"
            )
            accuracy_headers = ["round", "client_id", "accuracy"]
            csv_processor.initialize_csv(
                accuracy_csv_file, accuracy_headers, Config().params["result_path"]
            )

        if hasattr(Config().server, "s3_endpoint_url"):
            self.s3_client = s3.S3()

    def init_trainer(self) -> None:
        """Setting up the global model, trainer, and algorithm."""
        if self.model is None and self.custom_model is not None:
            self.model = self.custom_model

        if self.trainer is None and self.custom_trainer is None:
            self.trainer = trainers_registry.get(model=self.model)
        elif self.trainer is None and self.custom_trainer is not None:
            self.trainer = self.custom_trainer(model=self.model)

        if self.algorithm is None and self.custom_algorithm is None:
            self.algorithm = algorithms_registry.get(trainer=self.trainer)
        elif self.algorithm is None and self.custom_algorithm is not None:
            self.algorithm = self.custom_algorithm(trainer=self.trainer)

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Process the client reports by aggregating their weights."""
        self.total_samples = sum(update.report.num_samples for update in updates)

        aggregated_weights = weights_received[0] # initialize with the first client's weights

        # Aggregate weights
        for i, client_weights in enumerate(weights_received):
            if i == 0:
                continue

            for key in aggregated_weights.keys():
                aggregated_weights[key] += client_weights[key]

        # Divide by total number of samples
        for key in aggregated_weights.keys():
            aggregated_weights[key] /= self.total_samples

        return aggregated_weights

    async def _process_reports(self):
        """Process the client reports by aggregating their weights."""
        weights_received = [update.payload for update in self.updates]

        weights_received = self.weights_received(weights_received)
        self.callback_handler.call_event("on_weights_received", self, weights_received)

        # Extract the current model weights as the baseline
        baseline_weights = self.algorithm.extract_weights()

        if hasattr(self, "aggregate_weights"):
            # Runs a server aggregation algorithm using weights rather than deltas
            logging.info(
                "[Server #%d] Aggregating model weights directly rather than weight deltas.",
                os.getpid(),
            )
            updated_weights = await self.aggregate_weights(
                self.updates, baseline_weights, weights_received
            )

            # Loads the new model weights
            self.algorithm.load_weights(updated_weights)
        else:
            # Computes the weight deltas by comparing the weights received with
            # the current global model weights
            deltas_received = self.algorithm.compute_weight_deltas(
                baseline_weights, weights_received
            )
            # Runs a framework-agnostic server aggregation algorithm, such as
            # the federated averaging algorithm
            logging.info("[Server #%d] Aggregating model weight deltas.", os.getpid())
            deltas = await self.aggregate_deltas(self.updates, deltas_received)
            # Updates the existing model weights from the provided deltas
            updated_weights = self.algorithm.update_weights(deltas)
            # Loads the new model weights
            self.algorithm.load_weights(updated_weights)

        # The model weights have already been aggregated, now calls the
        # corresponding hook and callback
        self.weights_aggregated(self.updates)
        self.callback_handler.call_event("on_weights_aggregated", self, self.updates)

        # Testing the global model accuracy
        if hasattr(Config().server, "do_test") and not Config().server.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.updates)
            logging.info(
                "[%s] Average client accuracy: %.2f%%.", self, 100 * self.accuracy
            )
        else:
            # Testing the updated model directly at the server
            logging.info("[%s] Started model testing.", self)
            self.accuracy = self.trainer.test(self.testset, self.testset_sampler)

        if hasattr(Config().trainer, "target_perplexity"):
            logging.info(
                fonts.colourize(
                    f"[{self}] Global model perplexity: {self.accuracy:.2f}\n"
                )
            )
        else:
            logging.info(
                fonts.colourize(
                    f"[{self}] Global model accuracy: {100 * self.accuracy:.2f}%\n"
                )
            )

        self.clients_processed()
        self.callback_handler.call_event("on_clients_processed", self)

    def clients_processed(self) -> None:
        """Additional work to be performed after client reports have been processed."""

    def get_logged_items(self) -> dict:
        """Get items to be logged by the LogProgressCallback class in a .csv file."""
        return {
            "round": self.current_round,
            "accuracy": self.accuracy,
            "elapsed_time": self.wall_time - self.initial_wall_time,
            "comm_time": max(update.report.comm_time for update in self.updates),
            "round_time": max(
                update.report.training_time + update.report.comm_time
                for update in self.updates
            ),
            "comm_overhead": self.comm_overhead,
        }

    @staticmethod
    def accuracy_averaging(updates):
        """Compute the average accuracy across clients."""
        # Get total number of samples
        total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        accuracy = 0
        for update in updates:
            accuracy += update.report.accuracy * (
                update.report.num_samples / total_samples
            )

        return accuracy

    def recover_secret(self, x, y, M):
        """
        generate the secret from the given points,
        uses Lagrange Basis Polynomial to find poly[0]
        """
        ans = fraction(0, 1)

        for i in range(0, M):
            frac = fraction(y[i], 1)
            for j in range(0, M):
                #compute the lagrange terms
                if i != j:
                    temp = fraction(0-x[j], x[i]-x[j])
                    frac = frac.mult(temp)
            ans = ans.add(frac)

        return ans.n / ans.d

    def decrypt_tensor(self, tensors, M=None):
        """recursively call recover_secret on a tensor"""
        tensor_size = list(tensors.size())
        N = tensor_size[0] #number of participating clients
        if M is None: #number of points used in decryption, K <= M <= N
            M = N - 2 #default
        num_weights = int(math.prod(tensor_size) / (N * 2))
        coords_shape = [N, num_weights, 2]
        coords = tensors.view(coords_shape)

        secret_arr = torch.zeros([num_weights])
        for i in range(num_weights):
            list_points = torch.zeros([M, 2])
            for j in range(M):
                #picking the first M points from the N points received
                list_points[j] = coords[j][i]
            x = np.zeros(M)
            y = np.zeros(M)
            for j in range(M):
                x[j] = list_points[j][0]
                y[j] = list_points[j][1]
            secret_arr[i] = self.recover_secret(x, y, M)

        tensor_size.pop(0)
        tensor_size.pop(-1)
        return secret_arr.view(tensor_size)


    def weights_received(self, weights_received):
        """
        Method called after the updated weights have been received.
        """
        # Load round_info object
        if self.s3_client is not None:
            s3_key = "round_info"
            logging.debug("Retrieving round_info from S3")
            round_info = self.s3_client.receive_from_s3(s3_key)
        else:
            round_info_filename = "mpc_data/round_info"
            logging.debug("Retrieving round_info from file")
            with open(round_info_filename, "rb") as round_info_file:
                round_info = pickle.load(round_info_file)

        # If there is only 1 client per round, skip the following step
        if len(round_info['selected_clients']) == 1:
            return weights_received

        # Combine the client's weights share with weights shares sent from other clients
        for i, from_client in enumerate(round_info['selected_clients']):
            for key in weights_received[i].keys():
                tensor_size =  list(weights_received[i][key].size())
                tensor_size.insert(0, len(round_info['selected_clients']))
                cur_val = torch.zeros(tensor_size)
                cur_val[0] = weights_received[i][key]
                insert_idx = 1

                for j, to_client in enumerate(round_info['selected_clients']):
                    if j == i:
                        continue
                    cur_val[insert_idx] = \
                        round_info[f"client_{to_client}_{from_client}_info"]["data"][key]
                    insert_idx = insert_idx + 1

                weights_received[i][key] = self.decrypt_tensor(cur_val)

        # Store the combined weights in files for testing
        for i, client in enumerate(round_info['selected_clients']):
            encrypted_weights_filename = \
                f"mpc_data/encrypted_weights_round{round_info['round_number']}_client{client}"
            file = open(encrypted_weights_filename, "w", encoding="utf8")
            file.write(str(weights_received[i]))
            file.close()

        return weights_received

    def weights_aggregated(self, updates):
        """
        Method called after the updated weights have been aggregated.
        """

    def choose_clients(self, clients_pool, clients_count):
        """Chooses a subset of the clients to participate in each round."""
        assert clients_count <= len(clients_pool)
        random.setstate(self.prng_state)

        # Select clients randomly
        selected_clients = random.sample(clients_pool, clients_count)

        round_info = {
            "round_number": self.current_round,
            "selected_clients": selected_clients
        }

        for client_to in selected_clients:
            round_info[f"client_{client_to}_info"]  = {
                "num_samples": None
            }
            for client_from in selected_clients:
                round_info[f"client_{client_to}_{client_from}_info"] = {
                    "data": None
                }

        # Store selected clients info into a file or S3 bucket
        if self.s3_client is not None:
            s3_key = "round_info"
            self.s3_client.put_to_s3(s3_key, round_info)
            logging.debug("[%s] Stored information for the current round in an S3 bucket", self)
        else:
            round_info_filename = "mpc_data/round_info"
            with open(round_info_filename, "wb") as round_info_file:
                pickle.dump(round_info, round_info_file)
            logging.debug("[%s] Stored information for the current "\
                "round in file mpc_data/round_info", self)

        self.prng_state = random.getstate()
        logging.info("[%s] Selected clients: %s", self, selected_clients)
        return selected_clients
