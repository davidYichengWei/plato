from typing import Any

from plato.processors import model
import pickle
import torch
import random
import numpy as np
from plato.config import Config
from plato.utils import s3
import logging
from kazoo.client import KazooClient
from kazoo.recipe.lock import Lock

class Processor(model.Processor):
    """
    A processor that decrypts model tensors
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client_share_index = 0
        self.s3_client = None
        self.client_id = kwargs["client_id"]
        self.zk = None

    # Randomly split a tensor into N shares
    def splitTensor(self, tensor, N, random_range):
        if N == 1:
            tensors = [tensor]
            return tensors

        # First split evenly
        tensors = [tensor / N for i in range(N)]

        # Generate a random number for each share
        rand_nums = [random.randrange(random_range) for i in range(N - 1)]
        rand_nums.append(0 - sum(rand_nums))

        # Add the random numbers to secret shares
        for i in range(N):
            tensors[i] += rand_nums[i]

        return tensors

    def process(self, data: Any) -> Any:
        # Load round_info object
        if hasattr(Config().server, "s3_endpoint_url"):
            self.zk = KazooClient(hosts = f'{Config().server.zk_address}:{Config().server.zk_port}')
            self.zk.start()
            lock = Lock(self.zk, '/my/lock/path')
            lock.acquire()
            logging.info("[%s] Acquired Zookeeper lock", self)

            self.s3_client = s3.S3()
            s3_key = "round_info"
            logging.debug("Retrieving round_info from S3")
            round_info = self.s3_client.receive_from_s3(s3_key)
        else:
            round_info_filename = "mpc_data/round_info"
            with open(round_info_filename, "rb") as round_info_file:
                round_info = pickle.load(round_info_file)

        num_clients = len(round_info['selected_clients'])

        # Store the client's weights before encryption in a file for testing
        weights_filename = "mpc_data/raw_weights_round%s_client%s" % (round_info['round_number'], self.client_id)
        f = open(weights_filename, "w")
        f.write(str(data))
        f.close()

        # Split weights randomly into n shares
        # Initialize data_shares to the shape of data
        data_shares = [data for i in range(num_clients)]

        # Iterate over the keys of data to split
        for key in data.keys():
            # multiply by num_samples used to train the client
            data[key] *= round_info[f"client_{self.client_id}_info"]['num_samples']

            # Split tensor randomly into num_clients shares
            tensor_shares = self.splitTensor(data[key], num_clients, 5)

            # Store tensor_shares into data_shares for the particular key
            for i in range(num_clients):
                tmp_tensor = tensor_shares[i]
                orig_size = list(tmp_tensor.size())
                dimen_val = np.prod(orig_size)
                t_1 = tmp_tensor.view(dimen_val)
                t_2 = t_1.view(orig_size)
                data_shares[i][key] = t_2


        # Store secret shares in round_info
        for i, client_id in enumerate(round_info['selected_clients']):
            # Skip the client itself
            if client_id == self.client_id:
                self.client_share_index = i # keep track of the index to return the client's share in the end
                continue

            if round_info[f"client_{client_id}_info"]["data"] == None:
                round_info[f"client_{client_id}_info"]["data"] = data_shares[i]
            else:
                for key in data.keys():
                    round_info[f"client_{client_id}_info"]["data"][key] += data_shares[i][key]

        logging.debug("Print round_info keys before filling client data")
        logging.debug(round_info.keys())

        # Store round_info object
        if hasattr(Config().server, "s3_endpoint_url"):
            self.s3_client.put_to_s3(s3_key, round_info)
            lock.release()
            logging.info("[%s] Released Zookeeper lock", self)
            self.zk.stop()
        else:
            with open(round_info_filename, "wb") as round_info_file:
                pickle.dump(round_info, round_info_file)

        return data_shares[self.client_share_index]