from typing import Any

from plato.processors import model
import pickle
import torch
import random

class Processor(model.Processor):
    """
    A processor that decrypts model tensors
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client_share_index = 0

    # Randomly split a tensor into N shares
    def splitTensor(self, tensor, N, random_range):
        if N == 1:
            tensors = [tensor]
            return tensors

        # get shape of the tensor
        # tensors = [torch.rand(tensor.size()) for i in range(N - 1)]
        # tensors.append(tensor - sum(tensors))

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
        # Get a list of client ids selected for this round of training
        round_info_filename = "mpc_data/round_info"

        with open(round_info_filename, "rb") as round_info_file:
            round_info = pickle.load(round_info_file)

        num_clients = len(round_info['selected_clients'])

        
        # Split weights randomly into n shares
        # Initialize data_shares to the shape of data
        data_shares = [data for i in range(num_clients)]

        # Iterate over the keys of data to split
        for key in data.keys():
            # multiply by num_samples used to train the client
            data[key] *= round_info['current_client_info']['num_samples']

            # Split tensor randomly into num_clients shares
            tensor_shares = self.splitTensor(data[key], num_clients, 5)

            # Check if split is correct
            # if torch.equal(data[key], sum(tensor_shares)):
            #     print("Split is correct")
            # else:
            #     print("Split is incorrect")
            #     print("Original tensor")
            #     print(data[key])
            #     print("Sum of shares")
            #     print(sum(tensor_shares))

            # Store tensor_shares into data_shares for the particular key
            for i in range(num_clients):
                data_shares[i][key] = tensor_shares[i]


        # Store secret shares in round_info
        for i, client_id in enumerate(round_info['selected_clients']):
            # Skip the client itself
            if client_id == round_info['current_client_info']['client_id']:
                self.client_share_index = i # keep track of the index to return the client's share in the end
                continue

            if round_info[f"client_{client_id}_data"] == None:
                round_info[f"client_{client_id}_data"] = data_shares[i]
            else:
                for key in data.keys():
                    round_info[f"client_{client_id}_data"][key] += data_shares[i][key]


        print("Print round info after filling client data")
        print(round_info.keys())

        # Dump round_info into file
        with open(round_info_filename, "wb") as round_info_file:
            pickle.dump(round_info, round_info_file)

        return data_shares[self.client_share_index]