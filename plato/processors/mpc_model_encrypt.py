from typing import Any

from plato.processors import model
import pickle


class Processor(model.Processor):
    """
    A processor that decrypts model tensors
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, data: Any) -> Any:
        # Get a list of client ids selected for this round of training
        round_info_filename = "mpc_data/round_info"

        with open(round_info_filename, "rb") as round_info_file:
            round_info = pickle.load(round_info_file)

        # print("Information for this round")
        # print(round_info)

        num_clients = len(round_info['selected_clients'])

        data_copy = data

        
        # multiply by num_samples used to train the client
        # Split weights into n shares
            # Split evenly for now
        for key in data_copy.keys():
            data_copy[key] *= round_info['num_samples']
            data_copy[key] /= num_clients

        # Store secret shares in round_info
        for client in round_info['selected_clients']:
            if round_info[f"client_{client}_data"] == None:
                round_info[f"client_{client}_data"] = data_copy
            else:
                for key in data_copy.keys():
                    round_info[f"client_{client}_data"][key] += data_copy[key]

            # round_info[f"client_{client}_data"].append(data_copy)


        print("Print round info after filling client data")
        print(round_info.keys())
        # print(round_info['client_3_data']['conv1.bias'])
        # print(round_info['client_2_data']['conv1.bias'])

        # Dump round_info into file
        with open(round_info_filename, "wb") as round_info_file:
            pickle.dump(round_info, round_info_file)

        return data