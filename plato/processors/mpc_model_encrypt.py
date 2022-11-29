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

        num_clients = len(round_info['selected_clients'])

        
        # Split weights into n shares
            # Split evenly for now
        for key in data.keys():
            # multiply by num_samples used to train the client
            data[key] *= round_info['current_client_info']['num_samples']
            data[key] /= num_clients

        # Store secret shares in round_info
        for client_id in round_info['selected_clients']:
            # Skip the client itself
            if client_id == round_info['current_client_info']['client_id']:
                continue

            if round_info[f"client_{client_id}_data"] == None:
                round_info[f"client_{client_id}_data"] = data
            else:
                for key in data.keys():
                    round_info[f"client_{client_id}_data"][key] += data[key]


        print("Print round info after filling client data")
        print(round_info.keys())

        # Dump round_info into file
        with open(round_info_filename, "wb") as round_info_file:
            pickle.dump(round_info, round_info_file)

        return data