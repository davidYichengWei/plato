from typing import Any

import torch
from plato.processors import model
import crypten


class Processor(model.Processor):
    """
    A processor that decrypts model tensors
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        crypten.init()

    def process(self, data: Any) -> Any:
        print("MPC encrypt, print data")
        
        for key in data.keys():
            data[key] = crypten.cryptensor(data[key])

        for key in data.keys():
            print(data[key])

        return data