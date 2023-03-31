from typing import Any

from plato.processors import model
import pickle
import torch
import random
from plato.config import Config
from plato.utils import s3
import logging
from kazoo.client import KazooClient
from kazoo.recipe.lock import Lock
import copy
import math

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
        self.lock = kwargs["file_lock"]

    #y = poly[0] + x*poly[1] + x^2*poly[2]
    def calculate_Y (self, x, poly): #np.array
        y = 0
        tmp = 1

        for coeff in poly:
            y = (y + coeff * tmp)
            tmp = tmp * x
        return y

    #Function to encode a given secret
    #Generates a K-degree polynomial, and N points
    def secret_sharing(self, S, N, K):
        #S is the secret int
        poly = torch.zeros(K)
        poly[0] = S #the y-intercept
        for i in range(1, K):
            #poly[i] = randint(1, 997) 
            poly[i] = 100 + i
        points = torch.zeros([N, 2])

        #Generate N points from the polynomial
        for j in range(1, N+1):
            points[j-1][0] = j #x value of point
            points[j-1][1] = self.calculate_Y(j, poly) #y value of point

        return points

    # Encrypt tensor using Shamir
    def splitTensor(self, secret_data, N, K=None):
        if K is None:
            K = max(N-2, 1) #threshold chosen by the user
        
        orig_size = list(secret_data.size())
        dimen_len = math.prod(orig_size) #product of the size array
        arr_one_dimen = secret_data.view(dimen_len)

        coords_size = [N, dimen_len, 2] #4 (num clients), 18 (num points), 2 (2-D point)
        coords = torch.empty(coords_size)   

        for i in range(dimen_len): #iterate through each weight value
            points = self.secret_sharing(arr_one_dimen[i], N, K) #size [N, 2] tensor
            for j in range( N):
                coords[j][i] = points[j] #coordinates[j] is the data points sending to client j

        encrypted_size = orig_size
        encrypted_size.insert(0, N)
        encrypted_size.append(2) # using 2-D points
        encrypted = coords.view(encrypted_size)  

        return encrypted #[N, (orig_size), 2]


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
            self.lock.acquire()
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
        data_shares = [copy.deepcopy(data) for i in range(num_clients)]

        # Iterate over the keys of data to split
        for key in data.keys():
            # multiply by num_samples used to train the client
            data[key] *= round_info[f"client_{self.client_id}_info"]['num_samples']

            tensor_shares = self.splitTensor(data[key], num_clients) 

            # Store tensor_shares into data_shares for the particular key
            for i in range(num_clients):
                data_shares[i][key] = tensor_shares[i]
        
        # Store secret shares in round_info
        for i, client_id in enumerate(round_info['selected_clients']):
            # Skip the client itself
            if client_id == self.client_id: 
                self.client_share_index = i # keep track of the index to return the client's share in the end
                continue

            round_info[f"client_{client_id}_{self.client_id}_info"]["data"] = data_shares[i]


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
            self.lock.release()

        return data_shares[self.client_share_index]