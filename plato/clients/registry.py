"""
The registry that contains all available federated learning clients.

Having a registry of all available classes is convenient for retrieving an instance based
on a configuration at run-time.
"""
import logging

from plato.config import Config
from plato.clients import (
    simple,
    mistnet,
    simple_personalized
)

registered_clients = {
    "simple": simple.Client,
    "simple_personalized": simple_personalized.Client,
    "mistnet": mistnet.Client,
}


def get(model=None, datasource=None, algorithm=None, trainer=None):
    """Get an instance of the server."""
    if hasattr(Config().clients, "type"):
        client_type = Config().clients.type
    else:
        client_type = Config().algorithm.type

    if client_type in registered_clients:
        logging.info("Client: %s", client_type)
        registered_client = registered_clients[client_type](
            model=model, datasource=datasource, algorithm=algorithm, trainer=trainer
        )
    else:
        raise ValueError(f"No such client: {client_type}")

    return registered_client
