"""
A federated learning server using federated averaging to aggregate updates with secure multiparty computation implemented.
"""

from plato.servers import fedavg

class Server(fedavg.Server):

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        self.weight_keys = []

    def configure(self) -> None:
        """Configure the model information like weight shapes and parameter numbers."""
        super().configure()

        # OrderedDict object
        extract_model = self.trainer.model.cpu().state_dict()

        for key in extract_model.keys():
            self.weight_keys.append(key)