import os
import sys

import flwr as fl
import mlcube_utils as mlcube



# Define Flower client
class MLCubeClient(fl.client.NumPyClient):
    def __init__(self, workspace: str, optimizer="adam", epochs=10, batch_size=32) -> None:

        super().__init__()
        self.workspace = workspace

        #Creating Workspace directory based on Client ID. Ex: /workspaces/1
        os.makedirs(workspace,exist_ok=True)
        print(f"Created Directory {workspace}")
        mlcube.write_hyperparameters(self.workspace, optimizer, epochs, batch_size)
        mlcube.run_task(self.workspace, "download",sys.argv[1])


    def fit(self, parameters, config):  # type: ignore
        print(f"Fit method parameters  or initial parameters shared by server {parameters} and config {config}")
        print(f"Client ID {sys.argv[1]}")
        mlcube.save_parameteras_as_model(self.workspace, parameters)
        mlcube.run_task(self.workspace, "train",sys.argv[1])
        print("Training done")
        parameters = mlcube.load_model_parameters(self.workspace)
        print(f"Parameters after training from the model saved in /model {parameters}")
        config = mlcube.load_train_metrics(self.workspace)
        print(f"Config after training {config}")
        return parameters, config["num_examples"], config

    def evaluate(self, parameters, config):  # type: ignore
        print(f"Parameters shared by server for evaluate {parameters} and config {config}")
        mlcube.save_parameteras_as_model(self.workspace, parameters)
        mlcube.run_task(self.workspace, "evaluate",sys.argv[1])
        config = mlcube.load_evaluate_metrics(self.workspace)
        print(f"Config shared to server after evaluate {config}")
        return config["loss"], config["num_examples"], config


def main():
    """Start Flower client.

    Use first argument passed as workspace name
    """
    client_id = sys.argv[1]

    print(f"Initating process for client {client_id}")

    workspace = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "workspaces", client_id
    )

    print(f"Workspace created {workspace}")

    fl.client.start_client(
        server_address="0.0.0.0:8087",
        client=MLCubeClient(workspace=workspace).to_client(),
    )
    print(f"Started the Client {client_id}")


if __name__ == "__main__":
    main()
