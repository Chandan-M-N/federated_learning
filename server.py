import flwr as fl
from flwr.server.strategy import FedAvg

import mlcube_utils as mlcube


def main():
    strategy = FedAvg(initial_parameters=mlcube.initial_parameters(),min_available_clients=2,min_fit_clients=2,min_evaluate_clients=2)
    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8087",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=1),
    )


if __name__ == "__main__":
    main()
