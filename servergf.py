from pathlib import Path
from typing import Dict, Optional, Tuple, Callable, List, Tuple, Union
from flwr.common import Scalar, FitRes, Parameters, EvaluateRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import numpy as np  
import keras  
import argparse
import flwr as fl  
import tensorflow as tf  
import pandas as pd 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout
import client
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desabilita a GPU


class SaveModelStrategy(fl.server.strategy.FaultTolerantFedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert 'Parameters' to 'List[np.ndarray]'
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregate_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        ac = aggregated_accuracy * 100
        al = aggregated_loss * 100
        with open('results/threshold.txt', 'w') as f:
            f.write(f"{ac}\n")
            f.write(f"{al}\n")
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")
        print(f"Round {server_round} loss aggregated from client results: {aggregated_loss}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}


def main() -> None:
    parser = argparse.ArgumentParser(description="FL")
    parser.add_argument("--clients", default=3, type=int)
    parser.add_argument("--rounds", default=3, type=int)
    parser.add_argument("--fraction", default=1.0, type=float)
    parser.add_argument("--address", type=str, required=True, help=f"String of the gRPC server address in the format 127.0.0.1:8080")
    args = parser.parse_args()

    # ALTERAR AQUI PARA QUE CARREGUE MODELO TREINADO

    # Load pre-trained model
    model = keras.models.load_model('models/pretrained_model_dataset.keras')

    # Create strategy - estratégia particular do FEDAVG
    strategy = SaveModelStrategy(
        fraction_fit=args.fraction,
        fraction_evaluate=args.fraction,
        min_fit_clients=args.clients,
        min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),  # Pass pre-trained weights
    )

    # Start Flower server (SSL-enabled) for federated learning rounds
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        certificates=(
            Path(".cache/certificates/ca.crt").read_bytes(),
            Path(".cache/certificates/server.pem").read_bytes(),
            Path(".cache/certificates/server.key").read_bytes(),
        ),
    )

    # NOVO MODELO GERADO
    model.save('models/model_final_gf.keras')


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    _, _, x_val, y_val = client.load_partition()
   
    #ALTERAÇÃO CÓDIGO ORIGINAL
    x_val = x_val[:, :10, :]  # Formato (num_amostras, 10, 1)


    with open('results/classgf.txt', 'w') as f:
        for line in y_val:
            f.write(f"{line}\n")

    # The 'evaluate' function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 16,
        "local_epochs": 5 if server_round < 2 else 5,
        "validation_split": 0.33,
        "learning_rate": 0.0001
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
