import json
import os
import sys
from tb_device_mqtt import TBDeviceMqttClient, TBPublishInfo
import tensorflow as tf
from flwr.common import ndarrays_to_parameters
from dev import mnist

MODULE_PATH = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(MODULE_PATH)
MLCUBE_DIR = os.path.join(MODULE_DIR, "mlcube")

def send_to_thingsboard(accuracy,loss):

    telemetry = {"accuracy": accuracy, "loss": loss}
    client = TBDeviceMqttClient("demo.thingsboard.io", username="S2gHkIiwwoIoEMcBzDYw")
    # Connect to ThingsBoard
    client.connect()
    # Sending telemetry without checking the delivery status
    client.send_telemetry(telemetry) 
    # Sending telemetry and checking the delivery status (QoS = 1 by default)
    result = client.send_telemetry(telemetry)
    # get is a blocking call that awaits delivery status  
    success = result.get() == TBPublishInfo.TB_ERR_SUCCESS
    # Disconnect from ThingsBoard
    client.disconnect()
    return True


def create_directory(path: str) -> None:
    print(f"Creating directory: {path}")
    os.makedirs(path, exist_ok=True)


def workspace_path(workspace_path: str, item_path: str, is_file=True) -> str:
    """Return filepath and create directories if required."""
    full_path = os.path.join(workspace_path, item_path)
    dir_path = os.path.dirname(full_path) if is_file else full_path
    create_directory(dir_path)
    return full_path


def run_task(workspace: str, task_name: str,client_id):
    #Check the task and call the specific function related to the task.
    if task_name == 'download':
        print("Initiating the download task from mlcube.py")
        mnist.download(workspace,client_id)
    elif task_name == 'train':
        print("Initiating the training task from mlcube.py")
        mnist.train(workspace,client_id)
    elif task_name == 'evaluate':
        print("Initiating the evaluate task from mlcube.py")
        mnist.evaluate(workspace,client_id)
    else:
        print(f"\n Unknown Task {task_name} \n")
        sys.exit(1)
    


def save_parameteras_as_model(workspace: str, parameters):
    """Write model to $WORKSPACE/model_in/mnist_model from parameters."""
    filepath = workspace_path(workspace, "model_in", False)
    model = get_model()
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.set_weights(parameters)
    model.save(f"{filepath}/mnist_model.keras")
    weights = model.get_weights()
    print(f"Saved the model into model_in and the weights of model are {weights} and parameters {parameters}")
    
    return True


def load_model_parameters(workspace: str):
    """Load and return model parameters."""
    filepath = workspace_path(workspace, "model", False)
    model = tf.keras.models.load_model(f"{filepath}/mnist_model.keras")
    parameters = model.get_weights()
    print("Getting parameters from model saved in /model after training")
    return parameters


def load_train_metrics(workspace: str):
    """Load and return metrics."""
    filepath = workspace_path(workspace, "metrics/train_metrics.json")
    with open(filepath, "r") as f:
        data = json.load(f)

    data["loss"] = float(data["loss"])
    data["accuracy"] = float(data["accuracy"])
    data["num_examples"] = 1  # int(data["num_examples"])
    print("Loading metrics from json file saved after training")
    return data


def load_evaluate_metrics(workspace: str):
    """Load and return metrics."""
    filepath = workspace_path(workspace, "metrics/evaluate_metrics.json")
    with open(filepath, "r") as f:
        data = json.load(f)

    data["loss"] = float(data["loss"])
    data["accuracy"] = float(data["accuracy"])
    data["num_examples"] = 1  # int(data["num_examples"])
    send_to_thingsboard(data["accuracy"],data["loss"])
    print(f"Metrics after evaluate {data}")
    return data


def write_hyperparameters(workspace: str, optimizer, epochs, batch_size):
    """Write hyperparameters to mlcube."""

    #Create parameters directory for the client

    os.makedirs(f"{workspace}/parameters",exist_ok=True)
    filepath = f"{workspace}/parameters/default.parameters.yaml"
    print(f"Created parameters directory to store the initial parameters required for the model {filepath}")
    with open(filepath, "w+") as f:
        params = [
            f'optimizer: "{optimizer}"',
            f"epochs: {epochs}",
            f"batch_size: {batch_size}",
        ]
        for param in params:
            f.write(f"{param}\n")
    print(f"Intial parameters for the model {params}")
    return True



def get_model():
    """Create example model."""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def initial_parameters():
    """Return initial checkpoint parameters."""
    model = get_model()
    return ndarrays_to_parameters(model.get_weights())
