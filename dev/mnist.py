from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.callbacks import EarlyStopping
import json
import logging
import logging.config
import os
import sys
from enum import Enum
import numpy as np
import tensorflow as tf
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Remove warning messages

logger = logging.getLogger(__name__)


class Task(str, Enum):
    DownloadData = "download"
    Train = "train"
    Evaluate = "evaluate"


def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def set_logger(log_path, task_name):
    logger_config = {
        "version": 1,
        "disable_existing_loggers": False,  # Change to False to ensure all loggers are considered
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "file_handler": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": os.path.join(log_path, f"mlcube_mnist_{task_name}.log"),
            }
        },
        "loggers": {
            "": {"level": "INFO", "handlers": ["file_handler"]},  # Root logger
            "tensorflow": {"level": "WARNING", "propagate": False},  # Reduce TensorFlow log level
            "__main__": {"level": "NOTSET", "propagate": True},
        },
    }
    logging.config.dictConfig(logger_config)
    logger = logging.getLogger(__name__)
    return logger


def download(workspace, client_id):
    """Task: download."""
    try:
        download_log_path = f"{workspace}/download_logs"
        os.makedirs(download_log_path, exist_ok=True)
        logger = set_logger(download_log_path, "download")
        
        print(f"Download Directory created to store the data {download_log_path}")
    
        logger.info("Starting download task")
        os.makedirs(f'{workspace}/data', exist_ok=True)
        data_file = os.path.join(f'{workspace}/data', "mnist.npz")

        if os.path.exists(data_file):
            logger.info("MNIST data has already been downloaded (file exists: %s)", data_file)
            print("MNIST file already exists")
            return True

        data_file = tf.keras.utils.get_file(
            fname=data_file,
            origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            file_hash="731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
        )

        if not os.path.isfile(data_file):
            raise ValueError(f"MNIST dataset has not been downloaded - dataset file does not exist: {data_file}")
        else:
            logger.info("MNIST dataset has been downloaded.")
            print("Dataset downloaded")
        return True
    
    except Exception as e:
        logger.exception(f"Error while executing the Download task. {e}")
        sys.exit(1)


def train(workspace,client_id) -> None:
    """Task: train."""

    try:

        train_log_path = f"{workspace}/train_logs"

        os.makedirs(train_log_path,exist_ok=True)
        print(f"Created log dir fot training logs {train_log_path}")
        #Calling the logger function to configure the logs for training the model.
        logger = set_logger(train_log_path,"train")

        with open(f'{workspace}/parameters/default.parameters.yaml', "r") as stream:
            parameters = yaml.load(stream, Loader=yaml.FullLoader)
        logger.info("Parameters have been read (%s).", f'{workspace}/parameters/default.parameters.yaml')
        print(f"Parameters for initial model {parameters}")
        dataset_file = os.path.join(f'{workspace}/data', "mnist.npz")
        with np.load(dataset_file, allow_pickle=True) as f:
            x_train, y_train = f["x_train"], f["y_train"]
        x_train = x_train / 255.0

        logger.info("Dataset has been loaded (%s).", dataset_file)
        print("Dataset ready")

        # Load from checkpoint;
        model = tf.keras.models.load_model(os.path.join(f'{workspace}/model_in', "mnist_model.keras"))
        print("Model loaded from directory ")
        

        logger.info("Model has been built.")

        model.compile(
            optimizer=parameters.get("optimizer", "adam"),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("Model has been compiled.")
        print("Model has been compiled.")
	

        # Train and evaluate
        history = model.fit(
            x_train,
            y_train,
            batch_size=parameters.get("batch_size", 32),
            epochs=parameters.get("epochs", 5),
        )
        logger.info("Model has been trained.")
        print("Model has been trained.")
        
        os.makedirs(f"{workspace}/metrics",exist_ok=True)
        print(f"Created metrics directory for saving loss and accuracy")
        with open(f"{workspace}/metrics/train_metrics.json", "w") as f:
            data_json = {
                "loss": str(history.history["loss"][-1]),
                "accuracy": str(history.history["accuracy"][-1]),
            }
            json.dump(data_json, f)
        print(f"After training round 1 {data_json}")
        os.makedirs(f"{workspace}/model", exist_ok=True)
        model.save(os.path.join(f"{workspace}/model", "mnist_model.keras"))
        logger.info("Model has been saved.")
        print("Model has been saved")
        return True

    except Exception as e:
        logger.error(e)
        print(f"Error during training {e}")
        sys.exit(1)


def evaluate(workspace,client_id) -> None:
    """Task: evaluate
    """
    try:
        eval_log_path = f"{workspace}/evaluate_logs"
        os.makedirs(eval_log_path,exist_ok=True)
        logger = set_logger(eval_log_path,"evaluate")

        
        print("Directory for evaluate logs created")
        #Calling the logger function to configure the logs for training the model.
        

        with open(f'{workspace}/parameters/default.parameters.yaml', "r") as stream:
            parameters = yaml.load(stream, Loader=yaml.FullLoader)
            logger.info("Parameters have been read (%s).", parameters)
            print(f"Parameters for evaluate fetched from yaml file {parameters}")

        dataset_file = os.path.join(f'{workspace}/data', "mnist.npz")
        with np.load(dataset_file, allow_pickle=True) as f:
            x_test, y_test = f["x_test"], f["y_test"]
        x_test = x_test / 255.0
        logger.info("Dataset has been loaded (%s).", dataset_file)
        print("Dataset has been loaded (%s).", dataset_file)
        model = tf.keras.models.load_model(os.path.join(f'{workspace}/model_in', "mnist_model.keras"))
        model.compile(
            optimizer=parameters.get("optimizer", "adam"),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        eval_result = model.evaluate(x_test, y_test)

        with open(f"{workspace}/metrics/evaluate_metrics.json", "w") as f:
            data_json = {"loss": str(eval_result[0]), "accuracy": str(eval_result[1])}
            json.dump(data_json, f)
        print(f"Results after evaluating {data_json}")
        logger.info("Model has been evaluated.")

    except Exception as e:
        logger.error(e)
        print(f"Error {e}")
        sys.exit(1)



