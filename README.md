---
tags: [quickstart, vision, deployment]
dataset: [MNIST]
framework: [tensorflow, Keras]
---

# Flower Example using TensorFlow/Keras

This introductory example to Flower uses MLCube together with Keras, but deep knowledge of Keras is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use-cases with MLCube. Running this example in itself is quite easy.

### Installing Dependencies

Project dependencies (such as `tensorflow` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly, you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors, you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

To verify that everything works correctly, you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors, you're good to go!

#### Docker

For the MLCube setup you will need to install Docker on your system. Please refer to the [Docker install guide](https://docs.docker.com/get-docker/) on how to do that.

## Run Federated Learning with TensorFlow/Keras in MLCube with Flower

Afterward, you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
./dev/server.sh
```

Now you are ready to start the clients. We have prepared a simple script called `client.sh`, which accepts a CLIENT_ID and can be executed as in:

```shell
# Shell 1
./dev/client.sh 1
```

```shell
# Shell 2
./dev/client.sh 2
```

