
### Poetry

Install poetry

```shell
sudo apt install python3-poetry
```

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly, you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors, you're good to go!

Install TB MQTT client

```shell
pip3 install tb-mqtt-client
```

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


## Run Federated Learning with TensorFlow/Keras with Flower

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

