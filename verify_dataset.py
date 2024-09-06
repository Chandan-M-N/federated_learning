# import numpy as np
# from pymongo import MongoClient

# # Load MNIST data
# mnist_file = 'workspaces//10//data//mnist.npz'
# with np.load(mnist_file) as data:
#     x_train = data['x_train']
#     y_train = data['y_train']

# # Connect to MongoDB
# client = MongoClient("mongodb://localhost:27017/")  # Adjust the connection string as needed
# db = client["myDatabase"]  # Replace with your database name
# collection = db["MNISTtrain"]

# # Prepare data for insertion
# documents = []
# for i in range(len(x_train)):
#     document = {
#         "image": x_train[i].tolist(),  # Convert numpy array to list
#         "label": int(y_train[i])
#     }
#     documents.append(document)

# # Insert data into MongoDB
# collection.insert_many(documents)

# print("Data uploaded successfully to MNISTtrain!")
# print(f"Total number of documents uploaded: {len(documents)}")

import numpy as np
from pymongo import MongoClient

# Load MNIST test data
mnist_file = 'workspaces//10//data//mnist.npz'
with np.load(mnist_file) as data:
    x_test = data['x_test']
    y_test = data['y_test']

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Adjust the connection string as needed
db = client["myDatabase"]  # Replace with your database name
collection = db["MNISTtest"]

# Prepare data for insertion
documents = []
for i in range(len(x_test)):
    document = {
        "image": x_test[i].tolist(),  # Convert numpy array to list
        "label": int(y_test[i])
    }
    documents.append(document)

# Insert data into MongoDB
collection.insert_many(documents)

print("Data uploaded successfully to MNISTtest!")

