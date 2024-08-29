import numpy as np
import matplotlib.pyplot as plt

def visualize_samples(npz_file_path):
    """Load and visualize samples from the .npz file."""
    try:
        with np.load(npz_file_path) as data:
            x_train = data['x_train']
            y_train = data['y_train']

            # Display the first 5 images and their labels
            for i in range(5):
                plt.subplot(1, 5, i + 1)
                plt.imshow(x_train[i], cmap='gray')
                plt.title(f'Label: {y_train[i]}')
                plt.axis('off')

            plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

# Path to the .npz file
npz_file_path = "workspaces/8/data/mnist.npz"
visualize_samples(npz_file_path)
