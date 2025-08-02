import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.
    
    Returns:
        (X_train, y_train, X_test, y_test): Preprocessed training and test data
    """
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to range [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension for CNN
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
    # Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

def visualize_samples(X_data, y_data, num_samples=10):
    """
    Visualize sample images from the dataset.
    
    Args:
        X_data: Image data
        y_data: Label data (one-hot encoded)
        num_samples: Number of samples to display
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Convert one-hot back to class label
        label = np.argmax(y_data[i])
        
        # Display image
        axes[i].imshow(X_data[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # Visualize some samples
    visualize_samples(X_train, y_train)