import os
import time
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data, visualize_samples
from model import create_cnn_model, create_simple_model, compile_model, get_callbacks

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=128):
    """
    Train the model with the given data.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data and labels
        X_test, y_test: Validation data and labels
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        history: Training history
    """
    print("Starting model training...")
    start_time = time.time()
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    return history

def plot_training_history(history):
    """
    Plot training and validation metrics.
    
    Args:
        history: Training history from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """
    Main training function.
    """
    print("MNIST Digit Classifier Training")
    print("=" * 40)
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # Visualize some samples
    print("Visualizing sample images...")
    visualize_samples(X_train, y_train)
    
    # Create and compile model
    print("Creating CNN model...")
    model = create_cnn_model()
    model = compile_model(model)
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Train the model
    print("\nStarting training...")
    history = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=30,  # Reduced for faster training
        batch_size=128
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    # Final evaluation
    print("\nFinal Model Evaluation:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save final model
    model.save('models/final_model.h5')
    print("Model saved to models/final_model.h5")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()