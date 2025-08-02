import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import argparse
import os

def load_trained_model(model_path='models/best_model.h5'):
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        model: Loaded Keras model
    """
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except:
        print(f"Could not load model from {model_path}")
        print("Trying alternative path...")
        try:
            model = load_model('models/final_model.h5')
            print("Loaded final_model.h5 instead")
            return model
        except:
            print("No trained model found. Please run train.py first.")
            return None

def preprocess_image(image_path):
    """
    Preprocess an image for model inference.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        processed_image: Preprocessed image ready for model input
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model input (add batch and channel dimensions)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def predict_digit(model, image_array):
    """
    Predict the digit in the image.
    
    Args:
        model: Trained Keras model
        image_array: Preprocessed image array
    
    Returns:
        prediction: Predicted digit (0-9)
        confidence: Confidence score
        all_probabilities: Probabilities for all classes
    """
    # Get prediction probabilities
    probabilities = model.predict(image_array)[0]
    
    # Get predicted class and confidence
    predicted_digit = np.argmax(probabilities)
    confidence = probabilities[predicted_digit]
    
    return predicted_digit, confidence, probabilities

def visualize_prediction(image_path, image_array, predicted_digit, confidence, all_probabilities):
    """
    Visualize the prediction results.
    
    Args:
        image_path: Path to the original image
        image_array: Preprocessed image array
        predicted_digit: Predicted digit
        confidence: Confidence score
        all_probabilities: Probabilities for all classes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display the processed image
    ax1.imshow(image_array.reshape(28, 28), cmap='gray')
    ax1.set_title(f'Input Image\nPrediction: {predicted_digit} (Confidence: {confidence:.3f})')
    ax1.axis('off')
    
    # Display probability distribution
    ax2.bar(range(10), all_probabilities)
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Probability')
    ax2.set_title('Prediction Probabilities')
    ax2.set_xticks(range(10))
    ax2.grid(True, alpha=0.3)
    
    # Highlight the predicted digit
    ax2.bar(predicted_digit, all_probabilities[predicted_digit], color='red', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = f'outputs/prediction_{os.path.basename(image_path)}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Prediction visualization saved to {output_path}")
    
    plt.show()

def predict_from_mnist_test():
    """
    Test inference on some MNIST test images.
    """
    from data_loader import load_and_preprocess_data
    
    # Load test data
    _, _, X_test, y_test = load_and_preprocess_data()
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    # Select a few random test images
    num_samples = 5
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i, idx in enumerate(indices):
        # Get image and true label
        image = X_test[idx]
        true_label = np.argmax(y_test[idx])
        
        # Make prediction
        predicted_digit, confidence, probabilities = predict_digit(model, image.reshape(1, 28, 28, 1))
        
        # Display image
        axes[0, i].imshow(image.reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f'True: {true_label}\nPred: {predicted_digit}\nConf: {confidence:.3f}')
        axes[0, i].axis('off')
        
        # Display probabilities
        axes[1, i].bar(range(10), probabilities)
        axes[1, i].set_xlabel('Digit')
        axes[1, i].set_ylabel('Probability')
        axes[1, i].set_xticks(range(10))
        axes[1, i].bar(predicted_digit, probabilities[predicted_digit], color='red', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('outputs/mnist_test_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """
    Main inference function.
    """
    parser = argparse.ArgumentParser(description='MNIST Digit Recognition Inference')
    parser.add_argument('--image', type=str, help='Path to image file for prediction')
    parser.add_argument('--test-mnist', action='store_true', help='Test on MNIST test images')
    
    args = parser.parse_args()
    
    print("MNIST Digit Recognition - Inference")
    print("=" * 40)
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    if args.image:
        # Load model
        model = load_trained_model()
        if model is None:
            return
        
        # Process image
        print(f"Processing image: {args.image}")
        image_array = preprocess_image(args.image)
        
        if image_array is None:
            return
        
        # Make prediction
        predicted_digit, confidence, all_probabilities = predict_digit(model, image_array)
        
        # Display results
        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence: {confidence:.4f}")
        print("\nProbabilities for each digit:")
        for digit, prob in enumerate(all_probabilities):
            print(f"  {digit}: {prob:.4f}")
        
        # Visualize results
        visualize_prediction(args.image, image_array, predicted_digit, confidence, all_probabilities)
    
    elif args.test_mnist:
        print("Testing on MNIST test images...")
        predict_from_mnist_test()
    
    else:
        print("Please specify either --image <path> or --test-mnist")
        print("Example usage:")
        print("  python inference.py --image my_digit.png")
        print("  python inference.py --test-mnist")

if __name__ == "__main__":
    main()