import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_and_preprocess_data

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

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and return detailed metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels (one-hot encoded)
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true)
    
    # Get loss
    loss = model.evaluate(X_test, y_test, verbose=0)[0]
    
    return {
        'accuracy': accuracy,
        'loss': loss,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }

def plot_confusion_matrix(y_true, y_pred, save_path='outputs/confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_classification_errors(X_test, y_true, y_pred, y_pred_prob, num_errors=20):
    """
    Plot examples of classification errors.
    
    Args:
        X_test: Test images
        y_true: True labels
        y_pred: Predicted labels
        y_pred_prob: Prediction probabilities
        num_errors: Number of errors to display
    """
    # Find misclassified examples
    errors = np.where(y_pred != y_true)[0]
    
    if len(errors) == 0:
        print("No classification errors found!")
        return
    
    # Select random errors to display
    np.random.shuffle(errors)
    errors_to_show = errors[:min(num_errors, len(errors))]
    
    # Create subplot grid
    cols = 5
    rows = (len(errors_to_show) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []
    
    for i, error_idx in enumerate(errors_to_show):
        if i < len(axes):
            # Get confidence of the wrong prediction
            confidence = y_pred_prob[error_idx][y_pred[error_idx]]
            
            axes[i].imshow(X_test[error_idx].reshape(28, 28), cmap='gray')
            axes[i].set_title(f'True: {y_true[error_idx]}, Pred: {y_pred[error_idx]}\nConf: {confidence:.2f}')
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(errors_to_show), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/classification_errors.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_prediction_confidence(y_pred_prob, y_true, y_pred):
    """
    Plot prediction confidence distribution.
    
    Args:
        y_pred_prob: Prediction probabilities
        y_true: True labels
        y_pred: Predicted labels
    """
    # Get confidence scores
    confidences = np.max(y_pred_prob, axis=1)
    correct_predictions = (y_true == y_pred)
    
    plt.figure(figsize=(12, 4))
    
    # Plot confidence distribution for correct vs incorrect predictions
    plt.subplot(1, 2, 1)
    plt.hist(confidences[correct_predictions], bins=50, alpha=0.7, label='Correct', color='green')
    plt.hist(confidences[~correct_predictions], bins=50, alpha=0.7, label='Incorrect', color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy vs confidence
    plt.subplot(1, 2, 2)
    confidence_bins = np.linspace(0, 1, 11)
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    accuracy_by_confidence = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
        if np.sum(mask) > 0:
            accuracy = np.mean(correct_predictions[mask])
            accuracy_by_confidence.append(accuracy)
        else:
            accuracy_by_confidence.append(0)
    
    plt.plot(bin_centers, accuracy_by_confidence, 'bo-')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Prediction Confidence')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/prediction_confidence.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """
    Main evaluation function.
    """
    print("MNIST Model Evaluation")
    print("=" * 30)
    
    # Load test data
    print("Loading test data...")
    _, _, X_test, y_test = load_and_preprocess_data()
    
    # Load trained model
    print("Loading trained model...")
    model = load_trained_model()
    
    if model is None:
        return
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    # Print results
    print(f"\nTest Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Loss: {results['loss']:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(results['y_true'], results['y_pred']))
    
    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(results['y_true'], results['y_pred'])
    
    # Plot classification errors
    print("Analyzing classification errors...")
    plot_classification_errors(X_test, results['y_true'], results['y_pred'], results['y_pred_prob'])
    
    # Plot prediction confidence
    print("Analyzing prediction confidence...")
    plot_prediction_confidence(results['y_pred_prob'], results['y_true'], results['y_pred'])
    
    print("\nEvaluation completed! Check the outputs/ directory for visualization files.")

if __name__ == "__main__":
    main()