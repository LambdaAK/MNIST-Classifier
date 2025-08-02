# MNIST Digit Classifier

A complete machine learning project for classifying handwritten digits using the MNIST dataset with TensorFlow/Keras.

## Project Structure

```
digit recognizer/
├── data_loader.py      # Data loading and preprocessing
├── model.py           # Neural network architectures
├── train.py           # Training script
├── evaluate.py        # Model evaluation and visualization
├── inference.py       # Inference on new images
├── requirements.txt   # Python dependencies
├── models/           # Saved model files
├── data/             # Dataset storage
└── outputs/          # Generated plots and results
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model
```bash
python train.py
```
This will:
- Load and preprocess the MNIST dataset
- Train a CNN model with validation
- Save the best model to `models/`
- Generate training history plots

### 2. Evaluate the Model
```bash
python evaluate.py
```
This will:
- Load the trained model
- Evaluate on test data
- Generate confusion matrix
- Show classification errors
- Analyze prediction confidence

### 3. Make Predictions
```bash
# Test on MNIST test images
python inference.py --test-mnist

# Predict on your own image
python inference.py --image path/to/your/image.png
```

## Model Architecture

The project includes two model architectures:

### CNN Model (Default)
- 3 Convolutional blocks with BatchNormalization and Dropout
- MaxPooling for dimensionality reduction
- Dense layers with regularization
- ~99%+ accuracy on MNIST

### Simple Model
- Fully connected neural network
- Faster training for quick experiments
- ~97-98% accuracy on MNIST

## Features

- **Data Preprocessing**: Normalization, reshaping, one-hot encoding
- **Model Training**: Early stopping, learning rate scheduling, model checkpointing
- **Evaluation**: Comprehensive metrics, confusion matrix, error analysis
- **Visualization**: Training curves, sample images, prediction confidence
- **Inference**: Support for custom images and batch prediction

## Results

The CNN model typically achieves:
- **Test Accuracy**: 99%+
- **Training Time**: ~5-10 minutes on CPU
- **Model Size**: ~2MB

## File Descriptions

- `data_loader.py`: Handles MNIST data loading, preprocessing, and visualization
- `model.py`: Defines CNN and simple model architectures with compilation
- `train.py`: Main training loop with callbacks and history plotting
- `evaluate.py`: Comprehensive model evaluation with multiple metrics
- `inference.py`: Prediction interface for new images with visualization

## Dependencies

- TensorFlow >= 2.12.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- Scikit-learn >= 1.1.0
- Pillow >= 9.0.0
- Seaborn >= 0.11.0# MNIST-Classifier
