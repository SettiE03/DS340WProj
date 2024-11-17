import numpy as np
import tensorflow as tf
import shap
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocessing.patch_generator import smash_n_reconstruct  # Custom texture analysis
from classification import preprocess_single_image  # Assumed preprocessing function
from keras import layers, saving

# Define custom feature extraction layer if needed for model loading
def hard_tanh(x):
    return tf.maximum(tf.minimum(x, 1), -1)

@saving.register_keras_serializable(package="Custom")
class featureExtractionLayer(layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')
        self.bn = layers.BatchNormalization()
        self.activation = layers.Lambda(hard_tanh)
        
    def call(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.activation(x)
        return x

# Function to load model with custom layers
def load_custom_model(model_path):
    return load_model(model_path, custom_objects={'hard_tanh': hard_tanh, 'featureExtractionLayer': featureExtractionLayer})

# Function to prepare background data for SHAP
def prepare_background_data(num_samples=10):
    background_data_rich = []
    background_data_poor = []
    
    # Define representative images for SHAP background
    background_paths = [
        r'./data/ai_images/6-236710992-255071.jpg',
        # Add more paths as needed for 5 real and 5 AI images
    ]
    
    for path in background_paths[:num_samples]:
        rich_texture, poor_texture = smash_n_reconstruct(path, coloured=False)
        
        # Ensure single-channel format and normalization
        rich_texture = cv2.resize(rich_texture, (256, 256))
        poor_texture = cv2.resize(poor_texture, (256, 256))
        rich_texture = np.expand_dims(rich_texture, axis=-1).astype(np.float32) / 255.0
        poor_texture = np.expand_dims(poor_texture, axis=-1).astype(np.float32) / 255.0
        
        background_data_rich.append(rich_texture)
        background_data_poor.append(poor_texture)
    
    return np.stack(background_data_rich), np.stack(background_data_poor)

# Function to generate SHAP explanation and overlay on original image
def explain_prediction_with_overlay(model, image_path, num_background_samples=10):
    # Load and preprocess the image
    original_image = cv2.imread(image_path)
    test_image = preprocess_single_image(image_path)
    
    # Prepare the test data as a list of inputs
    test_data = [
        np.expand_dims(test_image['rich_texture'], axis=0),  # First input
        np.expand_dims(test_image['poor_texture'], axis=0)   # Second input
    ]
    
    # Prepare background data for SHAP
    background_data = prepare_background_data(num_background_samples)
    
    # Create SHAP explainer
    explainer = shap.DeepExplainer(model, [background_data[0], background_data[1]])
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(test_data)
    shap_values_rich = shap_values[0][0]
    shap_values_poor = shap_values[1][0]
    
    # Resize SHAP values to match original image's dimensions
    shap_values_rich_resized = cv2.resize(shap_values_rich[:, :, 0], (original_image.shape[1], original_image.shape[0]))
    shap_values_poor_resized = cv2.resize(shap_values_poor[:, :, 0], (original_image.shape[1], original_image.shape[0]))

    # Normalize SHAP values for visualization
    max_val = max(np.abs(shap_values_rich_resized).max(), np.abs(shap_values_poor_resized).max())
    shap_values_rich_normalized = shap_values_rich_resized / max_val
    shap_values_poor_normalized = shap_values_poor_resized / max_val

    # Overlay SHAP values on the original image with a color map
    plt.figure(figsize=(10, 10))
    
    # Rich texture overlay
    plt.subplot(1, 2, 1)
    plt.title("Rich Texture SHAP Overlay")
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.imshow(shap_values_rich_normalized, cmap="RdBu", alpha=0.5, vmin=-1, vmax=1)
    plt.axis("off")
    
    # Poor texture overlay
    plt.subplot(1, 2, 2)
    plt.title("Poor Texture SHAP Overlay")
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.imshow(shap_values_poor_normalized, cmap="RdBu", alpha=0.5, vmin=-1, vmax=1)
    plt.axis("off")
    
    # Save the figure to a file and show it
    plt.tight_layout()
    plt.savefig('shap_overlay.png')
    plt.show()
    
    print("SHAP explanation figure saved as 'shap_overlay.png'.")
    
    # Return the SHAP values and prediction
    prediction = model.predict(test_data)
    return shap_values, prediction[0][0]

if __name__ == "__main__":
    # Load the model with custom layers
    model_path = './classifier.keras'
    model = load_custom_model(model_path)

    # Example usage
    test_image_path = r'./data/ai_images/6-236710992-255071.jpg'
    shap_values, prediction = explain_prediction_with_overlay(model, test_image_path)
    print(f"Prediction: {'AI-generated' if prediction > 0.5 else 'Real image'} with confidence {prediction * 100:.2f}%")
