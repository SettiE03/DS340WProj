import numpy as np
import tensorflow as tf
import shap
from tensorflow.keras.models import load_model
from preprocessing.patch_generator import smash_n_reconstruct
import matplotlib.pyplot as plt
from classification import preprocess_single_image
from keras import layers, saving

def prepare_background_data(num_samples=10):
    background_data_rich = []
    background_data_poor = []
    
    # Paths to some representative background images
    background_paths = [
        r'./data/ai_images/6-236710992-255071.jpg',
        # ... additional paths
    ]
    
    for path in background_paths[:num_samples]:
        rich_texture, poor_texture = smash_n_reconstruct(path, coloured=False)
        rich_texture = np.expand_dims(rich_texture, axis=-1).astype(np.float32) / 255.0
        poor_texture = np.expand_dims(poor_texture, axis=-1).astype(np.float32) / 255.0
        
        background_data_rich.append(rich_texture)
        background_data_poor.append(poor_texture)
    
    return np.stack(background_data_rich), np.stack(background_data_poor)


def explain_prediction(model, image_path, num_background_samples=10):
    """
    Generate and visualize SHAP explanations for a single image prediction.
    """
    # Prepare the test image as a list of inputs
    test_image = preprocess_single_image(image_path)
    test_data = [
        np.expand_dims(test_image['rich_texture'], axis=0),  # First input
        np.expand_dims(test_image['poor_texture'], axis=0)   # Second input
    ]
    
    print(f"Test image shapes: rich_texture: {test_data[0].shape}, poor_texture: {test_data[1].shape}")
    
    # Prepare background data
    background_data = prepare_background_data(num_background_samples)
    print(f"Background data shapes: rich_texture: {background_data[0].shape}, poor_texture: {background_data[1].shape}")
    
    # Create explainer
    explainer = shap.DeepExplainer(model, [background_data[0], background_data[1]])
    
    # Calculate SHAP values with the multi-input format
    shap_values = explainer.shap_values(test_data)
    
    # Process SHAP values for visualization
    shap_values_rich = shap_values[0][0]
    shap_values_poor = shap_values[1][0]
    
    abs_shap_values_rich = np.abs(shap_values_rich)
    abs_shap_values_poor = np.abs(shap_values_poor)
    max_value = np.percentile(np.concatenate([abs_shap_values_rich, abs_shap_values_poor]), 99)
    
    plt.figure(figsize=(12, 6))
    
    # Original rich_texture
    plt.subplot(2, 3, 1)
    plt.title('Original Rich Texture')
    plt.imshow(test_data[0][0, :, :, 0], cmap='gray')
    plt.axis('off')
    
    # SHAP values for rich_texture
    plt.subplot(2, 3, 2)
    plt.title('SHAP Values (Rich Texture)')
    plt.imshow(shap_values_rich[:, :, 0], cmap='RdBu', vmin=-max_value, vmax=max_value)
    plt.colorbar()
    plt.axis('off')
    
    # Absolute SHAP values for rich_texture
    plt.subplot(2, 3, 3)
    plt.title('Absolute SHAP Values (Rich Texture)')
    plt.imshow(abs_shap_values_rich[:, :, 0], cmap='hot', vmax=max_value)
    plt.colorbar()
    plt.axis('off')
    
    # Original poor_texture
    plt.subplot(2, 3, 4)
    plt.title('Original Poor Texture')
    plt.imshow(test_data[1][0, :, :, 0], cmap='gray')
    plt.axis('off')
    
    # SHAP values for poor_texture
    plt.subplot(2, 3, 5)
    plt.title('SHAP Values (Poor Texture)')
    plt.imshow(shap_values_poor[:, :, 0], cmap='RdBu', vmin=-max_value, vmax=max_value)
    plt.colorbar()
    plt.axis('off')
    
    # Absolute SHAP values for poor_texture
    plt.subplot(2, 3, 6)
    plt.title('Absolute SHAP Values (Poor Texture)')
    plt.imshow(abs_shap_values_poor[:, :, 0], cmap='hot', vmax=max_value)
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('shap_explanation_1.png')
    plt.close()
    
    # Model prediction
    prediction = model.predict(test_data)
    return shap_values, prediction[0][0]


if __name__ == "__main__":
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

    # Load the model
    model = load_model('./classifier.keras', custom_objects={'hard_tanh': hard_tanh, 'featureExtractionLayer': featureExtractionLayer})

    # Example usage
    test_image_path = './data/ai_images/6-236710992-255071.jpg'
    shap_values, prediction = explain_prediction(model, test_image_path)
