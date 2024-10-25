import numpy as np
import tensorflow as tf
import shap
from tensorflow.keras.models import load_model
from preprocessing.patch_generator import smash_n_reconstruct  # Using your existing texture analysis
import matplotlib.pyplot as plt

def prepare_background_data(num_samples=10):
    """
    Prepare background data for SHAP analysis using both AI and real images.
    """
    background_data = []
    
    # Add paths to some representative background images
    background_paths = [
        # Add paths to 5 real and 5 AI images that represent your dataset well
        r'C:\Users\esull\OneDrive\Documents\ds340_attempt3_shap\Detection-of-AI-generated-images\data\evaluate\7-103192408-508471_ai_2.jpg',
        # ... add more paths
    ]
    
    for path in background_paths[:num_samples]:
        rich_texture, _ = smash_n_reconstruct(path, coloured=False)
        # Ensure correct shape and type
        rich_texture = rich_texture.astype(np.float32) / 255.0  # Normalize to [0,1]
        if rich_texture.shape != (256, 256, 1):
            rich_texture = np.reshape(rich_texture, (256, 256, 1))
        background_data.append(rich_texture)
    
    return np.stack(background_data)

def preprocess_single_image(image_path):
    """
    Preprocess a single image for prediction and SHAP analysis.
    """
    # Use your existing texture analysis
    rich_texture, _ = smash_n_reconstruct(image_path, coloured=False)
    
    # Ensure correct shape and type
    rich_texture = rich_texture.astype(np.float32) / 255.0  # Normalize to [0,1]
    if rich_texture.shape != (256, 256, 1):
        rich_texture = np.reshape(rich_texture, (256, 256, 1))
    
    return np.expand_dims(rich_texture, axis=0)  # Add batch dimension

def explain_prediction(model_path, image_path, num_background_samples=10):
    """
    Generate and visualize SHAP explanations for a single image prediction.
    """
    # Load the trained model
    model = load_model(model_path)
    
    # Prepare the test image
    test_image = preprocess_single_image(image_path)
    print(f"Test image shape: {test_image.shape}")
    
    # Prepare background data
    background_data = prepare_background_data(num_background_samples)
    print(f"Background data shape: {background_data.shape}")
    
    # Create explainer
    explainer = shap.DeepExplainer(model, background_data)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(test_image)
    
    # Handle different return types from shap_values
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Prepare for visualization
    # For binary classification, we typically want to show the impact on the positive class
    abs_shap_values = np.abs(shap_values)
    max_value = np.percentile(abs_shap_values, 99)
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(test_image[0, :, :, 0], cmap='gray')
    plt.axis('off')
    
    # SHAP values
    plt.subplot(2, 2, 2)
    plt.title('SHAP Values')
    plt.imshow(shap_values[0, :, :, 0], cmap='RdBu', vmin=-max_value, vmax=max_value)
    plt.colorbar()
    plt.axis('off')
    
    # Absolute SHAP values
    plt.subplot(2, 2, 3)
    plt.title('Absolute SHAP Values')
    plt.imshow(abs_shap_values[0, :, :, 0], cmap='hot', vmax=max_value)
    plt.colorbar()
    plt.axis('off')
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig('shap_explanation_3.png')
    plt.close()
    
    # Return the SHAP values and prediction
    prediction = model.predict(test_image)[0][0]
    return shap_values, prediction

if __name__ == "__main__":
    # Set paths
    MODEL_PATH = "./checkpoints/my_model.keras"
    TEST_IMAGE_PATH = r'C:\Users\esull\OneDrive\Documents\ds340_attempt3_shap\Detection-of-AI-generated-images\data\evaluate\7-103192408-508471_ai_2.jpg'
    
    # Generate explanation
    shap_values, prediction = explain_prediction(
        model_path=MODEL_PATH,
        image_path=TEST_IMAGE_PATH,
        num_background_samples=10
    )
    
    print(f"Prediction: {prediction:.3f}")
    print(f"SHAP values shape: {shap_values.shape}")
    print("Explanation has been saved as 'shap_explanation_3.png'")