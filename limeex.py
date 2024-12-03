
# Import necessary libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image  # Add PIL for image conversion
import cv2  # Add OpenCV for additional image processing
from tensorflow.keras.models import load_model
from preprocessing.patch_generator import smash_n_reconstruct
from classification import preprocess_single_image
from keras import layers, saving




def ensure_rgb_image(image_path):
    """
    Ensure the image is converted to RGB format
    """
    # Read image with OpenCV
    img = cv2.imread(image_path)
    
    # If image is grayscale, convert to RGB
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # If image is in BGR (OpenCV default), convert to RGB
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img



from skimage.transform import resize

def explain_prediction_lime(model, image_path):
    """
    Generate and visualize LIME explanations for a single image prediction.
    """
    # Preprocess the test image
    test_image = preprocess_single_image(image_path)
    test_data = [
        np.expand_dims(test_image['rich_texture'], axis=0),  # First input
        np.expand_dims(test_image['poor_texture'], axis=0)   # Second input
    ]
    
    # Convert original image to RGB
    original_image = ensure_rgb_image(image_path)
    
    # Wrapper function for model prediction
    def model_predict_wrapper(images):
        # Convert images to match model's expected input
        images_rich = images[:, :, :, 0:1]  # Extract rich texture channel
        images_poor = images[:, :, :, 1:2]  # Extract poor texture channel
        return model.predict([images_rich, images_poor])
    
    # Prepare image for LIME
    lime_image_data = np.stack([
        test_data[0].squeeze(),  # Rich texture
        test_data[1].squeeze(),  # Poor texture
        np.zeros_like(test_data[0].squeeze())  # Placeholder for third channel
    ], axis=-1)
    
    # Initialize the LimeImageExplainer
    explainer = lime_image.LimeImageExplainer()
    
    # Get explanation for the image
    explanation = explainer.explain_instance(
        image=lime_image_data,
        classifier_fn=model_predict_wrapper,
        top_labels=2,
        hide_color=0,
        num_samples=1000
    )
    
    # Select the top predicted label
    label_to_visualize = explanation.top_labels[0]
    
    # Create the explanation visualization
    temp, mask = explanation.get_image_and_mask(
        label=label_to_visualize,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    
    # Resize the mask to match the original image dimensions
    resized_mask = resize(mask, original_image.shape[:2], preserve_range=True, anti_aliasing=True)
    resized_mask = resized_mask.astype(bool)  # Ensure mask is boolean

    # Plot only the LIME explanation overlay
    plt.figure(figsize=(8, 8))  # Adjust size for better visibility
    plt.title(f'LIME Explanation (Label {label_to_visualize})')
    lime_overlay = mark_boundaries(original_image / 255.0, resized_mask, color=(1, 0, 0))
    plt.imshow(lime_overlay)
    plt.axis('off')  # Remove axis for a cleaner look
    plt.tight_layout()
    plt.show()
    
    # Model prediction
    prediction = model.predict(test_data)
    return label_to_visualize, prediction[0][0]





# Example model loading (replace 'your_model_path.h5' with actual model path)
model = load_model(r'C:\Users\esull\OneDrive\Documents\ds340_final_attempt\Detection-of-AI-generated-images\checkpoints\model_checkpoint1.keras')

# Example image path (replace 'example_image.jpg' with actual image path)
image_path = r'C:\Users\esull\OneDrive\Documents\ds340_final_attempt\Detection-of-AI-generated-images\data\evaluate\hiring-ai-artist-hyper-realistic-human-images-v0-njvnx7w1ykjc1.webp'

# Example usage of the LIME explanation function
label, prediction = explain_prediction_lime(model, image_path)
