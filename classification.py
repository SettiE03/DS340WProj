import tensorflow as tf
import numpy as np
from preprocessing.patch_generator import smash_n_reconstruct
import preprocessing.filters as f
from keras import layers, saving

# Function to preprocess a single image for testing
'''
def preprocess_single_image(filepath):
    # Load and preprocess the image, including smash_n_reconstruct
    rich_texture, poor_texture = smash_n_reconstruct(filepath)
    
    # Apply any additional transformations or filters if needed
    frt = tf.cast(tf.expand_dims(f.apply_all_filters(rich_texture), axis=-1), dtype=tf.float32)
    fpt = tf.cast(tf.expand_dims(f.apply_all_filters(poor_texture), axis=-1), dtype=tf.float32)
    
    # Prepare inputs in the required dictionary format
    return {'rich_texture': frt, 'poor_texture': fpt}
'''
import cv2
import numpy as np
import tensorflow as tf

def preprocess_single_image(filepath):
    # Process the original image to grayscale, resized, and normalized
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Directly load as grayscale
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=-1)  # Shape becomes (256, 256, 1)
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Apply texture processing with smash_n_reconstruct
    rich_texture, poor_texture = smash_n_reconstruct(filepath)

    # Convert both outputs to single-channel grayscale if needed
    if len(rich_texture.shape) == 3 and rich_texture.shape[-1] == 3:
        rich_texture = cv2.cvtColor(rich_texture, cv2.COLOR_BGR2GRAY)
    if len(poor_texture.shape) == 3 and poor_texture.shape[-1] == 3:
        poor_texture = cv2.cvtColor(poor_texture, cv2.COLOR_BGR2GRAY)

    # Resize and normalize the textures to be consistent with the model input
    rich_texture = cv2.resize(rich_texture, (256, 256))
    poor_texture = cv2.resize(poor_texture, (256, 256))
    rich_texture = np.expand_dims(rich_texture, axis=-1).astype(np.float32) / 255.0
    poor_texture = np.expand_dims(poor_texture, axis=-1).astype(np.float32) / 255.0

    # Convert to tensors expected by the model
    rich_texture = tf.convert_to_tensor(rich_texture, dtype=tf.float32)
    poor_texture = tf.convert_to_tensor(poor_texture, dtype=tf.float32)

    return {
        'rich_texture': rich_texture,
        'poor_texture': poor_texture
    }

# Function to make a prediction
def predict_image(model, filepath):
    # Preprocess the image
    processed_image = preprocess_single_image(filepath)
    
    # Add a batch dimension to the inputs
    input_data = {key: tf.expand_dims(value, axis=0) for key, value in processed_image.items()}
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Convert the prediction to a percentage
    probability = prediction[0][0] * 100  # Convert to percentage
    
    # Interpret and print the result with probability
    if prediction[0][0] > 0.5:
        print(f"Prediction: AI-generated image with {probability:.2f}% confidence")
    else:
        print(f"Prediction: Real image with {100 - probability:.2f}% confidence")


def hard_tanh(x):
    return tf.maximum(tf.minimum(x, 1), -1)

@saving.register_keras_serializable(package="Custom")
class featureExtractionLayer(layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')
        self.bn = layers.BatchNormalization()
        self.activation = layers.Lambda(hard_tanh)
    '''    
    def call(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.activation(x)
        return x
    '''
    def call(self, input):
        assert input.shape[-1] == 1, f"Expected input with 1 channel, but got {input.shape[-1]} channels"
        x = self.conv(input)
        x = self.bn(x)
        x = self.activation(x)
        return x
# Load the model from the file path
model = tf.keras.models.load_model('./classifier.keras', custom_objects={'hard_tanh': hard_tanh, 'featureExtractionLayer': featureExtractionLayer})

# Example usage
test_image_path = r'C:\Users\esull\OneDrive\Documents\ds340_final_attempt\Detection-of-AI-generated-images\data\evaluate\7-101055791-560690_ai.jpg'
predict_image(model, test_image_path)
