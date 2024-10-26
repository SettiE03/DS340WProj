import tensorflow as tf
import numpy as np
import os

# Your special processing method
def smash_n_reconstruct(filepath):
    # Actual logic to read and process the image
    # Replace with your real image processing logic
    rich_texture = np.random.rand(256, 256).astype(np.float32)  # Mock data for rich texture
    poor_texture = np.random.rand(256, 256).astype(np.float32)  # Mock data for poor texture
    return rich_texture, poor_texture

def preprocess(filepath):
    # Load the image using the special processing method
    rich_texture, poor_texture = smash_n_reconstruct(filepath)
    
    # Ensure the outputs are grayscale images of shape (256, 256, 1)
    rich_texture = np.expand_dims(rich_texture, axis=-1)  # Shape: (256, 256, 1)
    poor_texture = np.expand_dims(poor_texture, axis=-1)  # Shape: (256, 256, 1)

    # Convert to tf.float32 for TensorFlow
    rich_texture = tf.convert_to_tensor(rich_texture, dtype=tf.float32)
    poor_texture = tf.convert_to_tensor(poor_texture, dtype=tf.float32)

    return rich_texture, poor_texture

# Load your image paths and labels
path_ai = r'C:\Users\esull\OneDrive\Documents\ds340_attempt3_shap\Detection-of-AI-generated-images\data\ai_images'
path_real = r'C:\Users\esull\OneDrive\Documents\ds340_attempt3_shap\Detection-of-AI-generated-images\data\real_images'

# Load AI images
ai_imgs = [os.path.join(path_ai, img) for img in os.listdir(path_ai)]
ai_labels = [1] * len(ai_imgs)  # 1 for AI-generated images

# Load real images
real_imgs = [os.path.join(path_real, img) for img in os.listdir(path_real)]
real_labels = [0] * len(real_imgs)  # 0 for real images

# Combine image paths and labels
X_train = ai_imgs + real_imgs
y_train = ai_labels + real_labels

# Shuffle the dataset
combined = list(zip(X_train, y_train))
np.random.shuffle(combined)
X_train[:], y_train[:] = zip(*combined)

# Preprocess all images
X_processed = []
for filepath in X_train:
    rich_texture, poor_texture = preprocess(filepath)
    # Only appending rich textures for the training set
    X_processed.append(rich_texture.numpy())  # Convert tensor to numpy array for the dataset

# Convert to numpy array and check the shape
X_processed = np.array(X_processed)

# Check the shape of processed data
print(f"Processed training data shape: {X_processed.shape}")  # Expecting (num_samples, 256, 256, 1)

# Convert labels to numpy array
y_train = np.array(y_train)

# Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(256, 256, 1)),  # Input shape for grayscale images
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fit the model
model.fit(X_processed, y_train, epochs=5, batch_size=2, validation_split=0.2)

# Save the model
model_save_path = "./checkpoints/my_model.keras"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
