import PIL.Image
import cv2
import numpy as np
import random

def img_to_patches(input_path: str) -> tuple:
    img = PIL.Image.open(fp=input_path)
    img = img.convert('RGB')
    img = img.resize((256, 256))  # Force resize to 256x256
        
    patch_size = 32
    grayscale_imgs = []
    imgs = []
    
    for i in range(0, img.height, patch_size):
        for j in range(0, img.width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            img_color = np.asarray(img.crop(box))
            grayscale_image = cv2.cvtColor(src=img_color, code=cv2.COLOR_RGB2GRAY)
            grayscale_imgs.append(grayscale_image.astype(dtype=np.float32))
            imgs.append(img_color)

    print(f"Generated {len(grayscale_imgs)} grayscale patches and {len(imgs)} color patches.")
    return grayscale_imgs, imgs

def get_pixel_var_degree_for_patch(patch: np.ndarray) -> float:
    """Calculate the variance of a given patch."""
    return np.var(patch)

def extract_rich_and_poor_textures(variance_values: list, patches: list) -> tuple:
    """Separates patches into rich and poor textures based on variance."""
    # Assuming higher variance = rich texture
    threshold = np.median(variance_values)
    rich_patches = [patches[i] for i in range(len(patches)) if variance_values[i] > threshold]
    poor_patches = [patches[i] for i in range(len(patches)) if variance_values[i] <= threshold]
    return rich_patches, poor_patches

def get_complete_image(patches: list, coloured=True) -> np.ndarray:
    """
    Develops a complete 256x256 image from texture patches.
    """
    required_patches = 64  # 8x8 grid of 32x32 patches
    
    # If we have more patches than needed, randomly select required number
    if len(patches) > required_patches:
        patches = random.sample(patches, required_patches)
    
    # If we have fewer patches than needed, duplicate random patches
    while len(patches) < required_patches:
        patches.append(random.choice(patches))
    
    # Convert to numpy array
    patches_array = np.array(patches)
    
    # Reshape based on whether the patches are colored or grayscale
    if coloured:
        grid = patches_array.reshape((8, 8, 32, 32, 3))  # For colored patches
    else:
        grid = patches_array.reshape((8, 8, 32, 32))  # For grayscale patches

    # Join patches together
    rows = [np.concatenate(grid[i,:], axis=1) for i in range(8)]
    final_image = np.concatenate(rows, axis=0)
    
    return final_image

def smash_n_reconstruct(input_path: str, coloured=False) -> tuple:
    """
    Process image and return rich and poor texture reconstructions.
    """
    # Get patches
    gray_scale_patches, color_patches = img_to_patches(input_path=input_path)
    
    # Calculate variance for each patch
    pixel_var_degree = [get_pixel_var_degree_for_patch(patch) for patch in gray_scale_patches]
    
    # Extract rich and poor textures
    if coloured:
        r_patch, p_patch = extract_rich_and_poor_textures(
            variance_values=pixel_var_degree, 
            patches=color_patches
        )
    else:
        r_patch, p_patch = extract_rich_and_poor_textures(
            variance_values=pixel_var_degree, 
            patches=gray_scale_patches
        )

    # Reconstruct complete images
    rich_texture = get_complete_image(r_patch, coloured)
    poor_texture = get_complete_image(p_patch, coloured)

    # Ensure correct output shape (256, 256, 1) for grayscale
    if not coloured:
        rich_texture = np.expand_dims(rich_texture, axis=-1)
        poor_texture = np.expand_dims(poor_texture, axis=-1)
    
    # Normalize to [0, 1] range
    rich_texture = rich_texture.astype(np.float32) / 255.0
    poor_texture = poor_texture.astype(np.float32) / 255.0

    print(f"Rich Texture Shape: {rich_texture.shape}, Poor Texture Shape: {poor_texture.shape}")
    
    return rich_texture, poor_texture

if __name__=="main":
    smash_n_reconstruct(input_path="placeholder")