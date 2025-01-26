import numpy as np


# Function to apply an occlusion attack
def apply_occlusion(image, occlusion_size=(50, 50), position=(100, 100)):
    occluded_image = image.copy()
    x, y = position
    width, height = occlusion_size
    occluded_image[y : y + height, x : x + width] = (
        0  # Black out the region return occluded_image
    )

    return occluded_image


def add_gaussian_noise(image, mean=0, var=0.01):
    sigma = var
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gauss * 255, 0, 255).astype(image.dtype)
    return noisy_image
