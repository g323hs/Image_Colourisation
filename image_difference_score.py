# in this script we implement different ways to calculate the difference between two images, we will use this to compare the original colour image with the reconstructed colour image to see how well our reconstruction is doing. We will implement two different methods to calculate the difference, the first will be a simple pixel-wise difference, the second will be a structural similarity index (SSIM). We will also implement a function to display the original image, the reconstructed image and the difference image side by side for visual comparison.

from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Method 1: Pixel-wise difference - frobenius norm
def pixelwise_difference(image1_path, image2_path):
    """Calculate the pixel-wise difference between two images."""
    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')
    
    # Convert images to numpy arrays
    arr1 = np.array(image1)
    arr2 = np.array(image2)

    difference = np.linalg.norm(arr1 - arr2)
    
    return difference

def ssim_difference(image1_path, image2_path):
    """Calculate the structural similarity index (SSIM) between two images."""
    image1 = Image.open(image1_path).convert('L')  # Convert to greyscale
    image2 = Image.open(image2_path).convert('L')  # Convert to greyscale
    
    # Convert images to numpy arrays
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    
    # Calculate SSIM
    ssim_index, _ = ssim(arr1, arr2, full=True)
    
    return ssim_index

if __name__ == "__main__":

    test = "images/flag1.png"

    test1 = Image.open(test)
    test1.save('differences/test1.png')

    test2 = Image.open(test)

    # put a black pixel in the middle of the image
    width, height = test2.size
    test2.putpixel((width // 2, height // 2), (0, 0, 0))
    test2.save('differences/test2.png')


    # Calculate pixel-wise difference
    diff = pixelwise_difference('differences/test1.png', 'differences/test2.png')
    print(f"Pixel-wise difference (Frobenius norm) between test1 and test2: {diff}")

    # Calculate SSIM difference
    ssim_diff = ssim_difference('differences/test1.png', 'differences/test2.png')
    print(f"SSIM difference between test1 and test2: {ssim_diff}")