import gen_point_set
from gen_greyscale import convert_to_greyscale
from PIL import Image
import numpy as np
import image_difference_score


def build_kernel_D_matrix(points, sigma1, sigma2, p):
    points = np.asarray(points)

    # Pairwise spatial squared distances: shape (n, n)
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff**2, axis=2)

    # Spatial kernel phi1. Use sigma1**2 to avoid an unnecessary sqrt.
    phi1 = np.exp(-(dist2 / (sigma1**2)))

    # Load greyscale image once and sample intensities for the points.
    grey_image = Image.open("working_images/grey.png")
    grey_arr = np.asarray(grey_image)

    intensities = grey_arr[points[:, 1], points[:, 0]]

    # Intensity differences and intensity kernel phi2
    intensity_diff = np.abs(intensities[:, None] - intensities[None, :]) ** p
    phi2 = np.exp(- (intensity_diff / sigma2) ** 2)

    # Combined kernel
    K_D = phi1 * phi2
    return K_D

def build_kernel_omega_matrix(points, sigma1, sigma2, p, height, width):
    n = points.shape[0]
    m = height * width
    k_omega = np.zeros((m, n))
    
    # Create all pixel coordinates
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    pixel_coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)
    # so pixel_coords is an m by 2 array of (x, y) coordinates for each pixel in the image

    grey_image = Image.open("working_images/grey.png")
    grey_pixels = np.array([grey_image.getpixel((x, y)) for x, y in pixel_coords])

    for i in range(n):
        # Vectorized distance calculation
        dist2 = np.sqrt(np.sum((pixel_coords - points[i])**2, axis=1))
        phi1 = np.exp(-(dist2 / sigma1)**2)
        
        # Vectorized gamma difference calculation
        grey_point = grey_image.getpixel((points[i][0], points[i][1]))
        grey_diff = np.abs(grey_pixels - grey_point)**p / sigma2
        phi2 = np.exp(-grey_diff**2)
        
        k_omega[:, i] = phi1 * phi2
        print(f"Processed point {i+1}/{n}")
    return k_omega

def solve_coefficients(points, colours, sigma1, sigma2, p, delta):
    """
    points : list of (i, j)
    colours : array of shape (n, 3) with RGB values
    """
    K_D = build_kernel_D_matrix(points, sigma1, sigma2, p)
    n = points.shape[0]
    A = np.linalg.solve(K_D + delta * np.eye(n), colours)
    return A

def colourise_image(height, width, points, colours, sigma1, sigma2, p, delta):
    print("Solving for coefficients...")
    A = solve_coefficients(points, colours, sigma1, sigma2, p, delta)

    # where A is an n by 3 matrix of coefficients for the R, G, B channels
    # form the kernel matrix K_omega for all pixels in the image a m by n matrix 
    # n is the number of points and m is the number of pixels in the image (height * width)
    K_omega = build_kernel_omega_matrix(points, sigma1, sigma2, p, height, width)

    F = K_omega @ A  # shape (height * width, 3)
    F = F.reshape((height, width, 3))

    # Clip values to [0, 255] and convert to uint8
    F = np.clip(F, 0, 255)
    return F.astype(np.uint8)



if __name__ == "__main__":
    # e.g. 
    N = 3  # Number of rows in the grid
    M = 3  # Number of columns in the grid
    test_image_path = "images/gradient2.png"

    colour_image = Image.open(test_image_path)
    colour_image.save('working_images/colour.png')

    convert_to_greyscale(test_image_path)

    points = gen_point_set.generate_regular_grid_points(test_image_path, N, M)
    #points = gen_point_set.generate_random_points(test_image_path, N*M)  # Generate N*M random points
    gen_point_set.gen_grey_and_colour_points(points) # not strictly necessary

    colourised = colourise_image(
        height=colour_image.height,
        width=colour_image.width,
        points=np.array([(x, y) for (x, y, rgb) in points]),
        colours=np.array([rgb for (x, y, rgb) in points]),
        sigma1 = 100.0,
        sigma2 = 100.0,
        p = 0.5,
        delta = 2.0e-4
    )

    colourised_image = Image.fromarray(colourised)
    colourised_image.save('working_images/colourised_image.png')

    # Calculate the image difference score
    frobenius_norm = image_difference_score.pixelwise_difference("working_images/colour.png", "working_images/colourised_image.png")
    print(f"Image Difference Score: {frobenius_norm}")
    ssim_index = image_difference_score.ssim_difference("working_images/colour.png", "working_images/colourised_image.png")
    print(f"SSIM Index: {ssim_index}")




