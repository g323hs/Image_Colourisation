import gen_point_set
from gen_greyscale import convert_to_greyscale
from PIL import Image
import numpy as np
import image_difference_score

def phi(z, method):
    if method == 'gaussian':
        return np.exp(-z**2) 
    elif method == 'wendland':
        return np.where(z <= 1.0, (1.0 - z)**4 * (4.0 * z + 1.0), 0.0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
def build_kernel_D_matrix(points, sigma1, sigma2, p, method):
    """Vectorized construction of the n-by-n kernel matrix K_D.

    The optional parameter `kernel` selects the spatial radial function applied to
    distances. Supported values: 'gaussian' (default) and 'wendland'.
    """
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    # Pairwise spatial squared distances: shape (n, n)
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff**2, axis=2)

    phi1 = phi(np.sqrt(dist2) / sigma1, method)

    # Load greyscale image once and sample intensities for the points.
    grey_image = Image.open("working_images/grey.png")
    grey_arr = np.asarray(grey_image)

    # Clip and convert to ints in case points are floats or out-of-bounds
    xs = np.clip(points[:, 0].astype(int), 0, grey_arr.shape[1] - 1)
    ys = np.clip(points[:, 1].astype(int), 0, grey_arr.shape[0] - 1)

    intensities = grey_arr[ys, xs]

    # Intensity differences and intensity kernel phi2
    intensity_diff = np.abs(intensities[:, None] - intensities[None, :]) ** p
    phi2 = phi(intensity_diff / sigma2, method)

    # Combined kernel
    K_D = phi1 * phi2
    return K_D

def build_kernel_omega_matrix(points, sigma1, sigma2, p, height, width, method):
    """Vectorized construction of the m-by-n kernel matrix K_omega.

    `method` selects the spatial radial function as in build_kernel_D_matrix.
    """
    n = points.shape[0]
    m = height * width
    k_omega = np.zeros((m, n))

    # Create all pixel coordinates (row-major ordering)
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    pixel_coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)

    # Load greyscale image once and flatten
    grey_image = Image.open("working_images/grey.png")
    grey_arr = np.asarray(grey_image)
    grey_pixels = grey_arr.ravel()

    # pixel_coords shape: (m, 2), points shape: (n, 2)
    pts = np.asarray(points)

    # Pairwise spatial squared distances: shape (m, n)
    diff = pixel_coords[:, None, :] - pts[None, :, :]
    dist2 = np.sum(diff**2, axis=2)
    phi1 = phi(np.sqrt(dist2) / sigma1, method)
    
    # Compute point intensities by indexing into the flattened grey_pixels array.
    xs = np.clip(pts[:, 0].astype(int), 0, width - 1)
    ys = np.clip(pts[:, 1].astype(int), 0, height - 1)
    point_idx = ys * width + xs
    point_intensities = grey_pixels[point_idx]

    # Intensity differences matrix (m, n) and intensity kernel phi2
    intensity_diff = np.abs(grey_pixels[:, None] - point_intensities[None, :]) ** p
    phi2 = phi(intensity_diff / sigma2, method)

    # Combined kernel (m, n)
    k_omega[:, :] = phi1 * phi2

    return k_omega

def solve_coefficients(points, colours, sigma1, sigma2, p, delta, method):
    K_D = build_kernel_D_matrix(points, sigma1, sigma2, p, method)
    n = points.shape[0]
    A = np.linalg.solve(K_D + delta * np.eye(n), colours)
    return A

def colourise_image(height, width, points, colours, sigma1, sigma2, p, delta, method='gaussian'):
    print("Solving for coefficients...")
    A = solve_coefficients(points, colours, sigma1, sigma2, p, delta, method)

    # form the kernel matrix K_omega for all pixels in the image
    print("Building kernel matrix K_omega for all pixels...")
    K_omega = build_kernel_omega_matrix(points, sigma1, sigma2, p, height, width, method)

    print("Applying kernel to coefficients to get colourised image...")
    F = K_omega @ A  # shape (height * width, 3)
    F = F.reshape((height, width, 3))

    # Clip values to [0, 255] and convert to uint8
    F = np.clip(F, 0, 255)
    return F.astype(np.uint8)


if __name__ == "__main__":
    # e.g. 
    N = 4  # Number of rows in the grid
    M = 4  # Number of columns in the grid
    test_image_path = "images/gradient1.png"

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
        delta = 2.0e-4, 
        method = 'wendland'
    )

    colourised_image = Image.fromarray(colourised)
    colourised_image.save('working_images/colourised_image_wendland.png')

    # Calculate the image difference score
    frobenius_norm = image_difference_score.pixelwise_difference("working_images/colour.png", "working_images/colourised_image_wendland.png")
    print(f"Image Difference Score: {frobenius_norm}")
    ssim_index = image_difference_score.ssim_difference("working_images/colour.png", "working_images/colourised_image_wendland.png")
    print(f"SSIM Index: {ssim_index}")

    colourised = colourise_image(
        height=colour_image.height,
        width=colour_image.width,
        points=np.array([(x, y) for (x, y, rgb) in points]),
        colours=np.array([rgb for (x, y, rgb) in points]),
        sigma1 = 100.0,
        sigma2 = 100.0,
        p = 0.5,
        delta = 2.0e-4, 
        method = 'gaussian'
    )

    colourised_image = Image.fromarray(colourised)
    colourised_image.save('working_images/colourised_image_gaussian.png')

    # Calculate the image difference score
    frobenius_norm = image_difference_score.pixelwise_difference("working_images/colour.png", "working_images/colourised_image_gaussian.png")
    print(f"Image Difference Score: {frobenius_norm}")
    ssim_index = image_difference_score.ssim_difference("working_images/colour.png", "working_images/colourised_image_gaussian.png")
    print(f"SSIM Index: {ssim_index}")




