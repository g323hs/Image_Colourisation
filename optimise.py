from gen_colourised import colourise_process
import numpy as np
import image_difference_score
import time
import matplotlib.pyplot as plt
from gen_images import generate_images
from scipy.optimize import minimize
from image_difference_score import pixelwise_difference
from scale_image_to_size import scale_image_to_size

def compare_N_to_time(image_path, save_path_folder):
    # find out how the number of points affects the time taken 
    times = []
    points = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    for N in points:
        start_time = time.time()
        colourise_process(image_path, save_path_folder, n=N, m=1, point_gen_method='random', sigma1=50, sigma2=100, p=0.5, delta=2.0e-10, phi_method='gaussian')
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Time taken for {N} points: {end_time - start_time:.2f} seconds")

    plt.plot(points, times, marker='o')
    plt.xlabel('Number of Points (N)')
    plt.ylabel('Time Taken (seconds)')
    plt.title('Time Taken vs Number of Points')
    plt.grid()
    plt.savefig('optimisation_results/time_vs_colour_points_given.png')
    plt.show()

def compare_image_size_to_time(image_path, save_path_folder):
    # find out how the image size affects the time taken 
    times = []
    sizes = [(100, 100), (200, 200), (400, 400), (800, 800), (1600, 1600)]
    for height, width in sizes:
        generate_images(height, width)
        start_time = time.time()
        colourise_process(image_path, save_path_folder, n=10, m=10, point_gen_method='grid', sigma1=50, sigma2=100, p=0.5, delta=2.0e-10, phi_method='gaussian')
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Time taken for {height}x{width} image: {end_time - start_time:.2f} seconds")

    plt.plot([h for h,w in sizes], times, marker='o')
    plt.xlabel('Image Size (h = w)')
    plt.ylabel('Time Taken (seconds)')
    plt.title('Time Taken vs Image Size')
    plt.grid()
    plt.savefig('optimisation_results/time_vs_image_size.png')
    plt.show()

def get_ssim_score(params, image_path, save_path_folder, phi_method='gaussian'):
    sigma1, sigma2, p, delta = params
    colourise_process(image_path, save_path_folder, n=20, m=20, point_gen_method='random', sigma1=sigma1, sigma2=sigma2, p=p, delta=delta, phi_method=phi_method)
    ssim_score = image_difference_score.ssim_difference(image_path, f"{save_path_folder}/colourised.png")
    return -ssim_score  # we want to maximise ssim so we return the negative

if __name__ == "__main__":
    # generate_images(100, 100)  # generate a 100x100 test image
    image_path = "images/IMG_6006.jpg"
    save_path_folder = "optimisation_results"

    scale_image_to_size(image_path, f"{save_path_folder}/colour.png", target_size=(100, 100))
    image_path = f"{save_path_folder}/colour.png"


    # compare_N_to_time(image_path, save_path_folder)
    # compare_image_size_to_time(image_path, save_path_folder)

    phi_methods = ['gaussian', 'wendland']
    ssim_scores = []
    for phi_method in phi_methods:
        initial_guess = [50, 150, 0.08, 2.0e-8]
        bounds = [(1, 100), (1, 2000), (0.01, 1), (1e-12, 1e-8)]
        result = minimize(get_ssim_score,
                          initial_guess,
                          args=(image_path, save_path_folder, phi_method),
                          bounds=bounds,
                          method='Nelder-Mead',
                          options={'maxiter': 20, 'xatol': 1e-3, 'fatol': 1e-3})
        print(result.success, result.message)
        ssim_scores.append([-result.fun, result.x])

    # select the best phi method based on the highest ssim score
    best_phi_method_index = np.argmax([score[0] for score in ssim_scores])
    best_phi_method = phi_methods[best_phi_method_index]
    best_params = ssim_scores[best_phi_method_index][1]
    print(f"Best phi parameters: {best_params} for phi method: {best_phi_method}")
    print(f"Best SSIM score: {ssim_scores[best_phi_method_index][0]}")

    colourise_process(image_path, save_path_folder, n=20, m=20, point_gen_method='random', sigma1=best_params[0], sigma2=best_params[1], p=best_params[2], delta=best_params[3], phi_method=best_phi_method, debug=False)

    # calculate the pixelwise difference between the original and the colourised image
    pixelwise_diff = pixelwise_difference(image_path, f"{save_path_folder}/colourised.png")
    print(f"Pixelwise difference: {pixelwise_diff}")