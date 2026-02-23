from gen_colourised import colourise_process
import numpy as np
import image_difference_score
import time
import matplotlib.pyplot as plt
from gen_images import generate_images
from scipy.optimize import minimize
from image_difference_score import pixelwise_difference
from scale_image_to_size import scale_image_to_size


def get_ssim_score_ish(params, image_path, save_path_folder, phi_method='gaussian', n=1, m=1):
    sigma1, sigma2, p, delta = params
    colourise_process(image_path, save_path_folder, n, m, point_gen_method='random', sigma1=sigma1, sigma2=sigma2, p=p, delta=delta, phi_method=phi_method)
    ssim_score = image_difference_score.ssim_difference(image_path, f"{save_path_folder}/colourised.png")
    return 1-ssim_score  # to match the pixelwise difference where we want to minimise the score to 0 being the best

def get_pixelwise_difference(params, image_path, save_path_folder, phi_method='gaussian', n=1, m=1):
    sigma1, sigma2, p, delta = params
    colourise_process(image_path, save_path_folder, n, m, point_gen_method='random', sigma1=sigma1, sigma2=sigma2, p=p, delta=delta, phi_method=phi_method)
    pixel_diff = pixelwise_difference(image_path, f"{save_path_folder}/colourised.png")
    return pixel_diff

if __name__ == "__main__":
    generate_images(1000, 1000)  # generate a 100x100 test image
    #image_path = "images/IMG_6006.jpg"
    image_path = "images/complex_pattern.png"
    save_path_folder = "optimisation_comparison"

    scale_image_to_size(image_path, f"{save_path_folder}/colour.png", target_size=(100, 100))
    image_path = f"{save_path_folder}/colour.png"

    num_points = 1000

    initial_guess = [20, 100, 0.5, 2.0e-10]
    #bounds = [(1, 1000), (0.001, 2000), (0.1, 1.0), (1e-12,1e-6)]

    
    phi_methods = ['gaussian', 'wendland']
    score_functions = [get_ssim_score_ish, get_pixelwise_difference]

    scores = []

    for score_function in score_functions:
        for phi_method in phi_methods:
            print(f"Optimising for score function: {score_function.__name__} and phi method: {phi_method}")
            result = minimize(score_function,
                              initial_guess,
                              args=(image_path, save_path_folder, phi_method, num_points),
                              #bounds=bounds,
                              method='Nelder-Mead',
                              options={'maxiter': 15, 'xatol': 1e-4, 'fatol': 1e-4})

            
            scores.append([score_function.__name__, phi_method, -result.fun, result.x])

            colourise_process(image_path, f"{save_path_folder}/{score_function.__name__}_{phi_method}", n=num_points, m=1, point_gen_method='random', sigma1=result.x[0], sigma2=result.x[1], p=result.x[2], delta=result.x[3], phi_method=phi_method, debug=False)

            print(f"Score for {score_function.__name__} and {phi_method}: {-result.fun} with parameters: {result.x}")