import gen_colourised
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gen_greyscale import convert_to_greyscale
import gen_point_set
import image_difference_score
import os

if __name__ == "__main__":


    test_image_path = "images/gradient2.png"

    colour_image = Image.open(test_image_path)
    colour_image.save('working_images/colour.png')

    convert_to_greyscale(test_image_path)

    frob_scores = []
    ssim_scores = []
    grid_sizes = []

    # empy the folder called series_of_images
    for filename in os.listdir('series_of_images'): 
        file_path = os.path.join('series_of_images', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for k in range(1, 10 + 1):
        N = k
        M = k
        print(f"Running for N = M = {k}")

        # points = gen_point_set.generate_regular_grid_points(test_image_path, N, M)
        points = gen_point_set.generate_random_points(test_image_path, N*M, seed=1)

        colourised = gen_colourised.colourise_image(
            height=colour_image.height,
            width=colour_image.width,
            points=np.array([(x, y) for (x, y, rgb) in points]),
            colours=np.array([rgb for (x, y, rgb) in points]),
            sigma1=100.0,
            sigma2=100.0,
            p=0.5,
            delta=2.0e-4, 
            method='gaussian'
        )

        colourised_image = Image.fromarray(colourised)
        colourised_image.save(f'series_of_images/colourised_image_{k}.png')

        frobenius_norm = image_difference_score.pixelwise_difference(
            "working_images/colour.png",
            f"series_of_images/colourised_image_{k}.png"
        )

        ssim_index = image_difference_score.ssim_difference(
            "working_images/colour.png",
            f"series_of_images/colourised_image_{k}.png"
        )

        frob_scores.append(frobenius_norm)
        ssim_scores.append(ssim_index)
        grid_sizes.append(k)

    # Plotting the results on the same graph but with two y-axes
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Grid Size (N = M)')
    ax1.set_ylabel('Frobenius Norm', color=color)
    ax1.plot(grid_sizes, frob_scores, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('SSIM Index', color=color)  # we already handled the x-label with ax1
    ax2.plot(grid_sizes, ssim_scores, marker='o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Image Difference Scores vs Grid Size')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
