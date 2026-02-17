from PIL import Image
from gen_greyscale import convert_to_greyscale

# Method 1: Regular grid of points
def generate_regular_grid_points(image_path, N, M):
    """Generate a regular grid of points from the image."""
    image = Image.open(image_path)
    width, height = image.size
    points = []
    for i in range(1,N+1):
        for j in range(1,M+1):
            x = int(i * width / (N + 1))  # Calculate x coordinate
            y = int(j * height / (M + 1))  # Calculate y coordinate
            rgb = image.getpixel((x, y))  # Get RGB values at the point
            points.append((x, y, rgb))
    return points

# Method 2: Random points
def generate_random_points(image_path, num_points, seed = 1):
    """Generate a specified number of random points from the image."""
    import random
    image = Image.open(image_path)
    width, height = image.size
    points = []
    random.seed(seed)
    count = 0
    while count < num_points:
        x = random.randint(0, width - 1)  # Random x coordinate
        y = random.randint(0, height - 1)  # Random y coordinate
        rgb = image.getpixel((x, y))  # Get RGB values at the point
        if (x, y, rgb) not in points:  # Avoid duplicates
            points.append((x, y, rgb))
            count += 1
    return points

# Method 3: User-specified points with cursor
def generate_user_specified_points(image_path):
    """Generate points specified by the user with a cursor."""
    import matplotlib.pyplot as plt
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title("Click to select points")
    selected_points = plt.ginput(n=-1)  # PRESS ENTER when done
    plt.close()
    
    # Get RGB values for the selected points
    points = []
    for (x, y) in selected_points:
        x, y = int(x), int(y)
        rgb = image.getpixel((x, y))
        if (x, y, rgb) not in points:  # Avoid duplicates
            points.append((x, y, rgb))
    return points

# Generate the greyscale and coulour point image
def gen_grey_and_colour_points(points, save_path_folder):
    """Generate a greyscale image with the colour points highlighted."""
    grey_and_colour_image = Image.open(f'{save_path_folder}/grey.png').convert('RGB')

    for (x, y, rgb) in points:
        grey_and_colour_image.putpixel((x, y), rgb)
    
    grey_and_colour_image.save(f'{save_path_folder}/grey_and_colour_points.png')


if __name__ == "__main__":

    # e.g. 
    N = 20  # Number of rows in the grid
    M = 10  # Number of columns in the grid
    test_image_path = "images/flag2.png"
    save_path_folder = "optimisation_results"

    # test_image_path = "images/gradient2.png"

    # --- Generate the colour image and save it --- #
    colour_image = Image.open(test_image_path)
    colour_image.save(f'{save_path_folder}/colour.png')

    # --- Generate the greyscale image and save it --- #
    convert_to_greyscale(test_image_path, save_path_folder)

    # --- Generate the points and save the grey and colour points image --- #
    #points = generate_regular_grid_points(test_image_path, N, M)
    points = generate_random_points(test_image_path, N*M)  # Generate N*M random points
    #points = generate_user_specified_points(test_image_path)
    
    gen_grey_and_colour_points(points, save_path_folder)
