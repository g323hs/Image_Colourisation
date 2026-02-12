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
def generate_random_points(image_path, num_points, seed):
    """Generate a specified number of random points from the image."""
    import random
    image = Image.open(image_path)
    width, height = image.size
    points = []
    random.seed(seed)
    for _ in range(num_points):
        x = random.randint(0, width - 1)  # Random x coordinate
        y = random.randint(0, height - 1)  # Random y coordinate
        rgb = image.getpixel((x, y))  # Get RGB values at the point
        points.append((x, y, rgb))
    return points

# Method 3: User-specified points with cursor
def generate_user_specified_points(image_path):
    """Generate points specified by the user with a cursor."""
    import matplotlib.pyplot as plt
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title("Click to select points")
    points = plt.ginput(n=-1)  # PRESS ENTER when done
    plt.close()

    print(f"Selected points: {points}")
    
    # Get RGB values for the selected points
    points_with_rgb = []
    for (x, y) in points:
        x, y = int(x), int(y)
        rgb = image.getpixel((x, y))
        points_with_rgb.append((x, y, rgb))
    
    return points_with_rgb

# Generate the greyscale and coulour point image
def gen_grey_and_colour_points(points):
    """Generate a greyscale image with the colour points highlighted."""
    grey_and_colour_image = Image.open('working_images/grey.png').convert('RGB')

    for (x, y, rgb) in points:
        grey_and_colour_image.putpixel((x, y), rgb)
    
    grey_and_colour_image.save('working_images/grey_and_colour_points.png')


if __name__ == "__main__":

    # e.g. 
    N = 20  # Number of rows in the grid
    M = 10  # Number of columns in the grid
    test_image_path = "images/flag2.png"
    # test_image_path = "images/gradient2.png"

    # --- Generate the colour image and save it --- #
    colour_image = Image.open(test_image_path)
    colour_image.save('working_images/colour.png')

    # --- Generate the greyscale image and save it --- #
    convert_to_greyscale(test_image_path)

    # --- Generate the points and save the grey and colour points image --- #
    #points = generate_regular_grid_points(test_image_path, N, M)
    points = generate_random_points(test_image_path, N*M, seed=1)  # Generate N*M random points
    #points = generate_user_specified_points(test_image_path)
    
    gen_grey_and_colour_points(points)
