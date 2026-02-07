from PIL import Image

def convert_to_greyscale(input_path):
    """Convert an image to greyscale and save it."""
    with Image.open(input_path) as image:
        grey_image = image.convert('L')  # Convert to greyscale 
        # save in working_images folder
        grey_image.save('working_images/grey.png')

if __name__ == "__main__":
    # Process all images in the 'images' folder
    image = "images/flag3.png"
    convert_to_greyscale(image)

