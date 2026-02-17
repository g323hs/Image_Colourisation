from PIL import Image

def convert_to_greyscale(input_path, save_path_folder):
    """Convert an image to greyscale and save it."""
    with Image.open(input_path) as image:
        grey_image = image.convert('L', matrix=(0.3, 0.6, 0.1, 0))  # Convert to greyscale 
        # save in working_images folder
        grey_image.save(f'{save_path_folder}/grey.png')

if __name__ == "__main__":
    # Process all images in the 'images' folder
    image = "images/flag2.png"
    convert_to_greyscale(image, save_path_folder="working_images")

