# python script to rescale a large image to a smaller size using PIL
from PIL import Image

def scale_image_to_size(image_path, output_path, target_size=(100, 100)):
    # if the image is larger than the target size, rescale it down
    image = Image.open(image_path)
    
    if image.size[0] > target_size[0] or image.size[1] > target_size[1]:
        print(f"Rescaling image from {image.size} to {target_size}...")
        image = image.resize(target_size, Image.LANCZOS)
        image.save(output_path)
    else:
        image.save(output_path)  # Save the original image to the output path as well


if __name__ == "__main__":
    image_path = "images/IMG_6006.jpg"
    output_path = "complex_pattern_100x100.png"
    
    scale_image_to_size(image_path, output_path, target_size=(100, 100))
    