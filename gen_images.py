from PIL import Image
import numpy as np
import os

def save_image_pil(image, filename):
    os.makedirs('images', exist_ok=True)
    img = Image.fromarray(image)
    img.save(f'images/{filename}')

def create_flag1(height, width):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :width//3] = [0, 0, 255]
    image[:, width//3:2*width//3] = [255, 255, 255]
    image[:, 2*width//3:] = [255, 0, 0]
    return image

def create_flag2(height, width):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:height//3, :] = [0, 0, 0]
    image[height//3:2*height//3, :] = [255, 0, 0]
    image[2*height//3:, :] = [255, 255, 0]
    return image

def create_flag3(height, width):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if j > (width / height) * i:
                image[i, j] = [0, 128, 0]
            else:
                image[i, j] = [128, 0, 128]
    return image

def create_flag4(height, width):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:height//2, :width//2] = [255, 0, 0]
    image[:height//2, width//2:] = [255, 255, 0]
    image[height//2:, :width//2] = [0, 0, 255]
    image[height//2:, width//2:] = [0, 255, 0]
    return image

def create_flag5(height, width):
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    center_y, center_x = height // 2, width // 2
    radius = min(height, width) // 4
    y, x = np.ogrid[:height, :width]
    mask = (y - center_y)**2 + (x - center_x)**2 <= radius**2
    image[mask] = [255, 0, 0]
    return image

def create_gradient1(height, width):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            r = int(255 * j / (width - 1))
            g = int(255 * i / (height - 1))
            b = int(255 * (i + j) / (height + width - 2))
            image[i, j] = [r, g, b]
    return image

def create_gradient2(height, width):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    center_y = height / 2
    center_x = width / 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    for i in range(height):
        for j in range(width):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2) / max_dist
            r = int(255 * (1 - dist))
            g = int(255 * dist)
            b = int(128 + 127 * np.sin(2 * np.pi * dist))
            image[i, j] = [r, g, b]
    return image

def create_checkerboard(height, width):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    block_size = 50
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            color = colors[(i // block_size + j // block_size) % len(colors)]
            image[i:i+block_size, j:j+block_size] = color
            if i + block_size < height:
                for k in range(10):
                    alpha = k / 10
                    image[i+block_size+k, j:j+block_size] = ((1-alpha) * np.array(color) + alpha * np.array([255, 255, 255])).astype(np.uint8)
            if j + block_size < width:
                for k in range(10):
                    alpha = k / 10
                    image[i:i+block_size, j+block_size+k] = ((1-alpha) * np.array(color) + alpha * np.array([255, 255, 255])).astype(np.uint8)
    return image

def create_complex_pattern(height, width):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            r = int(255 * (1 + np.sin(i / 20)) / 2)
            g = int(255 * (1 + np.cos(j / 20)) / 2)
            b = int(255 * (1 + np.sin((i+j) / 30)) / 2)
            image[i, j] = [r, g, b]
    return image

def generate_images(height=100, width=100):
    save_image_pil(create_flag1(height, width), 'flag1.png')
    save_image_pil(create_flag2(height, width), 'flag2.png')
    save_image_pil(create_flag3(height, width), 'flag3.png')
    save_image_pil(create_flag4(height, width), 'flag4.png')
    save_image_pil(create_flag5(height, width), 'flag5.png')
    save_image_pil(create_gradient1(height, width), 'gradient1.png')
    save_image_pil(create_gradient2(height, width), 'gradient2.png')
    save_image_pil(create_checkerboard(height, width), 'checkerboard.png')
    save_image_pil(create_complex_pattern(height, width), 'complex_pattern.png')


if __name__ == "__main__":
    generate_images(100, 100)