from PIL import Image
import numpy as np
import os


if __name__ == "__main__":
    HEIGHT = 200
    WIDTH = 500

    def save_image_pil(image, filename):
        os.makedirs('images', exist_ok=True)
        img = Image.fromarray(image)            # image is (H,W,3) uint8
        img.save(f'images/{filename}')


    # Flag 1: Blue-White-Red (vertical stripes)
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, :WIDTH//3] = [0, 0, 255]  # blue
    image[:, WIDTH//3:2*WIDTH//3] = [255, 255, 255]  # white
    image[:, 2*WIDTH//3:] = [255, 0, 0]  # red
    save_image_pil(image, 'flag1.png')

    # Flag 2: Black-Red-Yellow (horizontal stripes)
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:HEIGHT//3, :] = [0, 0, 0]  # black
    image[HEIGHT//3:2*HEIGHT//3, :] = [255, 0, 0]  # red
    image[2*HEIGHT//3:, :] = [255, 255, 0]  # yellow
    save_image_pil(image, 'flag2.png')

    # Flag 3: Green and Purple diagonal
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if j > (WIDTH / HEIGHT) * i:
                image[i, j] = [0, 128, 0]  # green
            else:
                image[i, j] = [128, 0, 128]  # purple
    save_image_pil(image, 'flag3.png')

    # Flag 4: Four quadrants (Red, Yellow, Blue, Green)
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:HEIGHT//2, :WIDTH//2] = [255, 0, 0]  # red
    image[:HEIGHT//2, WIDTH//2:] = [255, 255, 0]  # yellow
    image[HEIGHT//2:, :WIDTH//2] = [0, 0, 255]  # blue
    image[HEIGHT//2:, WIDTH//2:] = [0, 255, 0]  # green
    save_image_pil(image, 'flag4.png')

    # Flag 5: White background with red circle
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    center_y, center_x = HEIGHT // 2, WIDTH // 2
    radius = min(HEIGHT, WIDTH) // 4
    y, x = np.ogrid[:HEIGHT, :WIDTH]
    mask = (y - center_y)**2 + (x - center_x)**2 <= radius**2
    image[mask] = [255, 0, 0]  # red
    save_image_pil(image, 'flag5.png')

    # Now generate more complicated images with gradients and patterns
    # Smooth 2D colour gradient
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for i in range(HEIGHT):
        for j in range(WIDTH):
            r = int(255 * j / (WIDTH - 1))          # horizontal gradient
            g = int(255 * i / (HEIGHT - 1))         # vertical gradient
            b = int(255 * (i + j) / (HEIGHT + WIDTH - 2))  # diagonal blend
            image[i, j] = [r, g, b]

    save_image_pil(image, 'gradient1.png')

    # Smooth radial colour gradient
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    center_y = HEIGHT / 2
    center_x = WIDTH / 2
    max_dist = np.sqrt(center_x**2 + center_y**2)

    for i in range(HEIGHT):
        for j in range(WIDTH):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2) / max_dist
            r = int(255 * (1 - dist))
            g = int(255 * dist)
            b = int(128 + 127 * np.sin(2 * np.pi * dist))
            image[i, j] = [r, g, b]

    save_image_pil(image, 'gradient2.png')

