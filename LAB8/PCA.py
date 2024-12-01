import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

rows, cols = 5, 5
grid = np.random.randint(0, 255, size=(rows, cols), dtype=np.uint8)

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

def apply_filter(grid, kernel):
    return convolve(grid, kernel, mode='constant', cval=0)

def update_grid(grid):
    edges_x = apply_filter(grid, sobel_x)
    edges_y = apply_filter(grid, sobel_y)

    new_grid = np.hypot(edges_x, edges_y)
    new_grid = (new_grid / new_grid.max()) * 255

    return new_grid.astype(np.uint8)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image (Random)")
plt.imshow(grid, cmap='gray')

new_grid = update_grid(grid)

plt.subplot(1, 2, 2)
plt.title("Edge Detection (Sobel Filter)")
plt.imshow(new_grid, cmap='gray')
plt.show()
