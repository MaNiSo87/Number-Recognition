import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image

def illustration(data):
    fig, ax = plt.subplots(figsize=(6, 6))

    if isinstance(data, str):
        pil_image = Image.open(data)
    else:
        pil_image = data
    
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    
    numpy_array = np.array(pil_image)
    
    if numpy_array.ndim == 0:
        print("Error: Image array is empty or invalid")
        return
    elif numpy_array.ndim == 1:
        if len(numpy_array) == 784:
            numpy_array = numpy_array.reshape(28, 28)
        else:
            print(f"Error: Cannot reshape 1D array of length {len(numpy_array)}")
            return
    elif numpy_array.ndim == 3:
        numpy_array = np.mean(numpy_array, axis=2)
    
    height, width = numpy_array.shape
    for x in range(width + 1):
        ax.axvline(x - 0.5, color='white', linewidth=0.5)
    for y in range(height + 1):
        ax.axhline(y - 0.5, color='white', linewidth=0.5)
    x_labels = [str(i) for i in range(1, width + 1)]
    y_labels = [str(i) for i in range(1, height + 1)]
    
    ax.imshow(numpy_array, cmap='gray')
    ax.set_xticks(np.arange(0, width))
    ax.set_yticks(np.arange(0, height))
    ax.set_xticklabels(x_labels, fontsize=6)
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_title("MNIST Digit with Pixel Grid and Numbered Axes")
    
    plt.show()


if __name__ == "__main__":
    illustration('test.png')
