"""Denoise a grayscale image with eigendecomposition.

Project uses local 'attampt2.jpg' image as test.
"""

import numpy as np
import skimage as ski
import matplotlib.pyplot as plt


def image_denoise(matrix: np.ndarray, values: int = 75) -> np.ndarray:
    """Reconstruct an image using its strongest eigencomponents.

    Parameters
    ----------
    matrix : numpy.ndarray
        Input image as a two-dimensional grayscale array or a three-dimensional
        RGB array. Grayscale values should be in the range ``[0, 1]``.
    values : int, default=75
        Number of eigencomponents to retain. Smaller values produce more
        smoothing, while larger values preserve more image detail.

    Returns
    -------
    numpy.ndarray
        Denoised grayscale image with values clipped to the range ``[0, 1]``.

    Notes
    -----
    The image is centered by subtracting its column mean before forming a square matrix. 
    The mean is added back after the low-rank reconstruction
    to preserve the image contrast.
    """
    grayscale = ski.color.rgb2gray(matrix) if matrix.ndim == 3 else matrix
    mean = grayscale.mean(axis=0)
    centered = grayscale - mean

    new_matx = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(new_matx)

    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order[:values]]

    compressed = centered @ eigenvectors @ eigenvectors.T + mean

    return np.clip(compressed, 0, 1)

image = ski.io.imread("attempt2.jpg", as_gray=True)
scale = 800 / max(image.shape)
if scale < 1:
    image = ski.transform.resize(
        image,
        (round(image.shape[0] * scale), round(image.shape[1] * scale)),
        anti_aliasing=True,
    )
denoised = image_denoise(image)

figure, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original")
axes[1].imshow(denoised, cmap="gray")
axes[1].set_title("Denoised")

for axis in axes:
    axis.axis("off")

figure.tight_layout()
figure.savefig("denoising_results.png", dpi=150)
print("Saved denoising_results.png")

if __name__ == "__main__":
    main()
