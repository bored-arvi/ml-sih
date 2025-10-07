from skimage import color
import cv2
import numpy as np
import matplotlib.pyplot as plt

def deltaE_visual_check(img1_path, img2_path, resize_dim=(512,512), threshold=5):
    """
    Compare two certificate images and show:
    - fraction of pixels noticeably different
    - visual map of differences
    """
    # Load and resize
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.cvtColor(cv2.resize(img1, resize_dim), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.resize(img2, resize_dim), cv2.COLOR_BGR2RGB)

    # Convert to LAB
    lab1 = color.rgb2lab(img1)
    lab2 = color.rgb2lab(img2)

    # DeltaE map
    deltaE_map = np.sqrt(np.sum((lab1 - lab2)**2, axis=2))

    # Fraction of pixels above threshold
    noticeable_fraction = np.mean(deltaE_map > threshold)
    print(f"Noticeable fraction of pixels (ΔE > {threshold}): {noticeable_fraction:.3f}")

    # Visualize differences
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.title("Certificate 1")
    plt.imshow(img1)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Certificate 2")
    plt.imshow(img2)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("ΔE Difference Map")
    plt.imshow(deltaE_map, cmap='hot')
    plt.colorbar(label='ΔE')
    plt.axis('off')

    plt.show()

# --- Example usage ---
cert1_path = "embed.png"
cert2_path = "trial color mismatch.png"

deltaE_visual_check(cert1_path, cert2_path, threshold=5)
