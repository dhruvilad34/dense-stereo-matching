import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr # type: ignore

# === ZNCC Similarity Function ===
def zncc_score(patch1, patch2):
    """Compute ZNCC score between two patches."""
    if patch1.shape != patch2.shape:
        return -1
    mean1, mean2 = np.mean(patch1), np.mean(patch2)
    std1, std2 = np.std(patch1), np.std(patch2)
    if std1 == 0 or std2 == 0:
        return -1
    return np.sum((patch1 - mean1) * (patch2 - mean2)) / (patch1.size * std1 * std2)

# === Step 1: Match Edge Pixels ===
def match_edge_pixels(left, right, edge_mask, window_size=11, max_disp=32, zncc_threshold=0.3):
    """Match edge pixels using ZNCC along epipolar lines."""
    h, w = left.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)
    half_win = window_size // 2

    for y in range(half_win, h - half_win):
        for x in range(half_win + max_disp, w - half_win):
            if edge_mask[y, x] == 0:
                continue
            best_score = -1
            best_disp = 0
            left_patch = left[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1]
            for d in range(max_disp):
                xr = x - d
                if xr - half_win < 0:
                    break
                right_patch = right[y - half_win:y + half_win + 1, xr - half_win:xr + half_win + 1]
                score = zncc_score(left_patch, right_patch)
                if score > best_score:
                    best_score = score
                    best_disp = d
            if best_score >= zncc_threshold:
                disparity_map[y, x] = best_disp
    return disparity_map

# === Step 2: Interpolate Non-Edge Pixels ===
def interpolate_non_edge(disparity_map, edge_mask):
    """Interpolate disparities in non-edge regions based on segment mapping."""
    h, w = disparity_map.shape
    filled = disparity_map.copy()
    for y in range(h):
        x = 0
        while x < w:
            if edge_mask[y, x] > 0:
                x += 1
                continue
            x_start = x
            while x < w and edge_mask[y, x] == 0:
                x += 1
            x_end = x - 1
            if x_start == 0 or x_end >= w - 1:
                continue
            d_start = filled[y, x_start - 1]
            d_end = filled[y, x_end + 1]
            if d_start == 0 or d_end == 0:
                continue
            for xi in range(x_start, x_end + 1):
                ratio = (xi - x_start + 1) / (x_end - x_start + 2)
                filled[y, xi] = (1 - ratio) * d_start + ratio * d_end
    return filled

# === Step 3: Reconstruct Right Image ===
def reconstruct_right_image(left, disparity_map):
    """Reconstruct the right image using the disparity map."""
    h, w = left.shape
    reconstructed = np.zeros_like(left)
    for y in range(h):
        for x in range(w):
            d = int(disparity_map[y, x])
            x_shift = x - d
            if 0 <= x_shift < w:
                reconstructed[y, x] = left[y, x_shift]
    return reconstructed

# === Main Pipeline ===
def main():
    # Load and split stereo image
    img = cv2.imread("windsor.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: 'windsor.jpg' not found.")
        return

    h, w = img.shape
    left = img[:, :w // 2]
    right = img[:, w // 2 - 30:w - 30]  # Adjusted for alignment

    # Preprocess: Gaussian Blur before edge detection
    left_blur = cv2.GaussianBlur(left, (5, 5), 0)

    # Edge detection using Sobel operator
    sobelx = cv2.Sobel(left_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(left_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    _, edge_mask = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)

    # Step 1: Match edge pixels using ZNCC
    print("🔍 Matching edge pixels using ZNCC...")
    disparity_edge = match_edge_pixels(left, right, edge_mask, window_size=11, max_disp=32, zncc_threshold=0.3)

    # Step 2: Interpolate non-edge pixel disparities
    print("🔧 Interpolating non-edge disparities...")
    disparity_full = interpolate_non_edge(disparity_edge, edge_mask)

    # Step 3: Smooth the disparity map with median filtering
    disparity_full = cv2.medianBlur(disparity_full.astype(np.uint8), 5)

    # Step 4: Reconstruct the right image from disparity
    print("🔄 Reconstructing right image...")
    reconstructed = reconstruct_right_image(left, disparity_full)

    # Step 5: Inpaint unmatched pixels (black areas)
    print("🧽 Inpainting unmatched pixels...")
    mask = (reconstructed == 0).astype(np.uint8) * 255
    reconstructed_inpainted = cv2.inpaint(reconstructed, mask, 3, cv2.INPAINT_TELEA)

    # Step 6: Save output images
    cv2.imwrite("disparity_map.png", (disparity_full * 4).astype(np.uint8))  # Scaled for visibility
    cv2.imwrite("reconstructed_right.png", reconstructed_inpainted)

    # Step 7: Evaluate with PSNR
    print("📈 PSNR:", psnr(right, reconstructed_inpainted))

    # Step 8: Show visualization
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.title("Dense Disparity Map")
    plt.imshow(disparity_full, cmap='plasma')
    plt.axis('off')

    plt.subplot(132)
    plt.title("Original Right Image")
    plt.imshow(right, cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.title("Reconstructed Right Image")
    plt.imshow(reconstructed_inpainted, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
