import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import morphology, measure, segmentation
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.morphology import h_maxima, closing, disk
from scipy.stats import gaussian_kde

def analyze_image_kde_fast(
    image_path,
    min_cell_area=500,
    max_cell_area=4000,
    min_circularity=0.25,
    max_circularity=0.85,
    kde_scale=0.25
):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    thresh = threshold_otsu(blurred)
    binary = (blurred < thresh).astype(np.uint8)
    binary = closing(binary, disk(3))
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=min_cell_area)
    binary = binary.astype(np.uint8)

    distance = ndi.distance_transform_edt(binary)
    distance_blurred = ndi.gaussian_filter(distance, sigma=1.0)
    h_peaks = h_maxima(distance_blurred, h=2.0)
    markers = ndi.label(h_peaks)[0]
    labels = segmentation.watershed(-distance, markers, mask=binary)

    valid_areas = []
    centers = []

    for p in measure.regionprops(labels):
        area = p.area
        perimeter = p.perimeter
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if min_cell_area <= area <= max_cell_area and min_circularity <= circularity <= max_circularity:
            valid_areas.append(area)
            centers.append(p.centroid)

    cell_count = len(valid_areas)
    total_area = sum(valid_areas)
    avg_area = np.mean(valid_areas) if valid_areas else 0
    image_area = gray.shape[0] * gray.shape[1]
    coverage = (total_area / image_area) * 100 if image_area else 0

    # Generate binary mask
    filtered_labels = np.zeros_like(labels, dtype=np.uint8)
    for p in measure.regionprops(labels):
        area = p.area
        perimeter = p.perimeter
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if min_cell_area <= area <= max_cell_area and min_circularity <= circularity <= max_circularity:
            filtered_labels[labels == p.label] = 255

    # KDE at downsampled scale
    h, w = gray.shape
    Z = np.zeros((int(h * kde_scale), int(w * kde_scale)))
    if centers:
        y_coords, x_coords = zip(*centers)
        xy = np.vstack([np.array(x_coords), np.array(y_coords)])
        kde = gaussian_kde(xy)
        xgrid = np.linspace(0, w, int(w * kde_scale))
        ygrid = np.linspace(0, h, int(h * kde_scale))
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # Upsample KDE to full resolution
    Z_upsampled = cv2.resize(Z, (w, h), interpolation=cv2.INTER_CUBIC)

    # Save outputs
    base_path = os.path.splitext(image_path)[0]
    mask_path = f"{base_path}_binary_mask.png"
    heatmap_overlay_path = f"{base_path}_kde_heatmap.png"
    heatmap_only_path = f"{base_path}_kde_only.png"

    cv2.imwrite(mask_path, filtered_labels)

    # Save KDE-only grayscale heatmap
    plt.imsave(heatmap_only_path, Z_upsampled, cmap='hot')


    # Save overlay
    plt.figure(figsize=(8, 6))
    plt.imshow(gray, cmap='gray')
    plt.imshow(Z_upsampled, cmap='hot', alpha=0.6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(heatmap_overlay_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


    return {
        "file": os.path.basename(image_path),
        "cell count": cell_count,
        "total area px2": total_area,
        "average cell area px2": avg_area,
        "coverage percent": coverage,
        "mask path": mask_path,
        "kde overlay path": heatmap_overlay_path,
        "kde only path": heatmap_only_path
    }

# Example usage
if __name__ == "__main__":
    stats = analyze_image_kde_fast(r"F:\Cropped slides\TS PV\Slide 1\TS PV Slide 1 Image 10.tif")
    print(stats)
