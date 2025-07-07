import os
import cv2
import numpy as np
import pandas as pd
from skimage import morphology, measure, segmentation
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.morphology import h_maxima, closing, disk

def analyze_image(image_path, min_cell_area=500, max_cell_area=4000, min_circularity=0.25, max_circularity=0.85):
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
    labels = segmentation.watershed(-distance_blurred, markers, mask=binary)

    valid_areas = []
    for p in measure.regionprops(labels):
        area = p.area
        perimeter = p.perimeter
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if min_cell_area <= area <= max_cell_area and min_circularity <= circularity <= max_circularity:
            valid_areas.append(area)

    total_area = sum(valid_areas)
    avg_area = np.mean(valid_areas) if valid_areas else 0
    cell_count = len(valid_areas)
    image_area = gray.shape[0] * gray.shape[1]
    coverage = (total_area / image_area) * 100 if image_area > 0 else 0

    return {
        "file": os.path.basename(image_path),
        "cell count": cell_count,
        "total area px2": total_area,
        "average cell_area px2": avg_area,
        "coverage percent": coverage
    }

def batch_process_images(folder_path, output_csv="batch_neuron_analysis_results.csv"):
    results = []
    supported_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_ext)]

    for file in image_files:
        path = os.path.join(folder_path, file)
        result = analyze_image(path)
        results.append(result)

    df = pd.DataFrame(results)
    output_path = os.path.join(folder_path, output_csv)
    df.to_csv(output_path, index=False)

# Example usage
if __name__ == "__main__":
    folder = r"F:\Cropped slides\TS PV\Slide 2"  # Replace with your image folder path
    batch_process_images(folder)
