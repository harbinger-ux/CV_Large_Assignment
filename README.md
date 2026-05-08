```markdown
# Robust ArUco Marker Detection: Classic & Hybrid Approaches

This repository contains a comprehensive pipeline for detecting and decoding ArUco markers (specifically `DICT_ARUCO_MIP_36H12`). It features two distinct detection methodologies designed to handle varying lighting conditions and image degradations, originally developed as a Computer Vision Large Assignment.

## Repository Structure

```text
CV_Large_Assignment/
├── data/                       # Contains images and ground truth CSVs
│   └── val/                    
├── models/                     # Core detection algorithms
│   ├── __init__.py
│   ├── ClassicAruco.py         # Traditional CV approach
│   ├── HybridAruco.py          # Deep Learning + CV approach
│   └── utils.py                # Math, dictionary generation, and enhancements
├── weight/                     # YOLO model weights (Must be downloaded/added)
│   └── det_luma_bnc_s.pt       
├── demo.py                     # Single-image inference and visualization
├── evaluate.py                 # Evaluation and scoring script
└── requirements.txt            # Project dependencies

```

---

## Installation

**1. Clone the repository:**

```bash
git clone [https://github.com/harbinger-ux/CV_Large_Assignment.git](https://github.com/harbinger-ux/CV_Large_Assignment.git)
cd CV_Large_Assignment

```

**2. Install dependencies:**
It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt

```

**3. YOLO Weights:**
To use the `HybridAruco` detector, ensure your YOLO weights file (`det_luma_bnc_s.pt`) is placed inside the `weight/` directory.

---

## Usage

### 1. Quick Start & Visualization (`demo.py`)

To test both the Classic and Hybrid models on a single image and view a side-by-side comparison:

```bash
python demo.py

```

*Note: Press the `ESC` key to close the visualization window.*

### 2. Batch Processing a Dataset

You can easily import and run the models on entire directories to generate prediction CSVs.

```python
from models.ClassicAruco import ClassicAruco
from models.HybridAruco import HybridAruco

# Run Classic Detector
classic = ClassicAruco(max_error=4)
classic.detect_dataset(image_dir="data/val", output_csv="predictions_classic.csv")

# Run Hybrid Detector
hybrid = HybridAruco(model_path="weight/det_luma_bnc_s.pt", max_error=4)
hybrid.detect_dataset(image_dir="data/val", output_csv="predictions_hybrid.csv")

```

### 3. Evaluation (`evaluate.py`)

Evaluate your model's predictions against a ground truth CSV. The script calculates a final score based on detection accuracy and heavily penalizes spam/false positives.

Run the evaluation from the command line using arguments:

```bash
# Evaluate the Hybrid model on the Validation set
python evaluate.py --data_type val --model_type hybrid

# Evaluate the Classic model on the Validation set
python evaluate.py --data_type val --model_type classic

```

**Visualizing Failures:**
To inspect where the model failed, uncomment `evaluator.visualize_failures(...)` at the bottom of `evaluate.py`. It displays a side-by-side view of the Ground Truth vs. Prediction for images that scored below a specific threshold.

---

## Methodology Details

### Image Enhancement (`utils.py`)

Both pipelines utilize pre-processing to stabilize edge detection in poor lighting:

* **Flat-Field Correction:** Erases harsh gradients by normalizing the image against a heavily blurred light map.
* **Gamma Correction:** Applies a non-linear LUT curve to recover details in under-exposed regions.

### Classic vs. Hybrid

* **ClassicAruco:** Iterates over the entire image using a sliding window for `cv2.adaptiveThreshold`. It is highly robust but computationally heavier as it tests every potential square contour.
* **HybridAruco:** Relies on YOLO to establish a contextual Region of Interest (ROI) with a slight pixel padding. It runs the extensive adaptive thresholding loops *only* within these ROIs, significantly reducing false positives and processing time while handling complex backgrounds.

```

```
