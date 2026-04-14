# Optical Gauge Length Tracking via Digital Image Correlation

Tracks black circular markers on tensile test specimens across thousands of video frames to measure gauge length evolution and engineering strain.

Based on:
> Pan, B. et al. (2009). *Two-dimensional digital image correlation for in-plane displacement and strain measurement: a review.* Meas. Sci. Technol. 20, 062001.

## How It Works

1. **Binary image generation** — Adaptive thresholding isolates dark markers from the bright specimen surface.
2. **Morphological cleanup** — Opening/closing removes noise while preserving marker shapes.
3. **Connected component labeling** — `cv2.connectedComponentsWithStats` identifies blobs, filtered by area and circularity.
4. **Centroid tracking** — Area-weighted centroids are computed per marker. Frame-to-frame matching uses nearest-neighbor on centroid distance.
5. **Gauge length** — Euclidean distance between marker pair centroids, converted to mm via calibration factor from `config1.dat`.
6. **Engineering strain** — `e = (d - d0) / d0` where `d0` is the initial gauge length.

## Dependencies

```
pip install numpy opencv-python matplotlib
```

## Project Structure

```
├── tracker.py    # Marker detection, matching, gauge/strain computation
├── main.py       # Entry point — runs tracking and plots results
└── Images/
    ├── dbe0b/    # Dataset 1: ~21,000 frames
    └── dbe0c/    # Dataset 2: ~18,800 frames
```

## Usage

```bash
# Run on all datasets (every 10th frame)
python3 main.py

# Single dataset
python3 main.py --dataset dbe0b

# Every frame (slow but precise)
python3 main.py --step 1

# Quick test (first 200 frames)
python3 main.py --max-frames 200
```

## Sample Output

```
Dataset: dbe0b
Total images: 21199

Marker positions (frame 0):
  Marker 0: (196.9, 138.2) px
  Marker 1: (233.5, 137.3) px
  Marker 2: (204.5, 415.6) px
  Marker 3: (237.6, 415.5) px

Gauge length results (mm):
  Pair 0-2: L0=30.84 mm, max strain=1.71%
  Pair 1-3: L0=30.91 mm, max strain=1.65%
```

Generates a plot with gauge length, engineering strain, and marker y-coordinates vs frame index.

## Calibration

Each dataset includes a `config1.dat` file with comma-separated calibration values. The tracker reads the mm/pixel factor automatically to report gauge lengths in physical units.
