# Whiteboard & Lecture Note-to-PDF Compiler

A high-performance computer vision tool and interactive web application that transforms photos of handwritten notes, whiteboards, and chalkboards into clean, shadow-free, pure-white background documents and multi-page PDFs while preserving colored marker strokes.

---

## 5-Stage Computer Vision Pipeline

```
Raw Camera Capture (Perspective Skew / Uneven Lighting)
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Deskew & Perspective Rectification           │
│ Canny Edge Filter → Convex PolyDP → 4-Point Homography │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Illumination Normalization & Glare Removal    │
│ Spatial Background Division / Homomorphic FFT Filter   │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Color Preservation & Tonal Binarization      │
│ Tonal Stretch + HSV Saturation Boost / Otsu Binarize  │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Denoising & Edge Sharpening                  │
│ Median Blur Filter + High-Boost Laplacian Unsharp Mask │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 5: Multi-Page Standard PDF Compilation           │
│ Aspect-Preserved Page Fitting (A4/Letter) + JPEG Opt.  │
└────────────────────────────────────────────────────────┘
```

1. **Stage 1 (Deskew & Perspective Rectification)**:
   - Gaussian pre-filter and Canny edge detection with rectangular morphological dilation.
   - External contour analysis and convex polygonal approximation (`approxPolyDP`).
   - Coordinate sorting via [top-left, top-right, bottom-right, bottom-left] heuristic.
   - Non-degenerate geometric validation (aspect bounds: 0.2 to 5.0) and cubic perspective transformation (`cv2.warpPerspective`).
   - Safe zero-distortion fallback if document bounds are obstructed or flat.

2. **Stage 2 (Illumination Normalization & Glare Removal)**:
   - **Division Method (Default)**: Morphological channel dilation and large-kernel Gaussian smoothing to estimate spatial illumination fields, followed by channel division (`np.maximum(bg, 1.0)`).
   - **Homomorphic Method**: Frequency-domain Butterworth/Gaussian high-pass filtering on log-transformed intensities (`ln(f) = ln(i) + ln(r)`), suppressing low-frequency shadows while enhancing high-frequency text reflectance.
   - **Hybrid Method**: Cascaded division pass followed by frequency normalization.
   - Non-linear gamma correction via lookup tables (`cv2.LUT`).

3. **Stage 3 (Color Preservation & Binarization)**:
   - Continuous tonal contrast stretching (`white_point` and `black_point`) with divide-by-zero clamping.
   - **Color Preservation Mode**: Preserves colored inks (blue, red, green, purple) by isolating chromaticity in HSV space and applying adjustable saturation boosting.
   - **Black & White Mode**: Clean binary segmentation supporting global thresholding (210) or local adaptive Gaussian thresholding.
   - **Grayscale Mode**: High-contrast continuous tonal reproduction.
   - **Chalkboard Inversion Mode**: Inverts dark backgrounds to pure white while preserving chalk pigment colors.

4. **Stage 4 (Denoising & Edge Sharpening)**:
   - Non-linear median filter (`k=3` or `k=5`) to eliminate paper grain and sensor noise without blurring fine handwriting strokes.
   - High-boost unsharp masking (`boost=0.0 - 0.8`) to sharpen strokes and restore crisp boundary definition.
   - Non-destructive pure white threshold clipping (`>= 244`) that eliminates residual boundary artifacts.

5. **Stage 5 (Multi-Page PDF Compilation)**:
   - Standard page layout normalization (A4: 1:1.414, Letter: 1:1.294, or Original).
   - Aspect ratio preservation with auto-centering on pure white canvas.
   - Configurable JPEG compression and Pillow PDF stream optimization.
   - Detailed compilation statistics including total pages, processing time, raw size vs. output PDF size, and compression ratios.

---

## Features

- **Pitch-Black Dark UI**: Modern high-contrast interface configured via `.streamlit/config.toml`.
- **Side-by-Side Comparison**: Real-time visual comparison between original captures and cleaned documents with dimension tags, edge sharpness indexes, and background whiteness percentages.
- **5-Stage Interactive Breakdown**: Step-by-step visual inspection of each intermediate stage.
- **Batch Processing & Individual Downloads**: Upload multiple images simultaneously, compile into a single consolidated PDF, or download individual cleaned pages as PNG files.
- **Synthetic Note Generator**: Built-in test generator simulating angled camera perspective, shadows, paper grain, and colored handwritten notes (lined paper, whiteboard, chalkboard).
- **Robust Input Handling**: Safe handling of RGB, BGR, BGRA (alpha transparency), 2D grayscale, and unusual image dimensions.

---

## Installation

Ensure Python 3.10+ is installed:

```bash
# Clone the repository
git clone https://github.com/dhruv1946/ipcv-mini-project.git
cd ipcv-mini-project

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Streamlit Web Application

Launch the interactive web interface:

```bash
python -m streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

### 2. Command-Line Interface

Process camera captures directly from the terminal or generate a synthetic demonstration:

```bash
# Process synthetic sample note:
python note_compiler.py

# Process custom image captures into a combined PDF:
python note_compiler.py page1.jpg page2.png page3.webp -o output_notes.pdf
```

### 3. Automated Test Suite

Run unit and integration tests covering all 5 stages, color-space layouts, edge cases, and memory safety:

```bash
python -m unittest test_pipeline.py
```

---

## Test Coverage Summary

The automated test suite in `test_pipeline.py` verifies 20 scenarios:
- `test_01_synthetic_generator_variants`: Generates lined paper, whiteboard, and chalkboard test images.
- `test_02_stage_1_perspective_rectification`: Validates quadrilateral contour detection and homography warping.
- `test_03_stage_2_illumination_methods`: Evaluates division, homomorphic, and hybrid filters.
- `test_04_stage_3_color_preservation_and_bw`: Tests HSV saturation boosting and binarization.
- `test_05_stage_4_denoise_and_sharpen`: Tests median smoothing and unsharp masking.
- `test_06_stage_5_pdf_compilation`: Validates multi-page PDF compilation and JPEG compression.
- `test_07_chalkboard_inversion`: Verifies luminance inversion to pure white background.
- `test_08_grayscale_and_rgba_channels`: Validates 2D grayscale and 4-channel BGRA alpha blending.
- `test_09_empty_and_corrupt_inputs`: Tests graceful rejection of empty or None arrays.
- `test_10_extreme_and_nonstandard_dimensions`: Tests tiny (32x32) and wide aspect ratios.
- `test_11_perspective_fallback_on_flat_image`: Ensures safe fallback when no polygon is detected.
- `test_12_denoise_median_filter_strength`: Verifies reduction of salt-and-pepper noise variance.
- `test_13_pdf_page_sizes_a4_and_letter`: Validates A4, Letter, and Original PDF canvas layouts.
- `test_14_grayscale_and_adaptive_modes`: Tests continuous grayscale and adaptive Gaussian thresholding.
- `test_15_image_metrics_calculation`: Verifies Laplacian sharpness and background uniformity calculators.
- `test_16_tonal_parameter_inversion_safety`: Tests inverted white/black points without zero division.
- `test_17_input_immutability_and_memory_safety`: Ensures input numpy buffers are never mutated in place.
- `test_18_multi_page_pdf_ordering_and_integrity`: Verifies multi-page ordering and byte-stream integrity.
- `test_19_ensure_bgr_image_helper`: Tests internal BGR standardization helpers.
- `test_20_config_overrides_and_defaults`: Verifies `ProcessingConfig` defaults and overrides.

---

## Project Structure

```
├── .streamlit/
│   └── config.toml        # Pitch-black minimal dark theme configuration
├── app.py                 # Streamlit web application & interactive workspace
├── note_compiler.py       # Core 5-stage CV pipeline, math algorithms, & PDF compiler
├── test_pipeline.py       # 20 automated unit and integration tests
├── requirements.txt       # Project dependencies
└── README.md              # Technical documentation and execution guide
```