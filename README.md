# Whiteboard & Lecture Note-to-PDF Compiler

A computer vision tool that transforms photos of handwritten notes, whiteboards, and chalkboards into clean, shadow-free, pure-white background PDF documents while preserving colored marker strokes.

## Features

- *Stage 1 (Deskew & Perspective Rectification)*: Contour detection, convex quadrilateral approximation, and perspective transformation to rectify angled captures.
- *Stage 2 (Illumination Normalization & Glare Removal)*: Spatial background division and frequency-domain homomorphic filtering to eliminate uneven lighting, shadows, and flash spots.
- *Stage 3 (Color Preservation & Binarization)*: HSV saturation thresholding and continuous tonal contrast stretching to keep colored ink strokes (blue, red, green) vibrant against a pure white background. Includes chalkboard inversion mode.
- *Stage 4 (Denoising & Edge Sharpening)*: Non-linear median filter for grain removal, morphological gap healing, and high-boost filtering for crisp handwriting edges.
- *Stage 5 (Multi-Page PDF Compilation)*: Automated scaling to standard page dimensions (A4/Letter), JPEG compression, and multi-page compilation using Pillow.
- *Synthetic Note Generator*: Built-in test generator simulating angled camera perspective, shadows, paper grain, and colored handwritten notes.

## Installation

```bash
pip install -r requirements.txt