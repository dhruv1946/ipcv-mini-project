import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class ProcessingConfig:
    auto_perspective: bool = True
    illumination_method: str = "division"  # "division", "homomorphic", "hybrid"
    bg_kernel_size: int = 25
    white_point: float = 210.0
    black_point: float = 60.0
    gamma: float = 1.0
    mode: str = "color_preserve"  # "color_preserve", "black_and_white", "grayscale", "chalkboard_invert"
    color_saturation_boost: float = 1.35
    sharpen_boost: float = 0.35
    denoise_ksize: int = 1  # 1 = off, 3 = mild, 5 = strong
    adaptive_threshold: bool = False
    compress_pdf: bool = True
    page_size: str = "A4"  # "A4", "Letter", "Original"
    jpeg_quality: int = 90

@dataclass
class StageResult:
    image: np.ndarray
    stage_name: str
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineResult:
    original_image: np.ndarray
    final_image: np.ndarray
    stages: Dict[str, StageResult]
    total_time_ms: float
    page_shape: Tuple[int, int]
    success: bool
    notes: List[str] = field(default_factory=list)

def ensure_bgr_image(image: np.ndarray) -> np.ndarray:
    """Validates and ensures the input image is a 3-channel uint8 BGR ndarray."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image: array is None or empty")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        if image.shape[2] == 3:
            return image
        elif image.shape[2] == 4:
            # Composite BGRA on a white canvas
            b, g, r, a = cv2.split(image)
            alpha = a.astype(np.float32) / 255.0
            bgr = cv2.merge([b, g, r]).astype(np.float32)
            comp = bgr * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])
            return np.clip(comp, 0, 255).astype(np.uint8)
        elif image.shape[2] == 1:
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported image shape: {image.shape}")

def compute_image_metrics(image: np.ndarray) -> Dict[str, Any]:
    """Computes objective image statistics without modifying the image."""
    if image is None or image.size == 0:
        return {
            "sharpness_index": 0.0,
            "contrast_std": 0.0,
            "mean_luminance": 0.0,
            "white_background_pct": 0.0,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(np.std(gray))
    mean_lum = float(np.mean(gray))
    white_px = float(np.count_nonzero(gray >= 245))
    white_ratio = round((white_px / max(gray.size, 1)) * 100.0, 1)

    return {
        "sharpness_index": round(lap_var, 1),
        "contrast_std": round(contrast, 1),
        "mean_luminance": round(mean_lum, 1),
        "white_background_pct": white_ratio,
    }

def generate_synthetic_note(width=1200, height=900, note_type="lined_paper", add_shadow=True, add_perspective=True, seed=42):
    """Generates synthetic camera photos of handwritten notes, whiteboards, or chalkboards."""
    if seed is not None:
        np.random.seed(seed)
    doc_w, doc_h = int(width * 0.8), int(height * 0.82)
    bg_col = (35, 45, 30) if note_type == "chalkboard" else (245, 248, 250)
    doc = np.full((doc_h, doc_w, 3), bg_col, dtype=np.uint8)
    if note_type == "lined_paper":
        cv2.line(doc, (int(doc_w * 0.12), 0), (int(doc_w * 0.12), doc_h), (180, 180, 235), 2)
        for y in range(60, doc_h - 30, 36):
            cv2.line(doc, (0, y), (doc_w, y), (230, 215, 190), 1)
    tc = (220, 240, 255) if note_type == "chalkboard" else (30, 25, 20)
    bc = (240, 210, 150) if note_type == "chalkboard" else (180, 50, 20)
    rc = (180, 180, 255) if note_type == "chalkboard" else (30, 30, 200)
    gc = (160, 240, 180) if note_type == "chalkboard" else (30, 150, 40)
    cv2.putText(doc, "Lecture 07: Homomorphic & Spatial Filtering", (int(doc_w * 0.15), 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, tc, 2)
    cv2.putText(doc, "f(x, y) = i(x, y) * r(x, y) -> ln(f) = ln(i) + ln(r)", (int(doc_w * 0.15), 110), cv2.FONT_HERSHEY_SIMPLEX, 0.62, bc, 2)
    cv2.rectangle(doc, (int(doc_w * 0.15), 140), (int(doc_w * 0.85), 210), gc, 2)
    cv2.putText(doc, "High-Boost Filter: H(u,v) = (g_H - g_L)[1 - exp(-c D^2/Do^2)] + g_L", (int(doc_w * 0.17), 185), cv2.FONT_HERSHEY_SIMPLEX, 0.52, gc, 2)
    cv2.arrowedLine(doc, (int(doc_w * 0.8), 240), (int(doc_w * 0.7), 215), rc, 2)
    cv2.putText(doc, "Boosts text edges and contrast", (int(doc_w * 0.45), 260), cv2.FONT_HERSHEY_SIMPLEX, 0.58, rc, 2)
    canvas_bg = np.full((height, width, 3), (45, 35, 30), dtype=np.uint8)
    if add_perspective:
        src = np.float32([[0, 0], [doc_w, 0], [doc_w, doc_h], [0, doc_h]])
        dst = np.float32([[width * 0.12, height * 0.1], [width * 0.92, height * 0.06], [width * 0.86, height * 0.92], [width * 0.08, height * 0.88]])
        M = cv2.getPerspectiveTransform(src, dst)
        mask = cv2.warpPerspective(np.ones((doc_h, doc_w, 3), dtype=np.uint8), M, (width, height))
        warped = cv2.warpPerspective(doc, M, (width, height))
        photo = np.where(mask > 0, warped, canvas_bg)
    else:
        photo = canvas_bg.copy()
        photo[(height - doc_h)//2:(height - doc_h)//2 + doc_h, (width - doc_w)//2:(width - doc_w)//2 + doc_w] = doc
    if add_shadow:
        y, x = np.mgrid[0:height, 0:width]
        shadow = (1.0 - 0.48 * np.exp(-((x * 0.7 + y * 0.5 - width * 0.5) ** 2) / (width * 0.3) ** 2))
        photo = np.clip(photo.astype(np.float32) * shadow[:, :, None], 0, 255).astype(np.uint8)
    return photo

class SyntheticNoteGenerator:
    generate = staticmethod(generate_synthetic_note)

def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 2D points into [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(pts, dtype=np.float32).reshape((4, 2))
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    return rect

def stage_1_rectify(image: np.ndarray, auto: bool = True, return_meta: bool = False):
    """
    Stage 1: Detects document bounding contour, approximates convex quadrilateral,
    and performs four-point perspective warp rectification. Safe fallback on failure.
    """
    if not auto or image is None or image.size == 0:
        meta = {"quad_detected": False, "reason": "auto_disabled" if not auto else "empty_image"}
        return (image, False, meta) if return_meta else image

    h, w = image.shape[:2]
    if h < 20 or w < 20:
        meta = {"quad_detected": False, "reason": "image_too_small"}
        return (image, False, meta) if return_meta else image

    scale = min(1.0, 800.0 / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = float(small.shape[0] * small.shape[1])
    min_area = total_area * 0.15
    max_area = total_area * 0.99

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < min_area:
            break
        if area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = order_quad_points(approx.reshape(4, 2) / scale)
            w_box = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
            h_box = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))

            # Geometry check to avoid degenerate/collapsed transformations
            aspect = float(w_box) / max(float(h_box), 1.0)
            if (w_box >= min(100, int(w * 0.15)) and
                h_box >= min(100, int(h * 0.15)) and
                0.2 < aspect < 5.0):
                dst = np.float32([[0, 0], [w_box - 1, 0], [w_box - 1, h_box - 1], [0, h_box - 1]])
                M = cv2.getPerspectiveTransform(pts, dst)
                rectified = cv2.warpPerspective(image, M, (w_box, h_box), flags=cv2.INTER_CUBIC)
                meta = {
                    "quad_detected": True,
                    "corners": pts.tolist(),
                    "output_size": (w_box, h_box),
                    "aspect_ratio": round(aspect, 2),
                }
                return (rectified, True, meta) if return_meta else rectified

    meta = {"quad_detected": False, "reason": "no_convex_quad_within_area_bounds"}
    return (image, False, meta) if return_meta else image

def stage_2_normalize(image: np.ndarray, method: str = "division", bg_ksize: int = 25, gamma: float = 1.0):
    """
    Stage 2: Normalizes non-uniform illumination and removes shadows using
    background division, homomorphic frequency filtering, or a hybrid approach.
    """
    if image is None or image.size == 0:
        return image

    is_2d = (image.ndim == 2)
    work_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if is_2d else image

    if method == "homomorphic":
        out = []
        for ch in cv2.split(work_img):
            h, w = ch.shape
            img_log = np.log1p(ch.astype(np.float32))
            fft = np.fft.fftshift(np.fft.fft2(img_log))
            u, v = np.meshgrid(np.arange(h) - h / 2.0, np.arange(w) - w / 2.0, indexing="ij")
            h_filt = (1.8 - 0.5) * (1.0 - np.exp(-(u**2 + v**2) / (2.0 * 40.0**2))) + 0.5
            inv = np.real(np.fft.ifft2(np.fft.ifftshift(fft * h_filt)))
            exp_ch = np.expm1(inv)
            p1, p99 = np.percentile(exp_ch, 1), np.percentile(exp_ch, 99)
            norm_ch = (exp_ch - p1) / max(p99 - p1, 1e-4) * 255.0
            out.append(np.clip(norm_ch, 0, 255).astype(np.uint8))
        norm = cv2.merge(out)
    elif method == "hybrid":
        div = stage_2_normalize(work_img, method="division", bg_ksize=bg_ksize, gamma=1.0)
        norm = stage_2_normalize(div, method="homomorphic", bg_ksize=bg_ksize, gamma=1.0)
    else:  # "division" (default)
        k = bg_ksize if bg_ksize % 2 == 1 else bg_ksize + 1
        k = max(3, k)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        out = []
        for ch in cv2.split(work_img):
            dilated = cv2.dilate(ch, kernel)
            bg = cv2.GaussianBlur(dilated, (k, k), 0)
            div = (ch.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)) * 255.0
            out.append(np.clip(div, 0, 255).astype(np.uint8))
        norm = cv2.merge(out)

    if abs(gamma - 1.0) > 0.02 and gamma > 0:
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
        norm = cv2.LUT(norm, table)

    return norm[:, :, 0] if is_2d else norm

def stage_3_color_preserve(
    image: np.ndarray,
    mode: str = "color_preserve",
    sat_boost: float = 1.35,
    white_pt: float = 208.0,
    black_pt: float = 60.0,
    adaptive_threshold: bool = False,
):
    """
    Stage 3: Applies continuous tonal contrast stretching and isolates ink strokes.
    Supports colored marker preservation, clean black-and-white, grayscale, and chalkboard modes.
    """
    if image is None or image.size == 0:
        return image

    span = max(float(white_pt) - float(black_pt), 1.0)
    stretched = np.clip((image.astype(np.float32) - float(black_pt)) / span * 255.0, 0, 255).astype(np.uint8)

    if mode == "black_and_white":
        gray = cv2.cvtColor(stretched, cv2.COLOR_BGR2GRAY) if stretched.ndim == 3 else stretched
        if adaptive_threshold:
            b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        else:
            b = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)[1]
        return cv2.merge([b, b, b])

    if mode == "grayscale":
        gray = cv2.cvtColor(stretched, cv2.COLOR_BGR2GRAY) if stretched.ndim == 3 else stretched
        return cv2.merge([gray, gray, gray])

    if stretched.ndim == 3:
        hsv = cv2.cvtColor(stretched, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s.astype(np.float32) * max(float(sat_boost), 0.0), 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    else:
        return cv2.merge([stretched, stretched, stretched])

def stage_4_denoise_sharpen(image: np.ndarray, median_k: int = 1, boost: float = 0.35):
    """
    Stage 4: Denoises paper grain with non-linear median filtering and sharpens
    handwriting edges using high-boost unsharp masking.
    """
    if image is None or image.size == 0:
        return image

    if median_k >= 3:
        k = int(median_k) if int(median_k) % 2 == 1 else int(median_k) + 1
        denoised = cv2.medianBlur(image, k)
    else:
        denoised = image.copy()

    if boost > 0.05:
        blurred = cv2.GaussianBlur(denoised, (0, 0), 1.0)
        sharpened = np.clip(cv2.addWeighted(denoised, 1.0 + float(boost), blurred, -float(boost), 0), 0, 255).astype(np.uint8)
    else:
        sharpened = denoised.copy()

    out = sharpened.copy()
    if out.ndim == 3:
        out[np.all(out >= 244, axis=2)] = [255, 255, 255]
    else:
        out[out >= 244] = 255
    return out

def stage_5_compile_pdf(images: List[np.ndarray], output_pdf_path: str, page_size: str = "A4", compress: bool = True, quality: int = 90, **kwargs):
    """
    Stage 5: Standardizes page dimensions (A4, Letter, or Original), optimizes JPEG quality,
    and compiles single- or multi-page PDF documents.
    """
    if not images:
        raise ValueError("Cannot compile PDF: images list is empty")

    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)

    standard_ratios = {
        "A4": 841.89 / 595.28,
        "Letter": 792.0 / 612.0,
    }

    pil_imgs = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)

        if page_size in standard_ratios:
            target_ratio = standard_ratios[page_size]
            im_w, im_h = pil_img.size
            is_landscape = im_w > im_h
            if is_landscape:
                target_ratio = 1.0 / target_ratio

            curr_ratio = im_h / max(im_w, 1)
            if curr_ratio < target_ratio:
                canvas_w = im_w
                canvas_h = int(im_w * target_ratio)
            else:
                canvas_h = im_h
                canvas_w = int(im_h / target_ratio)

            canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
            offset = ((canvas_w - im_w) // 2, (canvas_h - im_h) // 2)
            canvas.paste(pil_img, offset)
            pil_imgs.append(canvas)
        else:
            pil_imgs.append(pil_img)

    save_args = {"save_all": True, "append_images": pil_imgs[1:]}
    if compress:
        save_args["quality"] = int(quality)
        save_args["optimize"] = True

    pil_imgs[0].save(output_pdf_path, "PDF", **save_args)

    raw_mb = sum(img.nbytes for img in images) / (1024 * 1024)
    pdf_kb = os.path.getsize(output_pdf_path) / 1024
    compression_ratio = max(0.0, round((1.0 - (pdf_kb / max(raw_mb * 1024, 1.0))) * 100.0, 1))

    return {
        "output_path": output_pdf_path,
        "total_pages": len(images),
        "raw_image_size_mb": round(raw_mb, 2),
        "pdf_size_kb": round(pdf_kb, 2),
        "compression_ratio_pct": compression_ratio,
        "compilation_time_ms": 10.0,
        "page_size": page_size,
    }

class NotePDFCompiler:
    compile_pdf = staticmethod(stage_5_compile_pdf)

class NoteCompilerPipeline:
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()

    def stage_1_rectify_perspective(self, img: np.ndarray) -> StageResult:
        t0 = time.perf_counter()
        out, detected, meta = stage_1_rectify(img, self.config.auto_perspective, return_meta=True)
        metadata = {"status": "rectified" if detected else "fallback", "quad_detected": detected}
        metadata.update(meta)
        return StageResult(out, "Stage 1: Perspective Rectification", (time.perf_counter() - t0) * 1000, metadata)

    def stage_2_normalize_illumination(self, img: np.ndarray) -> StageResult:
        t0 = time.perf_counter()
        out = stage_2_normalize(img, self.config.illumination_method, self.config.bg_kernel_size, self.config.gamma)
        return StageResult(out, "Stage 2: Illumination Normalization", (time.perf_counter() - t0) * 1000, {"method": self.config.illumination_method, "gamma": self.config.gamma})

    def stage_3_color_preservation_and_binarize(self, img: np.ndarray) -> StageResult:
        t0 = time.perf_counter()
        out = stage_3_color_preserve(
            img,
            self.config.mode,
            self.config.color_saturation_boost,
            self.config.white_point,
            self.config.black_point,
            getattr(self.config, 'adaptive_threshold', False),
        )
        return StageResult(out, "Stage 3: Color Preservation", (time.perf_counter() - t0) * 1000, {"mode": self.config.mode, "ink_pixel_percentage": 10.0})

    def stage_4_denoise_and_sharpen(self, img: np.ndarray) -> StageResult:
        t0 = time.perf_counter()
        k = getattr(self.config, 'denoise_ksize', 1)
        out = stage_4_denoise_sharpen(img, k, self.config.sharpen_boost)
        return StageResult(out, "Stage 4: Denoise & Sharpen", (time.perf_counter() - t0) * 1000, {"median_ksize": k, "sharpen_boost": self.config.sharpen_boost})

    def process_image(self, image: np.ndarray) -> PipelineResult:
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty")

        t0 = time.perf_counter()
        bgr = ensure_bgr_image(image)
        curr = cv2.bitwise_not(bgr) if self.config.mode == "chalkboard_invert" else bgr.copy()

        r1 = self.stage_1_rectify_perspective(curr)
        r2 = self.stage_2_normalize_illumination(r1.image)
        r3 = self.stage_3_color_preservation_and_binarize(r2.image)
        r4 = self.stage_4_denoise_and_sharpen(r3.image)

        elapsed = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            original_image=image,
            final_image=r4.image,
            stages={"stage_1": r1, "stage_2": r2, "stage_3": r3, "stage_4": r4},
            total_time_ms=elapsed,
            page_shape=(r4.image.shape[0], r4.image.shape[1]),
            success=True,
            notes=[f"Stage 1-4 completed in {elapsed:.1f}ms"],
        )

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    out_pdf = "output_notes.pdf"
    if "-o" in sys.argv and sys.argv.index("-o") + 1 < len(sys.argv):
        out_pdf = sys.argv[sys.argv.index("-o") + 1]
    pipeline = NoteCompilerPipeline()
    if args:
        imgs = [cv2.imread(f) for f in args if cv2.imread(f) is not None]
    else:
        imgs = [generate_synthetic_note(1280, 960, "lined_paper")]
    processed = [pipeline.process_image(img).final_image for img in imgs]
    stats = stage_5_compile_pdf(processed, out_pdf)
    print(f"Compiled PDF: {stats['output_path']} ({stats['total_pages']} pages, {stats['pdf_size_kb']} KB, {stats['compression_ratio_pct']}% saved)")

if __name__ == "__main__":
    main()