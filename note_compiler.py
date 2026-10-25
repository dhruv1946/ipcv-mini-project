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
    illumination_method: str = "division"
    bg_kernel_size: int = 25
    gamma: float = 1.0
    sharpen_boost: float = 0.35
    page_size: str = "A4"

@dataclass
class StageResult:
    image: np.ndarray
    stage_name: str
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

def generate_synthetic_note(width=1200, height=900, note_type="lined_paper", add_shadow=True, add_perspective=True, seed=42):
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
        photo = canvas_bg
        photo[(height - doc_h)//2:(height - doc_h)//2 + doc_h, (width - doc_w)//2:(width - doc_w)//2 + doc_w] = doc
    if add_shadow:
        y, x = np.mgrid[0:height, 0:width]
        shadow = (1.0 - 0.48 * np.exp(-((x * 0.7 + y * 0.5 - width * 0.5) ** 2) / (width * 0.3) ** 2))
        photo = np.clip(photo.astype(np.float32) * shadow[:, :, None], 0, 255).astype(np.uint8)
    return photo

class SyntheticNoteGenerator:
    generate = staticmethod(generate_synthetic_note)

def order_quad_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

def stage_1_rectify(image, auto=True):
    if not auto:
        return image
    h, w = image.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(image, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (small.shape[0] * small.shape[1]) * 0.18
    max_area = (small.shape[0] * small.shape[1]) * 0.98
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < min_area:
            break
        if area > max_area:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = order_quad_points(approx.reshape(4, 2) / scale)
            w_box = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
            h_box = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))
            if w_box > 200 and h_box > 200:
                dst = np.float32([[0, 0], [w_box - 1, 0], [w_box - 1, h_box - 1], [0, h_box - 1]])
                return cv2.warpPerspective(image, cv2.getPerspectiveTransform(pts, dst), (w_box, h_box), flags=cv2.INTER_CUBIC)
    return image

def stage_2_normalize(image, method="division", bg_ksize=25, gamma=1.0):
    if method == "homomorphic":
        out = []
        for ch in cv2.split(image):
            h, w = ch.shape
            img_log = np.log1p(ch.astype(np.float32))
            fft = np.fft.fftshift(np.fft.fft2(img_log))
            u, v = np.meshgrid(np.arange(h) - h / 2.0, np.arange(w) - w / 2.0, indexing="ij")
            h_filt = (1.8 - 0.5) * (1.0 - np.exp(-(u**2 + v**2) / (2.0 * 40.0**2))) + 0.5
            inv = np.real(np.fft.ifft2(np.fft.ifftshift(fft * h_filt)))
            exp_ch = np.expm1(inv)
            p1, p99 = np.percentile(exp_ch, 1), np.percentile(exp_ch, 99)
            out.append(np.clip((exp_ch - p1) / (p99 - p1 + 1e-6) * 255.0, 0, 255).astype(np.uint8))
        norm = cv2.merge(out)
    elif method == "hybrid":
        div = stage_2_normalize(image, method="division", bg_ksize=bg_ksize, gamma=1.0)
        norm = stage_2_normalize(div, method="homomorphic", bg_ksize=bg_ksize, gamma=1.0)
    else:
        k = bg_ksize if bg_ksize % 2 == 1 else bg_ksize + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        out = []
        for ch in cv2.split(image):
            bg = cv2.GaussianBlur(cv2.dilate(ch, kernel), (k, k), 0)
            div = (ch.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)) * 255.0
            out.append(np.clip(div, 0, 255).astype(np.uint8))
        norm = cv2.merge(out)

    if abs(gamma - 1.0) > 0.02:
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
        norm = cv2.LUT(norm, table)
    return norm

def stage_4_denoise_sharpen(image, median_k=1, boost=0.35):
    if boost > 0.05:
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
        sharpened = np.clip(cv2.addWeighted(image, 1.0 + boost, blurred, -boost, 0), 0, 255).astype(np.uint8)
    else:
        sharpened = image
    sharpened[np.all(sharpened >= 244, axis=2)] = [255, 255, 255]
    return sharpened
