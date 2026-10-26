import os
import io
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from note_compiler import (
    generate_synthetic_note,
    NoteCompilerPipeline,
    ProcessingConfig,
    stage_5_compile_pdf,
    compute_image_metrics,
    ensure_bgr_image,
)

st.set_page_config(
    page_title="Whiteboard & Note-to-PDF Compiler",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Pitch-black minimal dark theme styling
st.markdown(
    """
    <style>
    .stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #FFFFFF !important;
    }
    button, button *, div.stDownloadButton button, div.stDownloadButton button * {
        color: #000000 !important;
    }
    button, div.stDownloadButton > button {
        background-color: #FFFFFF !important;
        border: 1px solid #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 4px !important;
    }
    button:hover, button:hover *, div.stDownloadButton > button:hover, div.stDownloadButton > button:hover * {
        background-color: #E5E5E5 !important;
        color: #000000 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #0A0A0A !important;
        border: 1px solid #262626 !important;
        border-radius: 6px !important;
        padding: 12px !important;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 1.5rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #888888 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stFileUploader"] {
        background-color: #0A0A0A !important;
        border: 1px dashed #333333 !important;
        border-radius: 6px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000 !important;
        border-bottom: 1px solid #262626 !important;
        gap: 1.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #888888 !important;
        font-weight: 500 !important;
        padding: 0.75rem 0.5rem !important;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom-color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    .stat-chip {
        display: inline-block;
        background-color: #141414;
        border: 1px solid #262626;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 0.8rem;
        color: #AAAAAA;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .stat-chip strong {
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("Whiteboard & Lecture Note-to-PDF Compiler")
st.caption("5-Stage CV Pipeline: Perspective Rectification, Illumination Normalization, Color Preservation, Denoise & Sharpen, Multi-Page PDF")

# Reset to defaults handling
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = 0

with st.sidebar:
    st.header("Pipeline Controls")

    mode = st.selectbox(
        "Document Mode",
        ["color_preserve", "black_and_white", "grayscale", "chalkboard_invert"],
        index=0,
        help="Select processing mode: preserve ink colors, clean binary B&W, continuous grayscale, or invert chalkboards",
        key=f"mode_{st.session_state.reset_trigger}",
    )

    auto_persp = st.checkbox(
        "Auto-Perspective Rectify",
        value=True,
        help="Detect document contour and rectify camera angle",
        key=f"persp_{st.session_state.reset_trigger}",
    )

    with st.expander("Illumination & Background", expanded=False):
        illum_method = st.selectbox(
            "Correction Method",
            ["division", "homomorphic", "hybrid"],
            index=0,
            help="Division: spatial background morph/blur. Homomorphic: log-frequency domain high-pass.",
            key=f"illum_{st.session_state.reset_trigger}",
        )
        bg_kernel = st.slider(
            "Background Filter Radius",
            15, 65, 25, 2,
            help="Kernel size for spatial background estimation",
            key=f"bgk_{st.session_state.reset_trigger}",
        )
        gamma = st.slider(
            "Gamma Correction",
            0.5, 2.0, 1.0, 0.05,
            help="Tonal gamma (<1.0 brightens midtones, >1.0 darkens)",
            key=f"gamma_{st.session_state.reset_trigger}",
        )

    white_pt = st.slider(
        "Paper & Line Clean Level",
        160.0, 240.0, 208.0, 2.0,
        help="Higher values whiten subtle paper texture and remove shadows",
        key=f"white_{st.session_state.reset_trigger}",
    )

    black_pt = st.slider(
        "Ink Darkening / Depth",
        15.0, 100.0, 60.0, 5.0,
        help="Lower values preserve fine ink strokes; higher values deepen text contrast",
        key=f"black_{st.session_state.reset_trigger}",
    )

    with st.expander("Color, Edges & Noise", expanded=False):
        sat_boost = st.slider(
            "Color Saturation Boost",
            1.0, 2.5, 1.35, 0.05,
            help="Vibrancy multiplier for colored pens/markers",
            key=f"sat_{st.session_state.reset_trigger}",
        )
        sharpen_boost = st.slider(
            "Text Edge Crisp (Sharpen)",
            0.0, 0.8, 0.35, 0.05,
            help="High-boost unsharp masking intensity",
            key=f"sharp_{st.session_state.reset_trigger}",
        )
        denoise_level = st.select_slider(
            "Denoise Filter (Median)",
            options=[1, 3, 5],
            value=1,
            format_func=lambda x: "Off (1px)" if x == 1 else f"Mild ({x}px)" if x == 3 else f"Strong ({x}px)",
            help="Removes sensor salt-and-pepper grain without blurring text strokes",
            key=f"denoise_{st.session_state.reset_trigger}",
        )
        adaptive_thresh = False
        if mode == "black_and_white":
            adaptive_thresh = st.checkbox(
                "Adaptive Thresholding",
                value=False,
                help="Local Gaussian adaptive binarization for uneven lighting",
                key=f"adapt_{st.session_state.reset_trigger}",
            )

    with st.expander("PDF Compilation Options", expanded=False):
        pdf_page_size = st.selectbox(
            "PDF Page Layout",
            ["A4", "Letter", "Original"],
            index=0,
            help="Standardizes page dimensions or retains original image aspect ratio",
            key=f"page_size_{st.session_state.reset_trigger}",
        )
        enable_compress = st.checkbox(
            "Compress PDF Document",
            value=True,
            help="Optimizes PDF with JPEG compression",
            key=f"compress_{st.session_state.reset_trigger}",
        )
        pdf_quality = st.slider(
            "JPEG Quality",
            40, 95, 80, 5,
            help="Higher values preserve visual detail; lower values create smaller PDFs",
            key=f"quality_{st.session_state.reset_trigger}",
        ) if enable_compress else 95

    if st.button("Reset to Defaults", use_container_width=True):
        st.session_state.reset_trigger += 1
        st.rerun()

# Upload and synthetic note section
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("<div style='font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem;'>Upload Notes or Whiteboard Photos</div>", unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Upload Notes or Whiteboard Photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
with c2:
    st.markdown("<div style='font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem;'>Or Test with Synthetic Note</div>", unsafe_allow_html=True)
    synth_type = st.selectbox("Note Type", ["lined_paper", "whiteboard", "chalkboard"], label_visibility="collapsed")
    gen_btn = st.button("Generate Synthetic Note", type="primary", use_container_width=True)

images = []
input_bytes_total = 0
failed_uploads = []

if uploads:
    for u in uploads:
        data = u.read()
        if not data:
            failed_uploads.append((u.name, "File is empty"))
            continue
        try:
            arr = np.asarray(bytearray(data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                failed_uploads.append((u.name, "Unable to decode image format"))
                continue
            bgr = ensure_bgr_image(img)
            input_bytes_total += len(data)
            images.append((u.name, bgr))
        except Exception as e:
            failed_uploads.append((u.name, str(e)))

    if failed_uploads:
        for fname, err in failed_uploads:
            st.warning(f"Skipped {fname}: {err}")

elif gen_btn or not images:
    sample = generate_synthetic_note(1280, 960, note_type=synth_type, seed=42)
    images = [(f"synthetic_{synth_type}.jpg", sample)]

if not images:
    st.info("Please upload note images or click 'Generate Synthetic Note' to test the pipeline.")
    st.stop()

# Configuration and pipeline execution
cfg = ProcessingConfig(
    auto_perspective=auto_persp,
    illumination_method=illum_method,
    bg_kernel_size=bg_kernel,
    white_point=white_pt,
    black_point=black_pt,
    gamma=gamma,
    mode=mode,
    color_saturation_boost=sat_boost,
    sharpen_boost=sharpen_boost,
    denoise_ksize=denoise_level,
    adaptive_threshold=adaptive_thresh,
    compress_pdf=enable_compress,
    page_size=pdf_page_size,
    jpeg_quality=pdf_quality,
)

pipeline = NoteCompilerPipeline(cfg)

with st.spinner("Processing document images through CV pipeline..."):
    results = []
    for fname, img in images:
        try:
            res = pipeline.process_image(img)
            results.append((fname, res))
        except Exception as ex:
            st.error(f"Error processing {fname}: {ex}")

if not results:
    st.error("No images could be processed.")
    st.stop()

# PDF Compilation (Stage 5)
with tempfile.TemporaryDirectory() as tmp_dir:
    pdf_path = os.path.join(tmp_dir, "notes.pdf")
    pdf_stats = stage_5_compile_pdf(
        [r.final_image for _, r in results],
        pdf_path,
        page_size=pdf_page_size,
        compress=enable_compress,
        quality=pdf_quality,
    )
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

# Quality metrics calculation
first_result = results[0][1]
orig_metrics = compute_image_metrics(first_result.original_image)
final_metrics = compute_image_metrics(first_result.final_image)

# Metric Monitors Bar
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Pages Compiled", pdf_stats["total_pages"])
m2.metric("Processing Time", f"{sum(r.total_time_ms for _, r in results):.0f} ms")
m3.metric("PDF Output Size", f"{pdf_stats['pdf_size_kb']} KB")

if input_bytes_total > 0 and enable_compress:
    comp_pct = max(0.0, (1.0 - (pdf_stats["pdf_size_kb"] / (input_bytes_total / 1024.0))) * 100.0)
    m4.metric("Compression vs Input", f"{comp_pct:.1f}%")
else:
    m4.metric("Raw Compression", f"{pdf_stats['compression_ratio_pct']}%" if enable_compress else "Lossless")

m5.metric("Edge Sharpness", f"{final_metrics['sharpness_index']:.0f}", delta=f"{final_metrics['sharpness_index'] - orig_metrics['sharpness_index']:+.0f}")

# PDF Download Action
st.download_button(
    f"Download PDF Document ({pdf_stats['total_pages']} Pages, {pdf_stats['pdf_size_kb']} KB)",
    data=pdf_bytes,
    file_name="compiled_notes.pdf",
    mime="application/pdf",
    type="primary",
    use_container_width=True,
)

# Page selector for multi-page documents
if len(results) > 1:
    sel_idx = st.selectbox("Inspect Page", range(len(results)), format_func=lambda i: f"Page {i+1}: {results[i][0]}")
else:
    sel_idx = 0

curr_fname, curr_res = results[sel_idx]
curr_orig_meta = compute_image_metrics(curr_res.original_image)
curr_final_meta = compute_image_metrics(curr_res.final_image)

tab1, tab2, tab3 = st.tabs(["Before & After", "5-Stage Breakdown", "Batch Overview & Export"])

# Tab 1: Before & After Comparison
with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Original Capture")
        st.markdown(
            f"""
            <div style='margin-bottom: 8px;'>
                <span class='stat-chip'>Dimensions: <strong>{curr_res.original_image.shape[1]} × {curr_res.original_image.shape[0]}</strong></span>
                <span class='stat-chip'>Sharpness: <strong>{curr_orig_meta['sharpness_index']}</strong></span>
                <span class='stat-chip'>White Px: <strong>{curr_orig_meta['white_background_pct']}%</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(cv2.cvtColor(curr_res.original_image, cv2.COLOR_BGR2RGB), width="stretch")

    with col_b:
        st.subheader("Cleaned Document")
        st.markdown(
            f"""
            <div style='margin-bottom: 8px;'>
                <span class='stat-chip'>Dimensions: <strong>{curr_res.final_image.shape[1]} × {curr_res.final_image.shape[0]}</strong></span>
                <span class='stat-chip'>Sharpness: <strong>{curr_final_meta['sharpness_index']}</strong></span>
                <span class='stat-chip'>White Px: <strong>{curr_final_meta['white_background_pct']}%</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(cv2.cvtColor(curr_res.final_image, cv2.COLOR_BGR2RGB), width="stretch")

# Tab 2: 5-Stage Pipeline Breakdown
with tab2:
    stg = curr_res.stages
    s1_meta = stg["stage_1"].metadata
    s2_meta = stg["stage_2"].metadata
    s3_meta = stg["stage_3"].metadata
    s4_meta = stg["stage_4"].metadata

    s_cols1 = st.columns(2)
    with s_cols1[0]:
        rect_status = "Rectified" if s1_meta.get("quad_detected") else "Fallback (Retained original crop)"
        st.markdown(f"**Stage 1: Perspective Rectification** — *{stg['stage_1'].execution_time_ms:.1f} ms*<br><span style='font-size: 0.85rem; color: #888;'>Status: {rect_status}</span>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(stg["stage_1"].image, cv2.COLOR_BGR2RGB), width="stretch")

    with s_cols1[1]:
        st.markdown(f"**Stage 2: Illumination Normalization** — *{stg['stage_2'].execution_time_ms:.1f} ms*<br><span style='font-size: 0.85rem; color: #888;'>Method: {s2_meta.get('method', 'division')} | Gamma: {s2_meta.get('gamma', 1.0)}</span>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(stg["stage_2"].image, cv2.COLOR_BGR2RGB), width="stretch")

    s_cols2 = st.columns(2)
    with s_cols2[0]:
        st.markdown(f"**Stage 3: Color Preservation & Contrast** — *{stg['stage_3'].execution_time_ms:.1f} ms*<br><span style='font-size: 0.85rem; color: #888;'>Mode: {s3_meta.get('mode', 'color_preserve')}</span>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(stg["stage_3"].image, cv2.COLOR_BGR2RGB), width="stretch")

    with s_cols2[1]:
        st.markdown(f"**Stage 4: Denoised & Sharpened** — *{stg['stage_4'].execution_time_ms:.1f} ms*<br><span style='font-size: 0.85rem; color: #888;'>Boost: {s4_meta.get('sharpen_boost', 0.35)} | Median: {s4_meta.get('median_ksize', 1)}px</span>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(stg["stage_4"].image, cv2.COLOR_BGR2RGB), width="stretch")

    st.markdown("---")
    st.markdown(f"**Stage 5: Standard Page Layout & Multi-Page PDF Compilation** — *{pdf_stats['compilation_time_ms']:.1f} ms*")
    st.markdown(
        f"""
        <div style='margin-bottom: 12px;'>
            <span class='stat-chip'>Format: <strong>{pdf_page_size}</strong></span>
            <span class='stat-chip'>Compression: <strong>{pdf_stats['compression_ratio_pct']}%</strong></span>
            <span class='stat-chip'>Page Count: <strong>{pdf_stats['total_pages']}</strong></span>
            <span class='stat-chip'>Size: <strong>{pdf_stats['pdf_size_kb']} KB</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Tab 3: Batch Overview & Single Image Downloads
with tab3:
    st.subheader("Processed Pages & Image Downloads")
    for idx, (fn, res) in enumerate(results):
        g_c1, g_c2, g_c3 = st.columns([1, 2, 1])
        with g_c1:
            st.image(cv2.cvtColor(res.final_image, cv2.COLOR_BGR2RGB), width="stretch")
        with g_c2:
            st.markdown(f"**Page {idx+1}: {fn}**")
            st.caption(f"Dimensions: {res.final_image.shape[1]} × {res.final_image.shape[0]} px | Pipeline time: {res.total_time_ms:.1f} ms")
        with g_c3:
            # Encode single image to PNG for download
            is_success, buffer = cv2.imencode(".png", res.final_image)
            if is_success:
                st.download_button(
                    label=f"Download Page {idx+1} (PNG)",
                    data=buffer.tobytes(),
                    file_name=f"page_{idx+1}_cleaned.png",
                    mime="image/png",
                    key=f"dl_page_{idx}",
                    use_container_width=True,
                )
