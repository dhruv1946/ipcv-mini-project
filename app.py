import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from note_compiler import (
    generate_synthetic_note,
    NoteCompilerPipeline,
    ProcessingConfig,
    stage_5_compile_pdf,
)

st.set_page_config(page_title="Whiteboard & Note-to-PDF Compiler", page_icon="📄", layout="wide")

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
    }
    div[data-testid="stMetricLabel"] {
        color: #888888 !important;
    }
    div[data-testid="stFileUploader"] {
        background-color: #0A0A0A !important;
        border: 1px dashed #333333 !important;
        border-radius: 6px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000 !important;
        border-bottom: 1px solid #262626 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #888888 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom-color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Whiteboard & Lecture Note-to-PDF Compiler")
st.caption("5-Stage CV Pipeline: Perspective Rectification, Illumination Normalization, Color Preservation, Denoise & Sharpen, Multi-Page PDF")

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Document Mode", ["color_preserve", "black_and_white", "chalkboard_invert"], index=0)
    auto_persp = st.checkbox("Auto-Perspective Rectify", value=True)
    white_pt = st.slider("Paper & Line Clean Level", 175.0, 235.0, 208.0, 2.0)
    black_pt = st.slider("Ink Darkening / Depth", 20.0, 100.0, 60.0, 5.0)
    sat_boost = st.slider("Color Saturation Boost", 1.0, 2.0, 1.35, 0.05)
    sharpen_boost = st.slider("Text Edge Crisp", 0.0, 0.8, 0.35, 0.05)
    enable_compress = st.checkbox("Compress PDF Document", value=True)
    pdf_quality = st.slider("Quality (Lower = Smaller PDF, Higher = Larger File)", 40, 95, 80, 5) if enable_compress else 100

c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("<div style='font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem;'>Upload Notes or Whiteboard Photos</div>", unsafe_allow_html=True)
    uploads = st.file_uploader("Upload Notes or Whiteboard Photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, label_visibility="collapsed")
with c2:
    st.markdown("<div style='font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem;'>Or Test with Synthetic Note</div>", unsafe_allow_html=True)
    synth_type = st.selectbox("Note Type", ["lined_paper", "whiteboard", "chalkboard"], label_visibility="collapsed")
    gen_btn = st.button("Generate Synthetic Note", type="primary", use_container_width=True)

images = []
input_bytes_total = 0
if uploads:
    for u in uploads:
        data = u.read()
        input_bytes_total += len(data)
        img = cv2.imdecode(np.asarray(bytearray(data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            images.append((u.name, img))
elif gen_btn or not images:
    sample = generate_synthetic_note(1280, 960, note_type=synth_type, seed=42)
    images = [(f"synthetic_{synth_type}.jpg", sample)]

cfg = ProcessingConfig(
    auto_perspective=auto_persp,
    white_point=white_pt,
    black_point=black_pt,
    mode=mode,
    color_saturation_boost=sat_boost,
    sharpen_boost=sharpen_boost,
    compress_pdf=enable_compress,
    jpeg_quality=pdf_quality,
)

pipeline = NoteCompilerPipeline(cfg)
results = [pipeline.process_image(img) for _, img in images]

with tempfile.TemporaryDirectory() as tmp_dir:
    pdf_path = os.path.join(tmp_dir, "notes.pdf")
    pdf_stats = stage_5_compile_pdf([r.final_image for r in results], pdf_path, compress=enable_compress, quality=pdf_quality)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Pages", pdf_stats["total_pages"])
m2.metric("Processing Time", f"{sum(r.total_time_ms for r in results):.0f} ms")
m3.metric("PDF Size", f"{pdf_stats['pdf_size_kb']} KB")
if input_bytes_total > 0 and enable_compress:
    comp_pct = (1.0 - (pdf_stats["pdf_size_kb"] / (input_bytes_total / 1024.0))) * 100.0
    m4.metric("Compression vs Upload", f"{comp_pct:.1f}%")
else:
    m4.metric("Compression", f"{pdf_stats['compression_ratio_pct']}%" if enable_compress else "Lossless")

st.download_button(
    f"Download PDF Document ({pdf_stats['total_pages']} Pages, {pdf_stats['pdf_size_kb']} KB)",
    data=pdf_bytes,
    file_name="output_notes.pdf",
    mime="application/pdf",
    type="primary",
    use_container_width=True,
)

tab1, tab2 = st.tabs(["Before & After", "5-Stage Breakdown"])
sel_idx = 0 if len(results) == 1 else st.selectbox("Select Page", range(len(results)), format_func=lambda i: images[i][0])
curr_res = results[sel_idx]

with tab1:
    col_a, col_b = st.columns(2)
    col_a.subheader("Original Capture")
    col_a.image(cv2.cvtColor(curr_res.original_image, cv2.COLOR_BGR2RGB), width="stretch")
    col_b.subheader("Cleaned Document")
    col_b.image(cv2.cvtColor(curr_res.final_image, cv2.COLOR_BGR2RGB), width="stretch")

with tab2:
    stg = curr_res.stages
    s_cols1 = st.columns(2)
    s_cols1[0].write("*Stage 1: Rectified*")
    s_cols1[0].image(cv2.cvtColor(stg["stage_1"].image, cv2.COLOR_BGR2RGB), width="stretch")
    s_cols1[1].write("*Stage 2: Illumination Normalized*")
    s_cols1[1].image(cv2.cvtColor(stg["stage_2"].image, cv2.COLOR_BGR2RGB), width="stretch")
    s_cols2 = st.columns(2)
    s_cols2[0].write("*Stage 3: Color Preserved*")
    s_cols2[0].image(cv2.cvtColor(stg["stage_3"].image, cv2.COLOR_BGR2RGB), width="stretch")
    s_cols2[1].write("*Stage 4: Denoised & Sharpened*")
    s_cols2[1].image(cv2.cvtColor(stg["stage_4"].image, cv2.COLOR_BGR2RGB), width="stretch")
