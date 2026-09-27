import numpy as np
import cv2
import streamlit as st
import streamlit.components.v1 as components

# ------------------------------------------------------------------------------
# Page Configuration & Enterprise Design System
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="VectorCraft AI Studio | Enterprise Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: #090d16;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    section[data-testid="stSidebar"] {
        background: #0e1422;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .stSlider label, 
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-weight: 500;
        font-size: 0.9rem;
    }

    .enterprise-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.2rem 2rem;
        background: rgba(14, 20, 34, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        backdrop-filter: blur(16px);
        margin-bottom: 2rem;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .nav-logo {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        padding: 10px 14px;
        border-radius: 12px;
        font-size: 1.5rem;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
    }
    .status-badge {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .studio-panel {
        background: #0e1422;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background: #131b2e;
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 14px 18px;
        border-radius: 12px;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700;
        font-size: 1.5rem;
    }

    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.35);
        width: 100%;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
    }

    .enterprise-footer {
        text-align: center;
        padding: 2.5rem 0 1rem 0;
        font-size: 0.85rem;
        color: #64748b;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 4rem;
        text-transform: uppercase;
    }
    .enterprise-footer span {
        color: #38bdf8;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Dual Vectorization Core (Sketch vs. Photo Engine)
# ------------------------------------------------------------------------------
def process_sketch(image, block_size, c_val, stroke_width):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    if block_size % 2 == 0:
        block_size += 1
        
    binary = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        max(3, block_size), 
        c_val
    )
    
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = image.shape[:2]
    svg_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%">']
    svg_parts.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    
    valid_contours = 0
    total_vertices = 0
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 5:
            epsilon = 0.001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) > 2:
                points = " ".join([f"{pt[0][0]},{pt[0][1]}" for pt in approx])
                svg_parts.append(f'<polygon points="{points}" fill="none" stroke="#111827" stroke-width="{stroke_width}" stroke-linejoin="round"/>')
                valid_contours += 1
                total_vertices += len(approx)
                
    svg_parts.append('</svg>')
    return "\n".join(svg_parts), valid_contours, total_vertices, binary

def process_photo(image, num_colors):
    h, w = image.shape[:2]
    if w > 600:
        scale = 600 / w
        image = cv2.resize(image, (600, int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]

    data = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(image.shape)
    
    svg_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%">']
    svg_parts.append(f'<rect width="{w}" height="{h}" fill="#0b0f19"/>')
    
    total_vertices = 0
    for color in centers:
        mask = cv2.inRange(quantized, color, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 30:
                approx = cv2.approxPolyDP(cnt, 0.003 * cv2.arcLength(cnt, True), True)
                if len(approx) >= 3:
                    points = " ".join([f"{pt[0][0]},{pt[0][1]}" for pt in approx])
                    svg_parts.append(f'<polygon points="{points}" fill="{hex_color}" stroke="{hex_color}" stroke-width="0.5"/>')
                    total_vertices += len(approx)
                    
    svg_parts.append('</svg>')
    return "\n".join(svg_parts), len(centers), total_vertices, quantized

# ------------------------------------------------------------------------------
# Navigation Header
# ------------------------------------------------------------------------------
st.markdown("""
    <div class="enterprise-nav">
        <div class="nav-brand">
            <div class="nav-logo">⚡</div>
            <div>
                <h2 style="margin: 0; font-size: 1.4rem; font-weight: 800; color: #ffffff;">VectorCraft AI Studio</h2>
                <p style="margin: 0; font-size: 0.8rem; color: #94a3b8;">Precision Sketch & Image Vectorization Suite</p>
            </div>
        </div>
        <div class="status-badge">🟢 Engine Online</div>
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ Studio Configuration")
asset_mode = st.sidebar.radio("Asset Type Mode", ["Sketch / Line Art (Stroke)", "Photograph / Render (Image)"])

if "Sketch" in asset_mode:
    st.sidebar.markdown("#### Sketch Parameters")
    block_size = st.sidebar.slider("Threshold Block Size", 3, 31, 11, step=2)
    c_val = st.sidebar.slider("Threshold Constant (C)", 1, 15, 3)
    stroke_width = st.sidebar.slider("Vector Stroke Weight", 0.5, 3.0, 1.2)
else:
    st.sidebar.markdown("#### Photo Parameters")
    num_colors = st.sidebar.slider("Color Layers", 2, 8, 4)

# ------------------------------------------------------------------------------
# Workspace
# ------------------------------------------------------------------------------
st.markdown("""
    <div class="studio-panel">
        <h3 style="margin-top: 0; font-size: 1.15rem; color: #f8fafc; margin-bottom: 6px;">📥 Upload Graphic Asset</h3>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">Upload a pencil sketch for clean strokes, or a photo for color posterization.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Drag and drop files here", 
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    label_visibility="collapsed"
)

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Failed to decode image file.")
        st.stop()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
            <div class="studio-panel">
                <h4 style="margin-top: 0; color: #38bdf8; font-size: 1rem;">📂 Source Raster Input</h4>
            """, unsafe_allow_html=True)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    try:
        if "Sketch" in asset_mode:
            svg_string, count, vertices, preview_matrix = process_sketch(image, block_size, c_val, stroke_width)
            mode_label = "Sketch Paths"
        else:
            svg_string, count, vertices, preview_matrix = process_photo(image, num_colors)
            mode_label = "Color Layers"
    except Exception as e:
        st.error(f"Pipeline execution failed: {e}")
        st.stop()

    with col2:
        st.markdown("""
            <div class="studio-panel">
                <h4 style="margin-top: 0; color: #38bdf8; font-size: 1rem;">✨ Clean Vectorized Output (.SVG)</h4>
            """, unsafe_allow_html=True)
        
        components.html(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; background-color: {'#ffffff' if 'Sketch' in asset_mode else '#0b0f19'}; border-radius: 12px; padding: 12px; height: 350px; overflow: auto; box-shadow: inset 0 2px 6px rgba(0,0,0,0.3);">
                {svg_string}
            </div>
            """,
            height=380,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Metrics Bar
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label=mode_label, value=count)
    with m2:
        st.metric(label="Anchor Vertices", value=vertices)
    with m3:
        st.metric(label="Export Format", value="Scalable Vector (.svg)")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    col_dl, col_dummy = st.columns([2, 1])
    with col_dl:
        st.download_button(
            label="📥 Download Production-Ready Vector Asset",
            data=svg_string,
            file_name=uploaded_file.name.rsplit(".", 1)[0] + "_vectorized.svg",
            mime="image/svg+xml",
        )
else:
    st.markdown("""
        <div class="studio-panel" style="text-align: center; padding: 4rem 2rem; border: 2px dashed rgba(255, 255, 255, 0.1); background: rgba(14, 20, 34, 0.4);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
            <h3 style="color: #ffffff; margin-bottom: 6px; font-size: 1.25rem;">Workspace Ready for Compilation</h3>
            <p style="color: #94a3b8; font-size: 0.95rem; max-width: 500px; margin: 0 auto;">Select your asset type in the sidebar (Sketch vs Photo) and upload your file above.</p>
        </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Developer Footer Credit
# ------------------------------------------------------------------------------
st.markdown("""
    <div class="enterprise-footer">
        Architected & Developed by <span>Azan Kareem</span> &bull; AI Vectorization Suite
    </div>
""", unsafe_allow_html=True)