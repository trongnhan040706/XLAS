import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import streamlit as st    

st.set_page_config(
    page_title="HUIT · Hệ Thống Khử Nhiễu & Lọc Không Gian Số",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=Fira+Code:wght@400;600&display=swap');
    .stApp {
        background: radial-gradient(circle at 15% 0%, #103656 0%, #061426 45%, #04101d 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .huit-header {
        background: linear-gradient(135deg, #09233b 0%, #061626 100%);
        border-bottom: 3px solid #1cc9df;
        border-radius: 14px;
        padding: 16px 22px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    .huit-title-box { flex-grow: 1; }
    .huit-school { font-size: 19px; font-weight: 900; color: #f8fafc; margin-bottom: 3px; }
    .huit-dept { font-size: 13px; color: #b8cbd9; margin-bottom: 3px; }
    .huit-course { font-size: 13px; color: #5ce4f4; font-weight: 700; letter-spacing: 0.8px; }
    .hero-box {
        background: linear-gradient(145deg, rgba(14, 45, 75, 0.7), rgba(7, 27, 48, 0.9));
        border: 1px solid #285d82; border-radius: 18px; padding: 22px 26px; margin-bottom: 25px;
    }
    .hero-badge { color: #5ce4f4; font-weight: 900; font-size: 13px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; }
    .hero-h1 { font-size: 30px; font-weight: 900; color: #ffffff; margin: 0 0 10px 0; }
    .hero-p { font-size: 14.5px; color: #cbd8e2; line-height: 1.6; margin: 0; }
    .ref-card { background: #081d33; border: 1px solid #2a5877; border-radius: 14px; padding: 14px 16px; height: 100%; box-shadow: 0 4px 14px rgba(0,0,0,0.2); }
    .ref-eq { font-family: 'Fira Code', monospace; font-size: 12px; background: #051424; border-radius: 6px; padding: 6px 8px; margin-bottom: 8px; color: #5ce4f4; border: 1px solid #1c3d59; }
    .ref-note { font-size: 12px; color: #a2b7c7; line-height: 1.45; }
    .theory-box { border-left: 4px solid #25cbe6; background: #081d33; border-radius: 0 10px 10px 0; padding: 12px 16px; margin-bottom: 12px; }
    .theory-box b { color: #25cbe6; font-size: 14px; display: block; margin-bottom: 4px; }
    .theory-box p { margin: 0; font-size: 13px; color: #d1dde6; line-height: 1.5; }
    .code-chip { font-family: 'Fira Code', monospace; font-size: 12px; background: #04101d; color: #5ce4f4; border: 1px solid #1c4361; border-radius: 8px; padding: 10px 14px; margin-top: 10px; }
    .conclusion-tag { background: rgba(37, 203, 230, 0.12); border: 1px solid rgba(37, 203, 230, 0.4); border-radius: 8px; padding: 10px 14px; font-size: 13px; font-weight: 700; color: #e2f4f8; margin-top: 10px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# 1. Header Banner HUIT
st.markdown("""
<div class="huit-header">
    <div style="width: 55px; height: 55px; border-radius: 50%; background: #1cc9df; color: #061426; display: flex; align-items: center; justify-content: center; font-weight: 900; font-size: 16px;">HUIT</div>
    <div class="huit-title-box">
        <div class="huit-school">TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP.HCM</div>
        <div class="huit-dept">Khoa Công nghệ thông tin · Bộ môn Khoa học dữ liệu và AI</div>
        <div class="huit-course">XỬ LÝ ẢNH SỐ · ĐỒ ÁN MINH HỌA CÁC BỘ LỌC MIỀN KHÔNG GIAN</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Hero Presentation Box
st.markdown("""
<div class="hero-box">
    <div class="hero-badge">BÁO CÁO THỰC NGHIỆM ĐỒ ÁN MÔN HỌC</div>
    <div class="hero-h1">HỆ THỐNG KHỬ NHIỄU ẢNH SỐ</div>
    <p class="hero-p">
        Hệ thống tích hợp đầy đủ cơ sở toán học, ma trận kernel tham chiếu, đồ thị mặt phẳng cắt lát phân tích mức xám 
        và bộ chỉ số định lượng tiêu chuẩn quốc tế (PSNR, SSIM, Processing Time) nhằm đánh giá 
        chính xác tương quan giữa <b>Mean, Gaussian, Median và Bilateral Filter</b> trên các mô hình nhiễu thực tế.
    </p>
</div>
""", unsafe_allow_html=True)

# 3. Bảng Tham Chiếu Nhanh Kernel & Công Thức
st.markdown("### 📑 Bảng Tham Chiếu Nhanh: Kernel & Công Thức Toán Học")
c_ref1, c_ref2, c_ref3, c_ref4 = st.columns(4)

with c_ref1:
    st.markdown("""
    <div class="ref-card" style="border-color: #25cbe6;">
        <b style="color: #25cbe6;">1. Mean Filter (3×3)</b>
        <div class="ref-eq">g(x,y) = 1/(mn) ∑ f(s,t)</div>
        <div class="ref-note"><b>Kernel h = 1/9 × [Ma trận 1]</b><br>Chia đều trọng số cho mọi pixel. San phẳng nhiễu nhanh nhưng làm nhòe mờ nghiêm trọng biên cạnh.</div>
    </div>
    """, unsafe_allow_html=True)

with c_ref2:
    st.markdown("""
    <div class="ref-card" style="border-color: #48d39c;">
        <b style="color: #48d39c;">2. Gaussian Filter (3×3)</b>
        <div class="ref-eq">G(x,y) ∝ exp(-(x²+y²)/(2σ²))</div>
        <div class="ref-note"><b>Kernel h ≈ 1/16 × [[1,2,1],[2,4,2],[1,2,1]]</b><br>Trọng số giảm dần theo hàm Gauss từ tâm ra biên. Làm mịn tự nhiên, triệt tiêu nhiễu Gauss tốt.</div>
    </div>
    """, unsafe_allow_html=True)

with c_ref3:
    st.markdown("""
    <div class="ref-card" style="border-color: #ffad3d;">
        <b style="color: #ffad3d;">3. Median Filter</b>
        <div class="ref-eq">g(x,y) = median{W(x,y)}</div>
        <div class="ref-note"><b>Thống kê thứ tự phi tuyến</b><br>Sắp xếp lân cận, chọn phần tử chính giữa. Triệt tiêu 100% hạt xung cực trị (0/255) của nhiễu muối tiêu.</div>
    </div>
    """, unsafe_allow_html=True)

with c_ref4:
    st.markdown("""
    <div class="ref-card" style="border-color: #8e69f2;">
        <b style="color: #8e69f2;">4. Bilateral Filter</b>
        <div class="ref-eq">I_filtered = 1/Wp ∑ G_σs · G_σr · I_q</div>
        <div class="ref-note"><b>Lọc phi tuyến 2 miền</b><br>Kết hợp khoảng cách không gian (σs) & sai lệch mức xám (σr). Bảo toàn nguyên vẹn biên nét khi khử nhiễu.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br
