import base64
import os
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
    initial_sidebar_state="collapsed",
)

# Hàm tự động đọc file logo_huit.png trong thư mục đồ án và chuyển sang Base64 để nhúng vào HTML
def get_logo_html():
    possible_files = ["logo_huit.png", "logo.png", "huit.png", "logo_huit.jpg", "logo.jpg"]
    for filename in possible_files:
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            mime = "image/png" if filename.endswith(".png") else "image/jpeg"
            return f'<img src="data:{mime};base64,{b64_data}" class="huit-logo-img" alt="HUIT Logo">'
    
    # Nếu chưa có file ảnh trong thư mục, hiển thị vòng tròn HUIT dự phòng
    return '<div class="huit-logo-fallback">HUIT</div>'

# CSS đồng bộ giao diện tối & Fix lỗi chìm chữ khi trình duyệt ở chế độ Light Mode mặc định
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=Fira+Code:wght@400;600&display=swap');
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #103656 0%, #061426 45%, #04101d 100%) !important;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }

    /* Ép màu chữ sáng cho toàn bộ Tiêu đề, Nhãn (Label), Chú thích ảnh (Caption) bất kể Light/Dark mode */
    h1, h2, h3, h4, h5, h6, p, span, label, 
    .stMarkdown, [data-testid="stWidgetLabel"] p, 
    [data-testid="stCaptionContainer"] p,
    [data-testid="stImageCaption"] {
        color: #f8fafc !important;
    }

    /* Tùy chỉnh khung Upload ảnh rõ nét trên nền tối */
    [data-testid="stFileUploader"] section {
        background-color: #081d33 !important;
        border: 1px dashed #25cbe6 !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
    }
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] section span {
        color: #cbd8e2 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #103656 !important;
        color: #ffffff !important;
        border: 1px solid #25cbe6 !important;
    }

    /* Tùy chỉnh Selectbox không bị chìm chữ ở Light Mode */
    div[data-baseweb="select"] > div {
        background-color: #081d33 !important;
        color: #f8fafc !important;
        border-color: #2a5877 !important;
    }
    div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }

    /* Khắc phục triệt để lỗi chìm chữ trong bảng (Table) ở Light Mode */
    .custom-dark-table {
        width: 100%;
        border-collapse: collapse;
        background: #081d33;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #2a5877;
        margin-bottom: 16px;
    }
    .custom-dark-table th {
        background: #0d2b49 !important;
        color: #5ce4f4 !important;
        font-weight: 700 !important;
        text-align: left;
        padding: 12px 16px;
        border-bottom: 1px solid #2a5877 !important;
        font-size: 14px;
    }
    .custom-dark-table td {
        color: #f8fafc !important;
        padding: 12px 16px;
        border-bottom: 1px solid #163654 !important;
        font-size: 14px;
    }
    .custom-dark-table tr:last-child td {
        border-bottom: none !important;
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
    .huit-logo-img {
        width: 62px;
        height: 62px;
        border-radius: 50%;
        background: #ffffff;
        padding: 4px;
        object-fit: contain;
        border: 2px solid #1cc9df;
        box-shadow: 0 4px 12px rgba(28, 201, 223, 0.25);
        flex-shrink: 0;
    }
    .huit-logo-fallback {
        width: 62px;
        height: 62px;
        border-radius: 50%;
        background: #1cc9df;
        color: #061426 !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 16px;
        flex-shrink: 0;
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
    .theory-box { border-left: 4px solid #25cbe6; background: #051424; border-radius: 0 10px 10px 0; padding: 12px 16px; margin-top: 12px; margin-bottom: 10px; }
    .theory-box b { color: #25cbe6; font-size: 14px; display: block; margin-bottom: 4px; }
    .theory-box p { margin: 0; font-size: 13px; color: #d1dde6 !important; line-height: 1.5; }
    .code-chip { font-family: 'Fira Code', monospace; font-size: 12px; background: #04101d; color: #5ce4f4 !important; border: 1px solid #1c4361; border-radius: 8px; padding: 10px 14px; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# 1. Header Banner HUIT 
logo_element = get_logo_html()
st.markdown(f"""
<div class="huit-header">
    {logo_element}
    <div class="huit-title-box">
        <div class="huit-school">TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP.HCM</div>
        <div class="huit-dept">Khoa Công nghệ thông tin · </div>
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
        Hệ thống tích hợp đầy đủ cơ sở toán học, ma trận kernel tham chiếu 
        và bộ chỉ số định lượng tiêu chuẩn quốc tế (PSNR, SSIM, Processing Time) nhằm đánh giá 
        chính xác tương quan giữa <b>Mean, Gaussian, Median và Bilateral Filter</b> trên các mô hình nhiễu thực tế.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📑 Bảng Tham Chiếu Nhanh: Kernel & Công Thức Toán Học")
c_ref1, c_ref2, c_ref3, c_ref4 = st.columns(4)

with c_ref1:
    with st.container(border=True):
        st.markdown('<b style="color: #25cbe6;">1. Mean Filter (k×k)</b>', unsafe_allow_html=True)
        st.latex(r"g(x,y) = \frac{1}{k^2} \sum_{(s,t)\in S_{xy}} f(s,t)")
        st.latex(r"h = \frac{1}{9}\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}")
        st.markdown('<div class="ref-note">Chia đều trọng số cho mọi pixel trong cửa sổ lân cận. San phẳng nhiễu nhanh nhưng làm nhòe mờ biên cạnh.</div>', unsafe_allow_html=True)

with c_ref2:
    with st.container(border=True):
        st.markdown('<b style="color: #48d39c;">2. Gaussian Filter</b>', unsafe_allow_html=True)
        st.latex(r"G(x,y) = \frac{1}{2\pi\sigma^2} \cdot \exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)")
        st.latex(r"h \approx \frac{1}{16}\begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}")
        st.markdown('<div class="ref-note">Hàm phân phối Gauss liên tục 2 chiều (kỳ vọng bằng 0). Làm mịn tự nhiên, triệt tiêu nhiễu Gauss tốt.</div>', unsafe_allow_html=True)

with c_ref3:
    with st.container(border=True):
        st.markdown('<b style="color: #ffad3d;">3. Median Filter</b>', unsafe_allow_html=True)
        st.latex(r"\hat{f}(x,y) = \underset{(s,t)\in S_{xy}}{\text{median}} \{g(s,t)\}")
        st.latex(r"W_{\text{sorted}} \rightarrow \text{phần tử chính giữa}")
        st.markdown('<div class="ref-note"><b>Bộ lọc trung vị:</b> Sắp xếp các mức xám lân cận và lấy giá trị ở giữa. Triệt tiêu 100% hạt xung cực trị của nhiễu muối tiêu.</div>', unsafe_allow_html=True)

with c_ref4:
    with st.container(border=True):
        st.markdown('<b style="color: #8e69f2;">4. Bilateral Filter</b>', unsafe_allow_html=True)
        st.latex(r"I_{\text{filtered}} = \frac{1}{W_P} \sum_{x_i \in \Omega} I(x_i) f_s(\|x_i-x\|) g_r(\|I(x_i)-I(x)\|)")
        st.latex(r"w = f_s(\cdot) \cdot g_r(\cdot)")
        st.markdown('<div class="ref-note"><b>Lọc phi tuyến 2 miền:</b> Kết hợp khoảng cách không gian ($f_s$) & độ lệch mức xám ($g_r$). Bảo toàn xuất sắc biên nét.</div>', unsafe_allow_html=True)
# 4. Hàm Tạo Ảnh Mẫu & Sinh Nhiễu Chuẩn Xác
def generate_default_sample():
    img = np.zeros((300, 420), dtype=np.uint8)
    for y in range(300):
        img[y, :] = int(40 + (y / 300.0) * 160)
    cv2.rectangle(img, (40, 70), (160, 200), 30, -1)
    cv2.rectangle(img, (260, 60), (380, 210), 220, -1)
    cv2.circle(img, (210, 140), 45, 128, -1)
    cv2.line(img, (40, 240), (380, 240), 255, 3)
    cv2.putText(img, "HUIT DATA SCIENCE", (75, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2, cv2.LINE_AA)
    return img

def add_gaussian_noise(image, var=0.01):
    rng = np.random.default_rng(42)
    sigma = (var ** 0.5) * 255.0
    gauss = rng.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, prob=0.05):
    rng = np.random.default_rng(42)
    output = image.copy()
    rdn = rng.random(image.shape)
    output[rdn < (prob / 2.0)] = 0
    output[rdn > (1.0 - prob / 2.0)] = 255
    return output

# Khởi tạo các giá trị mặc định cho bộ lọc và bảng ánh xạ sigma
d_val, sc_val, ss_val = 9, 75, 75
k_med = 5
k_gauss = 5
k_mean = 5
sigma_mapping = {3: 1.1, 5: 1.4, 7: 1.7, 9: 2.0, 11: 2.3}

# 5. Bố Cục Chính: Cột Trái (Upload + So Sánh Ảnh) & Cột Phải (Cấu Hình Nhiễu + Bộ Lọc + Bản Chất Thuật Toán)
col_visual, col_control = st.columns([1.3, 0.7], gap="large")

with col_visual:
    st.markdown("### 📁 Nguồn Dữ Liệu Ảnh Đầu Vào")
    uploaded_file = st.file_uploader("Tải lên ảnh tùy chọn (Ground Truth):", type=["jpg", "png", "jpeg", "bmp"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        clean_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if clean_img is None:
            st.error("Lỗi đọc file ảnh! Vui lòng chọn ảnh định dạng hợp lệ.")
            clean_img = generate_default_sample()
        else:
            if clean_img.shape[0] > 600 or clean_img.shape[1] > 600:
                clean_img = cv2.resize(clean_img, (420, 300), interpolation=cv2.INTER_AREA)
        img_status = f"Đang dùng ảnh tải lên: {uploaded_file.name}"
    else:
        clean_img = generate_default_sample()
        img_status = "Đang dùng ảnh mẫu mô phỏng đồ họa chuẩn"
    
    st.info(f"📌 **Trạng thái ảnh hiện tại:** {img_status}")

with col_control:
    st.markdown("###  1. Cấu Hình Mô Phỏng Nhiễu")
    noise_type = st.selectbox("Chọn mô hình nhiễu:", ["Gaussian Noise", "Salt & Pepper Noise"])

    if "Gaussian" in noise_type:
        var_val = st.slider("Phương sai nhiễu var:", min_value=0.001, max_value=0.050, value=0.010, step=0.001, format="%.3f")
        _ = add_gaussian_noise(clean_img, var=var_val)
        t_n0 = time.perf_counter()
        noisy_img = add_gaussian_noise(clean_img, var=var_val)
        noise_time_ms = (time.perf_counter() - t_n0) * 1000
    else:
        prob_val = st.slider("Tỷ lệ hạt nhiễu Probability:", min_value=0.01, max_value=0.20, value=0.06, step=0.01, format="%.2f")
        _ = add_salt_and_pepper_noise(clean_img, prob=prob_val)
        t_n0 = time.perf_counter()
        noisy_img = add_salt_and_pepper_noise(clean_img, prob=prob_val)
        noise_time_ms = (time.perf_counter() - t_n0) * 1000

    st.markdown("---")
    st.markdown("###  2. Lựa Chọn Thuật Bộ Lọc")
    filter_choice = st.selectbox("Bộ lọc đang khảo sát chi tiết:", [
        "Bilateral Filter",
        "Median Filter",
        "Gaussian Filter",
        "Mean Filter"
    ])

    if "Bilateral" in filter_choice:
        d_val = st.slider("Đường kính lân cận (d):", 3, 15, 9, step=2)
        sc_val = st.slider("Sigma Color (σ_range):", 10, 150, 75, step=5)
        ss_val = st.slider("Sigma Space (σ_spatial):", 10, 150, 75, step=5)
        _ = cv2.bilateralFilter(noisy_img, d=d_val, sigmaColor=sc_val, sigmaSpace=ss_val)
        start_t = time.perf_counter()
        filtered_img = cv2.bilateralFilter(noisy_img, d=d_val, sigmaColor=sc_val, sigmaSpace=ss_val)
        exec_time_ms = (time.perf_counter() - start_t) * 1000

        st.markdown(f"""
        <div class="theory-box">
            <b>Bản chất thuật toán: Bảo toàn biên cạnh bằng 2 hàm Gauss</b>
            <p>Trọng số kết hợp khoảng cách không gian (σ_s = {ss_val}) và độ tương đồng mức xám (σ_r = {sc_val}) trên vùng lân cận d = {d_val}.</p>
        </div>
        <div class="code-chip">filtered = cv2.bilateralFilter(noisy_img, d={d_val}, sigmaColor={sc_val}, sigmaSpace={ss_val})</div>
        """, unsafe_allow_html=True)

    elif "Median" in filter_choice:
        k_med = st.slider("Kích thước Kernel Median:", 3, 11, 5, step=2)
        _ = cv2.medianBlur(noisy_img, k_med)
        start_t = time.perf_counter()
        filtered_img = cv2.medianBlur(noisy_img, k_med)
        exec_time_ms = (time.perf_counter() - start_t) * 1000

        st.markdown(f"""
        <div class="theory-box">
            <b>Bản chất thuật toán: Cơ chế thống kê thứ tự phi tuyến</b>
            <p>Sắp xếp các điểm ảnh trong cửa sổ {k_med}×{k_med} và chọn phần tử trung vị để triệt tiêu nhiễu xung cực trị.</p>
        </div>
        <div class="code-chip">filtered = cv2.medianBlur(noisy_img, ksize={k_med})</div>
        """, unsafe_allow_html=True)

    elif "Gaussian" in filter_choice:
        k_gauss = st.slider("Kích thước Kernel Gauss:", 3, 11, 5, step=2)
        sigma_val = sigma_mapping.get(k_gauss, 1.4)
        _ = cv2.GaussianBlur(noisy_img, (k_gauss, k_gauss), sigmaX=sigma_val, sigmaY=sigma_val, borderType=cv2.BORDER_REFLECT)
        start_t = time.perf_counter()
        filtered_img = cv2.GaussianBlur(noisy_img, (k_gauss, k_gauss), sigmaX=sigma_val, sigmaY=sigma_val, borderType=cv2.BORDER_REFLECT)
        exec_time_ms = (time.perf_counter() - start_t) * 1000

        st.markdown(f"""
        <div class="theory-box">
            <b>Bản chất thuật toán: Lọc Gauss theo quy chuẩn đồ án</b>
            <p>Kernel {k_gauss}×{k_gauss} tương ứng với độ lệch chuẩn cố định σ = {sigma_val}, làm mịn tự nhiên theo phân phối chuẩn.</p>
        </div>
        <div class="code-chip">filtered = cv2.GaussianBlur(noisy_img, ({k_gauss},{k_gauss}), sigmaX={sigma_val})</div>
        """, unsafe_allow_html=True)

    elif "Mean" in filter_choice:
        k_mean = st.slider("Kích thước Kernel Mean (Box):", 3, 11, 5, step=2)
        _ = cv2.blur(noisy_img, (k_mean, k_mean))
        start_t = time.perf_counter()
        filtered_img = cv2.blur(noisy_img, (k_mean, k_mean))
        exec_time_ms = (time.perf_counter() - start_t) * 1000

        st.markdown(f"""
        <div class="theory-box">
            <b>Bản chất thuật toán: Nguyên lý trung bình khối (Box Filter)</b>
            <p>Chia đều trọng số 1/{k_mean*k_mean} cho mọi pixel trong cửa sổ {k_mean}×{k_mean}.</p>
        </div>
        <div class="code-chip">filtered = cv2.blur(noisy_img, ({k_mean}, {k_mean}))</div>
        """, unsafe_allow_html=True)

# Hiển thị 3 ảnh so sánh nằm ngay dưới phần Upload ảnh ở cột bên trái
with col_visual:
    st.markdown("### 🖼️ Kết Quả So Sánh Hình Ảnh Song Song")
    c_img1, c_img2, c_img3 = st.columns(3)
    with c_img1:
        st.image(clean_img, caption="1. Ảnh Gốc Chuẩn (Ground Truth)", use_container_width=True)
    with c_img2:
        st.image(noisy_img, caption=f"2. Ảnh Nhiễm Nhiễu ({noise_type.split(' ')[0]})", use_container_width=True)
    with c_img3:
        st.image(filtered_img, caption=f"3. Sau Khi Lọc ({filter_choice.split(' ')[0]})", use_container_width=True)

# 6. Tính Định Lượng PSNR & SSIM và Hiển Thị Bảng Đánh Giá (Đầy đủ 2 ô Thời gian thực thi)
psnr_raw = psnr(clean_img, noisy_img, data_range=255)
ssim_raw = ssim(clean_img, noisy_img, data_range=255)

psnr_filtered = psnr(clean_img, filtered_img, data_range=255)
ssim_filtered = ssim(clean_img, filtered_img, data_range=255)

st.markdown("---")
st.markdown("### 📊 Bảng Đánh Giá Định Lượng & Hiệu Năng Bộ Lọc Hiện Tại")

eval_html = (
    '<table class="custom-dark-table">'
    '<thead><tr>'
    '<th>Trạng thái thực nghiệm</th>'
    '<th>PSNR (dB) — [Càng cao càng tốt]</th>'
    '<th>SSIM — [Càng gần 1 càng tốt]</th>'
    '<th>Thời gian thực thi</th>'
    '</tr></thead>'
    '<tbody>'
    f'<tr><td>Ảnh Nhiễm Nhiễu Đầu Vào</td><td>{psnr_raw:.2f} dB</td><td>{ssim_raw:.4f}</td><td>{noise_time_ms:.2f} ms</td></tr>'
    f'<tr><td>Sau Khi Khử Nhiễu ({filter_choice.split(" ")[0]})</td><td>{psnr_filtered:.2f} dB</td><td>{ssim_filtered:.4f}</td><td>{exec_time_ms:.2f} ms</td></tr>'
    '</tbody></table>'
)
st.markdown(eval_html, unsafe_allow_html=True)

# 7. CASE STUDY TOÀN DIỆN 4 BỘ LỌC (SO SÁNH ĐỒNG BỘ 4 BỘ LỌC)
st.markdown("---")
st.markdown("### 🔬 Case Study Đề Tài: So Sánh Đối Chứng Toàn Diện 4 Bộ Lọc Không Gian")
st.caption("Thực nghiệm chạy song song đồng thời 4 bộ lọc trên cùng ảnh nhiễu hiện tại:")

sigma_case = sigma_mapping.get(k_gauss, 1.4)

t_m0 = time.perf_counter()
res_mean = cv2.blur(noisy_img, (k_mean, k_mean))
t_m = (time.perf_counter() - t_m0) * 1000

t_g0 = time.perf_counter()
res_gauss = cv2.GaussianBlur(noisy_img, (k_gauss, k_gauss), sigmaX=sigma_case, sigmaY=sigma_case, borderType=cv2.BORDER_REFLECT)
t_g = (time.perf_counter() - t_g0) * 1000

t_med0 = time.perf_counter()
res_median = cv2.medianBlur(noisy_img, k_med)
t_med = (time.perf_counter() - t_med0) * 1000

t_b0 = time.perf_counter()
res_bilateral = cv2.bilateralFilter(noisy_img, d=d_val, sigmaColor=sc_val, sigmaSpace=ss_val)
t_b = (time.perf_counter() - t_b0) * 1000

comp_results = [
    {
        "Bộ lọc": f"Mean Filter ({k_mean}×{k_mean})",
        "PSNR (dB)": psnr(clean_img, res_mean, data_range=255),
        "SSIM": ssim(clean_img, res_mean, data_range=255),
        "Thời gian (ms)": t_m,
        "Nhận xét kỹ thuật": "Làm mịn nhanh nhưng nhòe mờ biên cạnh nghiêm trọng."
    },
    {
        "Bộ lọc": f"Gaussian Filter ({k_gauss}×{k_gauss}, σ={sigma_case})",
        "PSNR (dB)": psnr(clean_img, res_gauss, data_range=255),
        "SSIM": ssim(clean_img, res_gauss, data_range=255),
        "Thời gian (ms)": t_g,
        "Nhận xét kỹ thuật": "Làm mịn tự nhiên hơn Mean, khử nhiễu Gauss tốt."
    },
    {
        "Bộ lọc": f"Median Filter ({k_med}×{k_med})",
        "PSNR (dB)": psnr(clean_img, res_median, data_range=255),
        "SSIM": ssim(clean_img, res_median, data_range=255),
        "Thời gian (ms)": t_med,
        "Nhận xét kỹ thuật": "Vượt trội tuyệt đối trên nhiễu muối tiêu, giữ biên khá."
    },
    {
        "Bộ lọc": f"Bilateral Filter (d={d_val})",
        "PSNR (dB)": psnr(clean_img, res_bilateral, data_range=255),
        "SSIM": ssim(clean_img, res_bilateral, data_range=255),
        "Thời gian (ms)": t_b,
        "Nhận xét kỹ thuật": "Tối ưu nhất cho nhiễu Gauss, bảo toàn xuất sắc biên nét."
    }
]

c_show1, c_show2, c_show3, c_show4 = st.columns(4)
with c_show1:
    st.image(res_mean, caption=f"Mean ({k_mean}x{k_mean})", use_container_width=True)
with c_show2:
    st.image(res_gauss, caption=f"Gaussian ({k_gauss}x{k_gauss})", use_container_width=True)
with c_show3:
    st.image(res_median, caption=f"Median ({k_med}x{k_med})", use_container_width=True)
with c_show4:
    st.image(res_bilateral, caption=f"Bilateral (d={d_val})", use_container_width=True)

# Render bảng so sánh 4 bộ lọc bằng chuỗi HTML liền mạch (không thụt đầu dòng) để tránh lỗi hiện thẻ <tr><td>
rows_html = "".join([
    f"<tr><td>{r['Bộ lọc']}</td><td>{r['PSNR (dB)']:.2f}</td><td>{r['SSIM']:.4f}</td><td>{r['Thời gian (ms)']:.2f} ms</td><td>{r['Nhận xét kỹ thuật']}</td></tr>"
    for r in comp_results
])

comp_table_html = (
    '<table class="custom-dark-table">'
    '<thead><tr>'
    '<th>Bộ lọc</th>'
    '<th>PSNR (dB)</th>'
    '<th>SSIM</th>'
    '<th>Thời gian (ms)</th>'
    '<th>Nhận xét kỹ thuật</th>'
    '</tr></thead>'
    f'<tbody>{rows_html}</tbody>'
    '</table>'
)
st.markdown(comp_table_html, unsafe_allow_html=True)

fig_bar, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2), facecolor="#07192d")
filter_names = ["Mean", "Gaussian", "Median", "Bilateral"]
colors = ["#25cbe6", "#48d39c", "#ffad3d", "#8e69f2"]
psnr_scores = [float(r["PSNR (dB)"]) for r in comp_results]
ssim_scores = [float(r["SSIM"]) for r in comp_results]

ax1.set_facecolor("#07192d")
bars1 = ax1.bar(filter_names, psnr_scores, color=colors, width=0.55)
ax1.set_title("So Sánh Chỉ Số PSNR (dB)", color="#cbd8e1", fontsize=11, fontweight="bold")
ax1.tick_params(colors="#8ba2b5", labelsize=9)
ax1.set_ylim(0, max(psnr_scores) * 1.25)
ax1.grid(axis="y", color="#1e3d59", linestyle=":", alpha=0.7)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{yval:.1f}", ha='center', va='bottom', color='#ffffff', fontsize=9, fontweight='bold')

ax2.set_facecolor("#07192d")
bars2 = ax2.bar(filter_names, ssim_scores, color=colors, width=0.55)
ax2.set_title("So Sánh Chỉ Số SSIM", color="#cbd8e1", fontsize=11, fontweight="bold")
ax2.tick_params(colors="#8ba2b5", labelsize=9)
ax2.set_ylim(0, 1.15)
ax2.grid(axis="y", color="#1e3d59", linestyle=":", alpha=0.7)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{yval:.3f}", ha='center', va='bottom', color='#ffffff', fontsize=9, fontweight='bold')

st.pyplot(fig_bar)
plt.close(fig_bar)

if "Gaussian" in noise_type:
    st.success("🎯 **Kết luận :** Trên mô hình **Nhiễu Gauss**, **Bilateral Filter** và **Gaussian Filter** đạt hiệu quả cao nhất.")
else:
    st.success("🎯 **Kết luận :** Trên mô hình **Nhiễu Muối Tiêu (Salt & Pepper)**, **Median Filter** là bộ lọc tối ưu và vượt trội hoàn toàn.")

st.markdown("""
<div style="margin-top: 45px; background: #07192c; border-top: 3px solid #1cc9df; padding: 16px; text-align: center; color: #b8cbd9; font-size: 13px; border-radius: 10px;">
    <b>TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP.HCM (HUIT)</b> · Khoa Công Nghệ Thông Tin ·
    Đồ án môn học: Xử Lý Ảnh Số — Giảng viên hướng dẫn: TS. Nguyễn Huy Liêm
</div>
""", unsafe_allow_html=True)
