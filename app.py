import time
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import streamlit as st

st.set_page_config(page_title="Khử Nhiễu Ảnh Số", layout="wide")

st.title("HỆ THỐNG KHỬ NHIỄU ẢNH SỐ")
st.markdown("---")

# Tải ảnh gốc (Ground Truth)
uploaded_file = st.file_uploader(
    "Tải lên ảnh gốc (Ground Truth):", type=["jpg", "png", "jpeg", "bmp"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    clean_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.sidebar.header("Cấu hình mô phỏng nhiễu")
    noise_type = st.sidebar.selectbox(
        "Chọn loại nhiễu:", ["Gaussian Noise", "Salt and Pepper Noise"]
    )

    # Hàm tạo nhiễu Gaussian
    def add_gaussian_noise(image, var=0.01):
        sigma = var**0.5
        gauss = np.random.normal(0, sigma, image.shape) * 255
        noisy = image.astype(np.float32) + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # Hàm tạo nhiễu Salt and Pepper (Vectorized bằng NumPy để chạy mượt mà, không bị lag)
    def add_salt_and_pepper_noise(image, prob=0.05):
        output = image.copy()
        rdn = np.random.random(image.shape)
        output[rdn < (prob / 2)] = 0            # Hạt tiêu (đen)
        output[rdn > (1 - prob / 2)] = 255      # Hạt muối (trắng)
        return output

    # Tạo ảnh nhiễu tương ứng
    if noise_type == "Gaussian Noise":
        # SỬA TẠI ĐÂY: Thêm step=0.001 và format="%.3f" để kéo mượt mà từ 0.001 chạm đúng 0.050
        var = st.sidebar.slider(
            "Độ phương sai (Variance):",
            min_value=0.001,
            max_value=0.050,
            value=0.010,
            step=0.001,
            format="%.3f"
        )
        noisy_img = add_gaussian_noise(clean_img, var=var)
    else:
        prob = st.sidebar.slider(
            "Tỷ lệ nhiễu (Probability):",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            format="%.2f"
        )
        noisy_img = add_salt_and_pepper_noise(clean_img, prob=prob)

    # Gợi ý thông minh tự động
    if noise_type == "Gaussian Noise":
        st.info(
            "Gợi ý tự động: Phát hiện nhiễu Gaussian -> Đề xuất sử dụng **Bilateral Filter** hoặc **Gaussian Filter**."
        )
    else:
        st.info(
            "Gợi ý tự động: Phát hiện nhiễu Salt & Pepper -> Đề xuất sử dụng **Median Filter**."
        )

    # Lựa chọn bộ lọc của các thành viên trong nhóm
    filter_choice = st.selectbox(
        "Lựa chọn bộ lọc khử nhiễu:",
        [
            "Bilateral Filter",
            "Median Filter",
            "Gaussian Filter",
            "Mean Filter",
        ],
    )

    start_time = time.perf_counter()
    if "Bilateral Filter" in filter_choice:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Thông số Bilateral Filter")
        d = st.sidebar.slider("Đường kính lân cận (d):", 5, 15, 9)
        sigma_color = st.sidebar.slider("Sigma Color:", 10, 150, 75)
        sigma_space = st.sidebar.slider("Sigma Space:", 10, 150, 75)
        filtered_img = cv2.bilateralFilter(
            noisy_img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
        )

    elif "Median Filter" in filter_choice:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Thông số Median Filter")
        kernel_median = st.sidebar.slider("Kích thước Kernel Median:", 3, 11, 3, step=2)
        filtered_img = cv2.medianBlur(noisy_img, kernel_median)

    elif "Gaussian Filter" in filter_choice:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Thông số Gaussian Filter")
        kernel_gauss = st.sidebar.slider("Kích thước Kernel Gaussian:", 3, 11, 3, step=2)
        filtered_img = cv2.GaussianBlur(
            noisy_img, (kernel_gauss, kernel_gauss), sigmaX=0, borderType=cv2.BORDER_REFLECT
        )

    elif "Mean Filter" in filter_choice:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Thông số Mean Filter")
        kernel_mean = st.sidebar.slider("Kích thước Kernel Mean:", 3, 11, 3, step=2)
        filtered_img = cv2.blur(noisy_img, (kernel_mean, kernel_mean))

    end_time = time.perf_counter()
    processing_time_ms = (end_time - start_time) * 1000

    # Hiển thị kết quả so sánh hình ảnh trực quan
    st.markdown("### Kết Quả So Sánh Hình Ảnh")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(clean_img, caption="Ảnh Sạch (Ground Truth)", use_container_width=True)
    with col2:
        st.image(
            noisy_img,
            caption=f"Ảnh Nhiễu ({noise_type})",
            use_container_width=True,
        )
    with col3:
        st.image(
            filtered_img,
            caption=f"Ảnh Sau Khi Lọc ({filter_choice})",
            use_container_width=True,
        )

    # Tính toán các chỉ số đánh giá chất lượng (PSNR, SSIM)
    psnr_before = psnr(clean_img, noisy_img, data_range=255)
    ssim_before = ssim(clean_img, noisy_img, data_range=255)

    psnr_after = psnr(clean_img, filtered_img, data_range=255)
    ssim_after = ssim(clean_img, filtered_img, data_range=255)

    st.markdown("### Bảng Đánh Giá Chỉ Số Định Lượng")
    comparison_data = {
        "Trạng thái": ["Trước khi lọc (Ảnh nhiễu)", "Sau khi lọc (Khử nhiễu)"],
        "PSNR (dB) — Càng cao càng tốt": [
            f"{psnr_before:.2f}",
            f"{psnr_after:.2f}",
        ],
        "SSIM — Càng gần 1 càng tốt": [f"{ssim_before:.4f}", f"{ssim_after:.4f}"],
    }
    st.table(pd.DataFrame(comparison_data))
    st.metric(label="Thời gian thực thi thuật toán", value=f"{processing_time_ms:.2f} ms")
