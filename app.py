import time
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import streamlit as st

st.set_page_config(
    page_title="Khử Nhiễu Ảnh Số", layout="wide"
)

st.title("HỆ THỐNG KHỬ NHIỄU ẢNH SỐ")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Tải lên ảnh gốc (Ground Truth):", type=["jpg", "png", "jpeg", "bmp"]
)

if uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  clean_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

  st.sidebar.header("Cấu hình mô phỏng")
  noise_type = st.sidebar.selectbox(
      "Chọn loại nhiễu:", ["Gaussian Noise", "Salt and Pepper Noise"]
  )


  def add_gaussian_noise(image, var=0.01):
    sigma = var**0.5
    gauss = np.random.normal(0, sigma, image.shape) * 255
    noisy = image.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


  def add_salt_and_pepper_noise(image, prob=0.05):
    output = image.copy()
    thres = 1 - prob
    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        rdn = np.random.random()
        if rdn < prob:
          output[i, j] = 0
        elif rdn > thres:
          output[i, j] = 255
    return output


  if noise_type == "Gaussian Noise":
    var = st.sidebar.slider("Độ phương sai (Variance):", 0.001, 0.05, 0.01)
    noisy_img = add_gaussian_noise(clean_img, var=var)
  else:
    prob = st.sidebar.slider("Tỷ lệ nhiễu (Probability):", 0.01, 0.2, 0.05)
    noisy_img = add_salt_and_pepper_noise(clean_img, prob=prob)

  # Gợi ý bộ lọc
  if noise_type == "Gaussian Noise":
    st.info(
        "Gợi ý tự động: Phát hiện nhiễu Gaussian -> Đề xuất sử dụng **Bilateral"
        " Filter**."
    )
  else:
    st.info(
        "Gợi ý tự động: Phát hiện nhiễu Salt & Pepper -> Đề xuất sử dụng **Median"
        " Filter**."
    )

  # Lựa chọn bộ lọc 
  filter_choice = st.selectbox(
      "Lựa chọn bộ lọc:",
      [
          "Bilateral Filter (TV5 - Của bạn)",
          "Median Filter (TV4 - Đã tích hợp)",
          "Mean Filter (Chờ code TV2)",
          "Gaussian Filter (Chờ code TV3)",
      ],
  )

  start_time = time.time()

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
    #TV4
    kernel_median = st.sidebar.slider(
        "Kích thước Kernel Median:", 3, 9, 3, step=2
    )
    filtered_img = cv2.medianBlur(noisy_img, kernel_median)

  elif "Mean Filter" in filter_choice:
    filtered_img = cv2.blur(noisy_img, (3, 3))
  else:
    filtered_img = cv2.GaussianBlur(noisy_img, (3, 3), 0)

  end_time = time.time()
  processing_time_ms = (end_time - start_time) * 1000

  # Hiển thị kết quả trực quan
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

  # Tính toán chỉ số trước và sau khi lọc
  psnr_before = psnr(clean_img, noisy_img, data_range=255)
  ssim_before = ssim(clean_img, noisy_img, data_range=255)

  psnr_after = psnr(clean_img, filtered_img, data_range=255)
  ssim_after = ssim(clean_img, filtered_img, data_range=255)

  st.markdown("### Bảng So Sánh Chỉ Số (Trước và Sau Khi Lọc)")

  import pandas as pd

  comparison_data = {
      "Trạng thái": ["Trước khi lọc (Ảnh nhiễu)", "Sau khi lọc (Ảnh khử nhiễu)"],
      "PSNR (dB)": [f"{psnr_before:.2f}", f"{psnr_after:.2f}"],
      "SSIM": [f"{ssim_before:.4f}", f"{ssim_after:.4f}"],
  }
  st.table(pd.DataFrame(comparison_data))
  st.metric(label="Thời gian thực thi thuật toán", value=f"{processing_time_ms:.2f} ms")
