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

# Custom CSS phong cách học thuật HUIT Dark Mode
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=Fira+Code:wght@400;600&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 15% 0%, #103656 0%, #061426 45%, #04101d 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Container */
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
    .huit-title-box {
        flex-grow: 1;
    }
    .huit-school {
        font-size: 19px;
        font-weight: 900;
        letter-spacing: 0.5px;
        color: #f8fafc;
        margin-bottom: 3px;
    }
    .huit-dept {
        font-size: 13px;
        color: #b8cbd9;
        margin-bottom: 3px;
    }
    .huit-course {
        font-size: 13px;
        color: #5ce4f4;
        font-weight: 700;
        letter-spacing: 0.8px;
    }
    
    /* Hero Title */
    .hero-box {
        background: linear-gradient(145deg, rgba(14, 45, 75, 0.7), rgba(7, 27, 48, 0.9));
        border: 1px solid #285d82;
        border-radius: 18px;
        padding: 22px 26px;
        margin-bottom: 25px;
    }
    .hero-badge {
        color: #5ce4f4;
        font-weight: 900;
        font-size: 13px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-h1 {
        font-size: 30px;
        font-weight: 900;
        color: #ffffff;
        margin: 0 0 10px 0;
    }
    .hero-p {
        font-size: 14.5px;
        color: #cbd8e2;
        line-height: 1.6;
        margin: 0;
    }

    /* Reference Cards Grid */
    .ref-card {
        background: #081d33;
        border: 1px solid #2a5877;
        border-radius: 14px;
        padding: 14px 16px;
        height: 100%;
        box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }
    .ref-card b {
        font-size: 15px;
        display: block;
        margin-bottom: 6px;
    }
    .ref-eq {
        font-family: 'Fira Code', monospace;
        font-size: 12px;
        background: #051424;
        border-radius: 6px;
        padding: 6px 8px;
        margin-bottom: 8px;
        color: #5ce4f4;
        border: 1px solid #1c3d59;
    }
    .ref-note {
        font-size: 12px;
        color: #a2b7c7;
        line-height: 1.45;
    }

    /* Theoretical Explain Box */
    .theory-box {
        border-left: 4px solid #25cbe6;
        background: #081d33;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .theory-box b {
        color: #25cbe6;
        font-size: 14px;
        display: block;
        margin-bottom: 4px;
    }
    .theory-box p {
        margin: 0;
        font-size: 13px;
        color: #d1dde6;
        line-height: 1.5;
    }

    /* Code Block Styling */
    .code-chip {
        font-family: 'Fira Code', monospace;
        font-size: 12px;
        background: #04101d;
        color: #5ce4f4;
        border: 1px solid #1c4361;
        border-radius: 8px;
        padding: 10px 14px;
        white-space: pre-wrap;
        margin-top: 10px;
    }

    .conclusion-tag {
        background: rgba(37, 203, 230, 0.12);
        border: 1px solid rgba(37, 203, 230, 0.4);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        font-weight: 700;
        color: #e2f4f8;
        margin-top: 10px;
        line-height: 1.5;
    }
</style>
""",
    unsafe_allow_html=True,
)

# 1. Header Banner HUIT
st.markdown(
    """
<div class="huit-header">
    <div style="width: 52px; height: 52px; border-radius: 50%; background: #ffffff; display: flex; align-items: center; justify-content: center; font-weight: 900; color: #0b2743; font-size: 18px; border: 2px solid #5ce4f4;">
        HUIT
    </div>
    <div class="huit-title-box">
        <div class="huit-school">TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP.HCM</div>
        <div class="huit-dept">Khoa Công nghệ thông tin · Bộ môn Khoa học dữ liệu và AI</div>
        <div class="huit-course">XỬ LÝ ẢNH SỐ · ĐỒ ÁN MINH HỌA CÁC BỘ LỌC MIỀN KHÔNG GIAN</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# 2. Hero Presentation Box
st.markdown(
    """
<div class="hero-box">
    <div class="hero-badge">BÁO CÁO THỰC NGHIỆM ĐỒ ÁN MÔN HỌC · WEB TƯƠNG TÁC THỜI GIAN THỰC</div>
    <div class="hero-h1">HỆ THỐNG KHẢO SÁT & ĐỐI CHỨNG 4 BỘ LỌC KHÔNG GIAN</div>
    <p class="hero-p">
        Hệ thống tích hợp đầy đủ cơ sở toán học, ma trận kernel tham chiếu, đồ thị mặt phẳng cắt lát phân tích mức xám 
       và bộ chỉ số định lượng tiêu chuẩn quốc tế (PSNR, SSIM, Processing Time) nhằm đánh giá 
        chính xác tương quan giữa <b>Mean, Gaussian, Median và Bilateral Filter</b> trên các mô hình nhiễu thực tế.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# 3. Bảng Tham Chiếu Nhanh Kernel & Công Thức
st.markdown("### 📑 Bảng Tham Chiếu Nhanh: Kernel & Công Thức Toán Học")
c_ref1, c_ref2, c_ref3, c_ref4 = st.columns(4)

with c_ref1:
  st.markdown(
      """
    <div class="ref-card" style="border-color: #25cbe6;">
        <b style="color: #25cbe6;">1. Mean Filter (3×3)</b>
        <div class="ref-eq">g(x,y) = 1/(mn) ∑ f(s,t)</div>
        <div class="ref-note">
            <b>Kernel h = 1/9 × [Ma trận 1]</b><br>
            Chia đều trọng số cho mọi pixel. San phẳng nhiễu nhanh nhưng làm nhòe mờ nghiêm trọng biên cạnh.
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

with c_ref2:
  st.markdown(
      """
    <div class="ref-card" style="border-color: #48d39c;">
        <b style="color: #48d39c;">2. Gaussian Filter (3×3)</b>
        <div class="ref-eq">G(x,y) ∝ exp(-(x²+y²)/(2σ²))</div>
        <div class="ref-note">
            <b>Kernel h ≈ 1/16 × [[1,2,1],[2,4,2],[1,2,1]]</b><br>
            Trọng số giảm dần theo hàm Gauss từ tâm ra biên. Làm mịn tự nhiên, triệt tiêu nhiễu Gauss tốt.
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

with c_ref3:
  st.markdown(
      """
    <div class="ref-card" style="border-color: #ffad3d;">
        <b style="color: #ffad3d;">3. Median Filter</b>
        <div class="ref-eq">g(x,y) = median{W(x,y)}</div>
        <div class="ref-note">
            <b>Thống kê thứ tự phi tuyến</b><br>
            Sắp xếp lân cận, chọn phần tử chính giữa. Triệt tiêu 100% hạt xung cực trị (0/255) của nhiễu muối tiêu.
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

with c_ref4:
  st.markdown(
      """
    <div class="ref-card" style="border-color: #8e69f2;">
        <b style="color: #8e69f2;">4. Bilateral Filter</b>
        <div class="ref-eq">I_filtered = 1/Wp ∑ G_σs · G_σr · I_q</div>
        <div class="ref-note">
            <b>Lọc phi tuyến 2 miền</b><br>
            Kết hợp khoảng cách không gian (σs) & sai lệch mức xám (σr). Bảo toàn nguyên vẹn biên nét khi khử nhiễu.
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

st.markdown("<br>", unsafe_allow_html=True)

# 4. Khu Vực Tải Ảnh & Chọn Ảnh Mẫu Mặc Định
st.sidebar.markdown("### 📁 Nguồn Dữ Liệu Ảnh")
uploaded_file = st.sidebar.file_uploader(
    "Tải lên ảnh tùy chọn (Ground Truth):", type=["jpg", "png", "jpeg", "bmp"]
)


def generate_default_sample():
  img = np.zeros((300, 420), dtype=np.uint8)
  for y in range(300):
    img[y, :] = int(40 + (y / 300.0) * 160)
  cv2.rectangle(img, (40, 70), (160, 200), 30, -1)
  cv2.rectangle(img, (260, 60), (380, 210), 220, -1)
  cv2.circle(img, (210, 140), 45, 128, -1)
  cv2.line(img, (40, 240), (380, 240), 255, 3)
  cv2.putText(
      img,
      "HUIT DATA SCIENCE",
      (75, 148),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      (255),
      2,
      cv2.LINE_AA,
  )
  return img


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
  img_status = (
      "Đang dùng: Ảnh mẫu mô phỏng đồ họa chuẩn HUIT (Geometric & Text Edges)"
  )

st.info(f"📌 **Trạng thái ảnh hiện tại:** {img_status}")


# 5. Hàm Sinh Nhiễu Chuẩn Xác
def add_gaussian_noise(image, var=0.01):
  sigma = (var**0.5) * 255.0
  gauss = np.random.normal(0, sigma, image.shape)
  noisy = image.astype(np.float32) + gauss
  return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(image, prob=0.05):
  output = image.copy()
  rdn = np.random.random(image.shape)
  output[rdn < (prob / 2.0)] = 0
  output[rdn > (1.0 - prob / 2.0)] = 255
  return output


# 6. Sidebar Cấu Hình Mô Phỏng Nhiễu
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ 1. Cấu Hình Mô Phỏng Nhiễu")
noise_type = st.sidebar.selectbox(
    "Chọn mô hình nhiễu:",
    [
        "Gaussian Noise (Nhiễu Gauss)",
        "Salt & Pepper Noise (Nhiễu Muối Tiêu)",
    ],
)

if "Gaussian" in noise_type:
  var_val = st.sidebar.slider(
      "Phương sai nhiễu (Variance - var):",
      min_value=0.001,
      max_value=0.050,
      value=0.010,
      step=0.001,
      format="%.3f",
      help=(
          "Mức 0.01 là mốc chuẩn thực nghiệm (sigma = 25.5 mức xám). Mức 0.05"
          " là mức nhiễu nặng."
      ),
  )
  noisy_img = add_gaussian_noise(clean_img, var=var_val)
  st.sidebar.caption(
      f"Độ lệch chuẩn pixel tương đương: σ ≈ {((var_val**0.5)*255):.1f} mức"
      " xám"
  )
else:
  prob_val = st.sidebar.slider(
      "Tỷ lệ hạt nhiễu (Probability):",
      min_value=0.01,
      max_value=0.20,
      value=0.06,
      step=0.01,
      format="%.2f",
      help=(
          "Tỷ lệ phần trăm tổng số điểm ảnh bị gán giá trị xung cực trị 0 hoặc"
          " 255."
      ),
  )
  noisy_img = add_salt_and_pepper_noise(clean_img, prob=prob_val)
  st.sidebar.caption(f" Mật độ hạt xung: {int(prob_val*100)}% toàn bộ ảnh")

# 7. Sidebar Lựa Chọn Bộ Lọc Khử Nhiễu
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ 2. Lựa Chọn Thuật Toán Lọc")
filter_choice = st.sidebar.selectbox(
    "Bộ lọc đang khảo sát chi tiết:",
    [
        "Bilateral Filter (Lọc Song Phương)",
        "Median Filter (Lọc Trung Vị)",
        "Gaussian Filter (Lọc Gauss)",
        "Mean Filter (Lọc Trung Bình)",
    ],
)

if "Bilateral" in filter_choice:
  st.sidebar.markdown("**Tham số Bilateral Filter:**")
  d_val = st.sidebar.slider("Đường kính lân cận (d):", 3, 15, 9, step=2)
  sc_val = st.sidebar.slider("Sigma Color (σ_range):", 10, 150, 75, step=5)
  ss_val = st.sidebar.slider("Sigma Space (σ_spatial):", 10, 150, 75, step=5)
  start_t = time.perf_counter()
  filtered_img = cv2.bilateralFilter(
      noisy_img, d=d_val, sigmaColor=sc_val, sigmaSpace=ss_val
  )
  exec_time_ms = (time.perf_counter() - start_t) * 1000

elif "Median" in filter_choice:
  st.sidebar.markdown("**Tham số Median Filter:**")
  k_med = st.sidebar.slider("Kích thước Kernel Median:", 3, 11, 5, step=2)
  start_t = time.perf_counter()
  filtered_img = cv2.medianBlur(noisy_img, k_med)
  exec_time_ms = (time.perf_counter() - start_t) * 1000

elif "Gaussian" in filter_choice:
  st.sidebar.markdown("**Tham số Gaussian Filter:**")
  k_gauss = st.sidebar.slider("Kích thước Kernel Gauss:", 3, 11, 5, step=2)
  sigma_g = st.sidebar.slider("Độ lệch chuẩn Sigma (σ):", 0.5, 4.0, 1.2, step=0.1)
  start_t = time.perf_counter()
  filtered_img = cv2.GaussianBlur(
      noisy_img, (k_gauss, k_gauss), sigmaX=sigma_g, borderType=cv2.BORDER_REFLECT
  )
  exec_time_ms = (time.perf_counter() - start_t) * 1000

elif "Mean" in filter_choice:
  st.sidebar.markdown("**Tham số Mean Filter:**")
  k_mean = st.sidebar.slider("Kích thước Kernel Mean (Box):", 3, 11, 5, step=2)
  start_t = time.perf_counter()
  filtered_img = cv2.blur(noisy_img, (k_mean, k_mean))
  exec_time_ms = (time.perf_counter() - start_t) * 1000

# 8. Tính Định Lượng PSNR & SSIM
psnr_raw = psnr(clean_img, noisy_img, data_range=255)
ssim_raw = ssim(clean_img, noisy_img, data_range=255)

psnr_filtered = psnr(clean_img, filtered_img, data_range=255)
ssim_filtered = ssim(clean_img, filtered_img, data_range=255)

col_visual, col_theory = st.columns([1.3, 0.7])

with col_visual:
  st.markdown("###  Kết Quả So Sánh Hình Ảnh Song Song")
  c_img1, c_img2, c_img3 = st.columns(3)
  with c_img1:
    st.image(
        clean_img,
        caption="1. Ảnh Gốc Chuẩn (Ground Truth)",
        use_container_width=True,
    )
  with c_img2:
    st.image(
        noisy_img,
        caption=f"2. Ảnh Nhiễm Nhiễu ({noise_type.split(' ')[0]})",
        use_container_width=True,
    )
  with c_img3:
    st.image(
        filtered_img,
        caption=f"3. Sau Khi Lọc ({filter_choice.split(' ')[0]})",
        use_container_width=True,
    )

  st.markdown(
      "####  Phân Tích Lát Cắt Mức Xám "
  )
  st.caption(
      "Đường cắt ngang tại hàng giữa ảnh đi xuyên qua các cạnh nét để kiểm tra"
      " hiện tượng làm mờ biên vs bảo toàn biên:"
  )

  slice_y = int(clean_img.shape[0] * 0.5)
  profile_clean = clean_img[slice_y, :]
  profile_noisy = noisy_img[slice_y, :]
  profile_filtered = filtered_img[slice_y, :]

  fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#07192d")
  ax.set_facecolor("#07192d")
  ax.plot(
      profile_noisy,
      label="Ảnh Nhiễu",
      color="#64748b",
      linewidth=1.0,
      alpha=0.65,
  )
  ax.plot(
      profile_clean,
      label="Ảnh Gốc (Chuẩn)",
      color="#ffffff",
      linewidth=1.6,
      linestyle="--",
  )
  ax.plot(
      profile_filtered,
      label=f"Sau {filter_choice.split(' ')[0]}",
      color="#25cbe6",
      linewidth=2.0,
  )

  ax.set_xlim(0, clean_img.shape[1] - 1)
  ax.set_ylim(0, 260)
  ax.set_xlabel("Tọa độ trục ngang X (pixels)", color="#cbd8e1", fontsize=10)
  ax.set_ylabel("Giá trị mức xám (0 - 255)", color="#cbd8e1", fontsize=10)
  ax.tick_params(colors="#8ba2b5", labelsize=9)
  ax.grid(True, color="#1e3d59", linestyle=":", alpha=0.7)
  legend = ax.legend(
      loc="upper right",
      facecolor="#0b2743",
      edgecolor="#2c5f81",
      fontsize=9,
  )
  for text in legend.get_texts():
    text.set_color("#ffffff")
  st.pyplot(fig)

with col_theory:
  st.markdown("### Bản Chất Thuật Toán")

  if "Bilateral" in filter_choice:
    st.markdown(
        """
        <div class="theory-box">
            <b>Bảo toàn biên cạnh bằng 2 hàm Gauss</b>
            <p>Trọng số được tính bằng tích của hàm khoảng cách hình học <i>(Spatial)</i> và hàm độ tương đồng mức xám <i>(Range)</i>. Khi gặp vách biên, độ chênh lệch mức xám rất lớn khiến trọng số triệt tiêu về gần 0, ngăn không cho làm mờ qua biên.</p>
        </div>
        <div class="theory-box">
            <b>Ảnh hưởng của tham số</b>
            <p>• <b>d</b>: Bán kính vùng lân cận.<br>
            • <b>σ_color</b>: Ngưỡng chênh lệch màu để quyết định biên.<br>
            • <b>σ_space</b>: Bán kính không gian ảnh hưởng.</p>
        </div>
        <div class="code-chip"># OpenCV Python Implementation
filtered = cv2.bilateralFilter(noisy_img, d=9, sigmaColor=75, sigmaSpace=75)</div>
        <div class="conclusion-tag">
            Bilateral Filter: <b>Làm phẳng mịn nhiễu nền nhưng giữ độ dốc biên dựng đứng!</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

  elif "Median" in filter_choice:
    st.markdown(
        """
        <div class="theory-box">
            <b>Cơ chế hoạt động của Median</b>
            <p>Không dùng phép tính trung bình cộng mà sắp xếp toàn bộ mức xám trong cửa sổ lân cận theo thứ tự tăng dần, sau đó chọn giá trị chính giữa (Median).</p>
        </div>
        <div class="theory-box">
            <b>Vì sao diệt sạch nhiễu muối tiêu?</b>
            <p>Các hạt nhiễu xung (0 hoặc 255) là các giá trị cực trị. Khi sắp xếp theo thứ tự, chúng luôn bị đẩy dồn về 2 đầu mút và không bao giờ được chọn làm phần tử trung vị.</p>
        </div>
        <div class="code-chip"># OpenCV Python Implementation
filtered = cv2.medianBlur(noisy_img, ksize=5)</div>
        <div class="conclusion-tag">
            Median Filter: <b>Khử nhiễu xung bằng quan hệ thứ tự, không pha trộn dữ liệu.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

  elif "Gaussian" in filter_choice:
    st.markdown(
        """
        <div class="theory-box">
            <b>Khác biệt so với Lọc Mean</b>
            <p>Điểm ảnh càng gần tâm ma trận lọc có trọng số đóng góp càng lớn theo hình chuông Gauss. Ảnh làm mịn tự nhiên hơn, ít bị viền hộp mờ hơn Mean Filter.</p>
        </div>
        <div class="theory-box">
            <b>Vai trò của Sigma (σ)</b>
            <p>σ điều khiển bề rộng của phân phối chuông. σ càng lớn thì phạm vi làm mờ càng rộng, triệt nhiễu mạnh nhưng cũng làm biên cạnh suy giảm độ nét.</p>
        </div>
        <div class="code-chip"># OpenCV Python Implementation
filtered = cv2.GaussianBlur(noisy_img, (5, 5), sigmaX=1.2)</div>
        <div class="conclusion-tag">
            Gaussian Filter: <b>Láng giềng gần có quyền biểu quyết lớn hơn láng giềng xa.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

  elif "Mean" in filter_choice:
    st.markdown(
        """
        <div class="theory-box">
            <b>Nguyên lý trung bình khối (Box Filter)</b>
            <p>Mỗi điểm ảnh đầu ra được thay thế bằng trung bình cộng thuần túy của toàn bộ các điểm ảnh nằm trong mặt nạ kích thước m×n.</p>
        </div>
        <div class="theory-box">
            <b>Hiện tượng mờ nhòe biên</b>
            <p>Do tất cả các điểm ảnh đều có trọng số như nhau (1 / mn), thông tin biên cạnh bị trộn đều cùng với các pixel nền khiến toàn bộ ảnh bị mờ đục.</p>
        </div>
        <div class="code-chip"># OpenCV Python Implementation
filtered = cv2.blur(noisy_img, (5, 5))</div>
        <div class="conclusion-tag">
            Mean Filter: <b>Chia đều quyền biểu quyết cho mọi điểm ảnh trong cửa sổ.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 9. Bảng Đánh Giá Định Lượng
st.markdown("---")
st.markdown("### Bảng Đánh Giá Định Lượng & Hiệu Năng Bộ Lọc Hiện Tại")

c_tab, c_met = st.columns([1.2, 0.8])

with c_tab:
  eval_df = pd.DataFrame({
      "Trạng thái thực nghiệm": [
          "Ảnh Nhiễm Nhiễu Đầu Vào",
          f"Sau Khi Khử Nhiễu ({filter_choice.split(' ')[0]})",
      ],
      "PSNR (dB) — [Càng cao càng tốt]": [
          f"{psnr_raw:.2f} dB",
          f"{psnr_filtered:.2f} dB",
      ],
      "SSIM — [Càng gần 1 càng tốt]": [
          f"{ssim_raw:.4f}",
          f"{ssim_filtered:.4f}",
      ],
      "Mức chênh lệch cải thiện": [
          "-",
          f"Δ PSNR: {psnr_filtered - psnr_raw:+.2f} dB | Δ SSIM:"
          f" {ssim_filtered - ssim_raw:+.4f}",
      ],
  })
  st.table(eval_df)

with c_met:
  st.metric(
      label="Thời Gian Thực Thi Thuật Toán (Latency)",
      value=f"{exec_time_ms:.2f} ms",
  )
  if exec_time_ms < 2.0:
    st.success("⚡ Tốc độ cực nhanh (Tích chập tuyến tính hoặc mảng phân tách)")
  elif exec_time_ms < 10.0:
    st.info("Đạt chuẩn xử lý thời gian thực (< 10 ms trên ảnh chuẩn)")
  else:
    st.warning("Đòi hỏi chi phí tính toán cao trên không gian phi tuyến")

# 10. CASE STUDY TOÀN DIỆN 4 BỘ LỌC
st.markdown("---")
st.markdown(
    "### 🔬 Case Study Đề Tài: So Sánh Đối Chứng Toàn Diện 4 Bộ Lọc Không Gian"
)
st.caption(
    "Thực nghiệm chạy song song đồng thời 4 bộ lọc trên cùng một ảnh nhiễu hiện"
    " tại để đối chiếu trực tiếp:"
)

t_m0 = time.perf_counter()
res_mean = cv2.blur(noisy_img, (5, 5))
t_m = (time.perf_counter() - t_m0) * 1000
t_g0 = time.perf_counter()
res_gauss = cv2.GaussianBlur(noisy_img, (5, 5), 1.2)
t_g = (time.perf_counter() - t_g0) * 1000
t_med0 = time.perf_counter()
res_median = cv2.medianBlur(noisy_img, 5)
t_med = (time.perf_counter() - t_med0) * 1000
t_b0 = time.perf_counter()
res_bilateral = cv2.bilateralFilter(noisy_img, 9, 75, 75)
t_b = (time.perf_counter() - t_b0) * 1000

comp_results = [
    {
        "Bộ lọc": "Mean Filter (5×5)",
        "PSNR (dB)": psnr(clean_img, res_mean, data_range=255),
        "SSIM": ssim(clean_img, res_mean, data_range=255),
        "Thời gian (ms)": t_m,
        "Nhận xét kỹ thuật": (
            "Làm mịn nhanh nhưng nhòe mờ biên cạnh nghiêm trọng."
        ),
    },
    {
        "Bộ lọc": "Gaussian Filter (5×5)",
        "PSNR (dB)": psnr(clean_img, res_gauss, data_range=255),
        "SSIM": ssim(clean_img, res_gauss, data_range=255),
        "Thời gian (ms)": t_g,
        "Nhận xét kỹ thuật": "Làm mịn tự nhiên hơn Mean, khử nhiễu Gauss tốt.",
    },
    {
        "Bộ lọc": "Median Filter (5×5)",
        "PSNR (dB)": psnr(clean_img, res_median, data_range=255),
        "SSIM": ssim(clean_img, res_median, data_range=255),
        "Thời gian (ms)": t_med,
        "Nhận xét kỹ thuật": (
            "Vượt trội tuyệt đối trên nhiễu muối tiêu, giữ biên khá."
        ),
    },
    {
        "Bộ lọc": "Bilateral Filter (d=9)",
        "PSNR (dB)": psnr(clean_img, res_bilateral, data_range=255),
        "SSIM": ssim(clean_img, res_bilateral, data_range=255),
        "Thời gian (ms)": t_b,
        "Nhận xét kỹ thuật": (
            "Tối ưu nhất cho nhiễu Gauss, bảo toàn xuất sắc biên nét."
        ),
    },
]

c_show1, c_show2, c_show3, c_show4 = st.columns(4)
with c_show1:
  st.image(res_mean, caption="Mean Filter (5x5)", use_container_width=True)
with c_show2:
  st.image(res_gauss, caption="Gaussian Filter (5x5)", use_container_width=True)
with c_show3:
  st.image(res_median, caption="Median Filter (5x5)", use_container_width=True)
with c_show4:
  st.image(
      res_bilateral, caption="Bilateral Filter (d=9)", use_container_width=True
  )

comp_df = pd.DataFrame(comp_results)
comp_df["PSNR (dB)"] = comp_df["PSNR (dB)"].map(lambda x: f"{x:.2f}")
comp_df["SSIM"] = comp_df["SSIM"].map(lambda x: f"{x:.4f}")
comp_df["Thời gian (ms)"] = comp_df["Thời gian (ms)"].map(
    lambda x: f"{x:.2f} ms"
)
st.table(comp_df)

fig_bar, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10, 3.2), facecolor="#07192d"
)
filter_names = ["Mean", "Gaussian", "Median", "Bilateral"]
colors = ["#25cbe6", "#48d39c", "#ffad3d", "#8e69f2"]
psnr_scores = [float(r["PSNR (dB)"]) for r in comp_results]
ssim_scores = [float(r["SSIM"]) for r in comp_results]

ax1.set_facecolor("#07192d")
bars1 = ax1.bar(filter_names, psnr_scores, color=colors, width=0.55)
ax1.set_title(
    "So Sánh Chỉ Số PSNR (dB)", color="#cbd8e1", fontsize=11, fontweight="bold"
)
ax1.tick_params(colors="#8ba2b5", labelsize=9)
ax1.set_ylim(0, max(psnr_scores) * 1.25)
ax1.grid(axis="y", color="#1e3d59", linestyle=":", alpha=0.7)
for bar in bars1:
  yval = bar.get_height()
  ax1.text(
      bar.get_x() + bar.get_width() / 2.0,
      yval + 0.6,
      f"{yval:.1f}",
      ha="center",
      va="bottom",
      color="#ffffff",
      fontsize=9,
      fontweight="bold",
  )

ax2.set_facecolor("#07192d")
bars2 = ax2.bar(filter_names, ssim_scores, color=colors, width=0.55)
ax2.set_title(
    "So Sánh Chỉ Số SSIM", color="#cbd8e1", fontsize=11, fontweight="bold"
)
ax2.tick_params(colors="#8ba2b5", labelsize=9)
ax2.set_ylim(0, 1.15)
ax2.grid(axis="y", color="#1e3d59", linestyle=":", alpha=0.7)
for bar in bars2:
  yval = bar.get_height()
  ax2.text(
      bar.get_x() + bar.get_width() / 2.0,
      yval + 0.03,
      f"{yval:.3f}",
      ha="center",
      va="bottom",
      color="#ffffff",
      fontsize=9,
      fontweight="bold",
  )

st.pyplot(fig_bar)

if "Gaussian" in noise_type:
  st.success(
      "🎯 **Kết luận tự động từ hệ thống:** Trên mô hình **Nhiễu Gauss**,"
      " **Bilateral Filter** và **Gaussian Filter** đạt hiệu quả cao nhất."
      " Trong đó, Bilateral Filter vượt trội về việc duy trì độ tương đồng cấu"
      " trúc SSIM nhờ cơ chế bảo toàn biên (edge-preserving) đặc thù."
  )
else:
  st.success(
      "🎯 **Kết luận tự động từ hệ thống:** Trên mô hình **Nhiễu Muối Tiêu"
      " (Salt & Pepper)**, **Median Filter** là bộ lọc tối ưu và vượt trội hoàn"
      " toàn về cả PSNR lẫn SSIM nhờ cơ chế thống kê thứ tự triệt tiêu hạt cực"
      " trị."
  )

st.markdown(
    """
<div style="margin-top: 45px; background: #07192c; border-top: 3px solid #1cc9df; padding: 16px; text-align: center; color: #b8cbd9; font-size: 13px; border-radius: 10px;">
    <b>TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP.HCM (HUIT)</b> · Khoa Công Nghệ Thông Tin · Bộ Môn Khoa Học Dữ Liệu & Trí Tuệ Nhân Tạo<br>
    Đồ án môn học: Xử Lý Ảnh Số — Giảng viên hướng dẫn: TS. Nguyễn Huy Liêm
</div>
""",
    unsafe_allow_html=True,
)
