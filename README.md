# ỨNG DỤNG PHÁT HIỆN VIÊM PHỔI TỪ ẢNH X-QUANG NGỰC

---

## Tổng quan

Dự án này sử dụng mô hình học sâu **MobileNetV2 / Xception** để phát hiện **bệnh viêm phổi (Pneumonia)** từ ảnh X-quang ngực.  
Bộ dữ liệu được thu thập từ **Trung tâm Y tế Phụ nữ và Trẻ em Quảng Châu (Trung Quốc)**, bao gồm các ảnh X-quang của trẻ từ 1–5 tuổi, được chẩn đoán và gán nhãn bởi hai bác sĩ chuyên khoa hô hấp.

Mô hình được tích hợp **Grad-CAM (Gradient-weighted Class Activation Mapping)** để trực quan hóa vùng phổi mà hệ thống tập trung vào khi ra quyết định.

Ứng dụng được xây dựng bằng **TensorFlow + Flask**, cho phép người dùng tải ảnh X-quang lên và nhận kết quả dự đoán cùng bản đồ kích hoạt (heatmap) trực tiếp trên giao diện web.

---

## 1 Mục tiêu

- Tự động phân loại ảnh X-quang thành **Bình thường (Normal)** hoặc **Viêm phổi (Pneumonia)**.
- Trực quan hóa khu vực nghi ngờ bằng **Grad-CAM** để tăng tính minh bạch của mô hình.
- Cung cấp giao diện web đơn giản, thân thiện, dễ sử dụng cho người dùng không chuyên kỹ thuật.

---

## 2 Dữ liệu

**Nguồn:** [Chest X-Ray Images (Pneumonia) – Kaggle Dataset](https://www.kaggle.com/datasets/ghost5612/chest-x-ray-images-normal-and-pneumonia)

**Tổng cộng:** 5.887 ảnh JPEG, chia theo ba tập:

| Tập dữ liệu | Số ảnh | NORMAL        | PNEUMONIA     |
| ----------- | ------ | ------------- | ------------- |
| Train       | 5.216  | 1.341 (25.7%) | 3.875 (74.3%) |
| Validation  | 47     | 24 (51%)      | 23 (49%)      |
| Test        | 624    | 234 (37.5%)   | 390 (62.5%)   |

---

## 3 Công nghệ sử dụng

| Thành phần              | Công cụ / Thư viện                          |
| ----------------------- | ------------------------------------------- |
| Ngôn ngữ                | Python 3.11                                 |
| Học sâu (Deep Learning) | TensorFlow / Keras                          |
| Framework web           | Flask                                       |
| Xử lý ảnh               | OpenCV, NumPy                               |
| Trực quan hóa           | Matplotlib, cv2 colormap (JET / VIRIDIS)    |
| Triển khai              | Localhost, Render, hoặc Hugging Face Spaces |

---

## 4 Cách chạy ứng dụng

- Bước 1. Tải mã nguồn :

```
git clone https://github.com/minhsuy/chest_xray_web.git
cd chest_xray_web

```

- Bước 2. Tạo và kích hoạt môi trường ảo :

```
python -m venv venv
venv\Scripts\activate       # Windows
# hoặc
source venv/bin/activate    # macOS / Linux
```

- Bước 3. Cài đặt các thư viện cần thiết :

```
pip install -r requirements.txt
```

- Bước 4 : Tải mô hình huấn luyện :

- Bước 5. Chạy ứng dụng :

```

python app.py

```

- Bước 6. Mở trình duyệt :

```

- Browser : http://127.0.0.1:5000

- Sau đó tải ảnh X-quang lên để xem:

   Kết quả dự đoán: Bình thường hoặc Viêm phổi

   Xác suất dự đoán (%)

   Bản đồ Grad-CAM tô sáng vùng phổi mà mô hình nhận diện là đáng ngờ

```

## 5 Kết quả mô hình

```

| Chỉ số                  | Giá trị |
| ----------------------- | ------- |
| Độ chính xác (Accuracy) | 88.46%  |

```

## 6 Ví dụ kết quả trực quan :

| Ảnh gốc                                            | Bản đồ kích hoạt (Grad-CAM)                                | Ảnh chồng (Overlay)                                           |
| -------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------- |
| ![Original](static/uploads/person11_virus_38.jpeg) | ![Grad-CAM](static/uploads/cam_jet_person11_virus_38.jpeg) | ![Overlay](static/uploads/overlay_person90_bacteria_442.jpeg) |

## 7 Cấu trúc thư mục

```

chest_xray_web/
├── app.py # Flask web app
├── xception_chestxray_finetuned1810.h5 # Mô hình đã huấn luyện
├── requirements.txt # Danh sách thư viện cần thiết
├── templates/
│ └── index.html # Giao diện người dùng
├── static/
│ ├── uploads/ # Ảnh do người dùng tải lên
│ └── results/ # Ảnh kết quả Grad-CAM
└── README.md

```

```

```
