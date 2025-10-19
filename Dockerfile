# Dùng Python 3.11 (tương thích với TensorFlow 2.15)
FROM python:3.11

# Tạo thư mục làm việc
WORKDIR /app

# Copy toàn bộ mã nguồn
COPY . /app

# Cài đặt dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose cổng 7860 (Hugging Face Spaces yêu cầu)
EXPOSE 7860

# Chạy Flask app
ENV PORT 7860
CMD ["python", "app.py"]
