import tensorflow as tf
from flask import Flask, render_template, request
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from matplotlib import cm   # dùng colormap giống Kaggle
import gdown
import os
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# ================================
# Flask setup
# ================================
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ================================
# Load model
# ================================
MODEL_PATH = "xception_chestxray_finetuned1810.h5"
IMG_SIZE = (224, 224)           
LAST_CONV_LAYER = "block14_sepconv2_act"
# Nếu chưa có file .h5 thì tải tự động từ Google Drive
if not os.path.exists(MODEL_PATH):
    print("📥 Tải mô hình từ Google Drive...")
    file_id = "19yrwibqdKqkvrYUIEPIQAIr-ute1KcRo"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)
else:
    print("✅ Đã có sẵn mô hình.")

# Tải mô hình
print("🔹 Đang load mô hình...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# ================================
# Grad-CAM 
# ================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Grad-CAM ổn định cho Xception fine-tuned."""
    base_model = model.get_layer("xception")
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    with tf.GradientTape() as tape:
        conv_output = grad_model(img_array)
        tape.watch(conv_output)

        # Truyền tiếp qua phần classifier
        x = model.get_layer("global_average_pooling2d")(conv_output)
        x = model.get_layer("dropout")(x)
        preds = model.get_layer("dense")(x)

        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(conv_output * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

# ================================
# Prediction + Grad-CAM overlay
# ================================
def predict_xray(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)[0][0]
    label = "Pneumonia" if pred > 0.5 else "Normal"
    prob = pred if pred > 0.5 else 1 - pred
    prob_str = f"{prob:.2%}"

    # --- Sinh heatmap Grad-CAM ---
    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)

    # Colormap
    jet = cm.get_cmap("jet")
    viridis = cm.get_cmap("viridis")

    heatmap_jet = np.uint8(255 * jet(heatmap_resized)[:, :, :3])
    heatmap_viridis = np.uint8(255 * viridis(heatmap_resized)[:, :, :3])

    # Ảnh gốc
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, IMG_SIZE)

    # Overlay
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_jet, 0.4, 0)

    # Lưu ảnh
    base_name = os.path.basename(image_path)
    cam_path = os.path.join(app.config["UPLOAD_FOLDER"], "cam_" + base_name)
    jet_path = os.path.join(app.config["UPLOAD_FOLDER"], "cam_jet_" + base_name)
    overlay_path = os.path.join(app.config["UPLOAD_FOLDER"], "overlay_" + base_name)

    cv2.imwrite(cam_path, heatmap_viridis)
    cv2.imwrite(jet_path, heatmap_jet)
    cv2.imwrite(overlay_path, overlay)

    return label, prob, prob_str, cam_path, jet_path, overlay_path

# ================================
# Flask route
# ================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        label, prob, prob_str, cam_path, jet_path, overlay_path = predict_xray(filepath)
        return render_template(
            "index.html",
            uploaded_image=filepath,
            cam_image=cam_path,
            jet_image=jet_path,
            overlay_image=overlay_path,
            label=label,
            prob_str=prob_str
        )
    return render_template("index.html")

# ================================
# Run server
# ================================
if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
