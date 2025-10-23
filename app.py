import tensorflow as tf
from flask import Flask, render_template, request
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from matplotlib import cm
import gdown

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_ONE = 'xception_chestxray_finetuned_val98.h5'
MODEL_TWO = 'mobilenetv2_chestxray_finetuned.h5'
MODEL_PATH = MODEL_ONE
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "block14_sepconv2_act"
# LAST_CONV_LAYER = "out_relu"

if not os.path.exists(MODEL_PATH):
    print("Tải mô hình từ Google Drive...")
    file_id = "19yrwibqdKqkvrYUIEPIQAIr-ute1KcRo"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)
else:
    print("✅ Đã có sẵn mô hình.")

print("🔹 Đang load mô hình...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")


def make_gradcam_heatmap(img_array, model, model_choice, last_conv_layer_name):
    import tensorflow as tf
    import numpy as np

    if model_choice == "xception":
        # ================== XCEPTION ==================
        base_model = model.get_layer("xception")
        last_conv_layer = base_model.get_layer(last_conv_layer_name)

        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=last_conv_layer.output
        )

        with tf.GradientTape() as tape:
            conv_output = grad_model(img_array)
            tape.watch(conv_output)

            x = tf.keras.layers.GlobalAveragePooling2D()(conv_output)
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

    elif model_choice == "mobilenetv2":
        # ================== MOBILENETV2 ==================
        try:
            base_model = model.get_layer("mobilenetv2_1.00_224")
        except:
            base_model = model.layers[0]

        last_conv_layer = base_model.get_layer(last_conv_layer_name)

        feature_extractor = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=last_conv_layer.output
        )

        with tf.GradientTape() as tape:
            conv_output = feature_extractor(img_array)
            tape.watch(conv_output)

            gap_layer = [l for l in model.layers if "global_average_pooling2d" in l.name][0]
            dropout_layer = [l for l in model.layers if "dropout" in l.name][0]
            dense_layer = [l for l in model.layers if "dense" in l.name][0]

            x = gap_layer(conv_output)
            x = dropout_layer(x)
            preds = dense_layer(x)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

def predict_xray(image_path, model, model_choice, last_conv_layer_name):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)[0][0]
    label = "Pneumonia" if pred > 0.5 else "Normal"
    prob = pred if pred > 0.5 else 1 - pred
    prob_str = f"{prob:.2%}"

    # ✅ thêm model_choice ở đây
    heatmap = make_gradcam_heatmap(img_array, model, model_choice, last_conv_layer_name)
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)

    jet = cm.get_cmap("jet")
    heatmap_jet = np.uint8(255 * jet(heatmap_resized)[:, :, :3])

    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, IMG_SIZE)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_jet, 0.4, 0)

    base_name = os.path.basename(image_path)
    jet_path = os.path.join(app.config["UPLOAD_FOLDER"], "cam_jet_" + base_name)
    overlay_path = os.path.join(app.config["UPLOAD_FOLDER"], "overlay_" + base_name)

    cv2.imwrite(jet_path, heatmap_jet)
    cv2.imwrite(overlay_path, overlay)

    return label, prob, prob_str, None, jet_path, overlay_path



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        model_choice = request.form.get("model_choice")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        if model_choice == "xception":
            model_path = MODEL_ONE
            last_layer = "block14_sepconv2_act"
        else:
            model_path = MODEL_TWO
            last_layer = "out_relu"

        model = tf.keras.models.load_model(model_path, compile=False)

        label, prob, prob_str, cam_path, jet_path, overlay_path = predict_xray(filepath, model, model_choice, last_layer)

        return render_template(
            "index.html",
            uploaded_image=filepath,
            jet_image=jet_path,
            overlay_image=overlay_path,
            label=label,
            prob_str=prob_str
        )
    return render_template("index.html")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    port = int(os.environ.get("PORT", 7860)) 
    app.run(host="0.0.0.0", port=port, debug=False)  
