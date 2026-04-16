from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Create static folder if not exists
if not os.path.exists("static"):
    os.makedirs("static")

# Load trained model
model = load_model("whr_model.h5", compile=False)

IMG_SIZE = 128

# 🔷 WHR Classification
def classify_whr(whr):
    if whr < 0.85:
        return "Normal"
    elif whr < 1.0:
        return "Central Obesity"
    else:
        return "High Risk"

# 🔷 Image Preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        return None

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 Improved silhouette processing
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)

    _, img_thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5,5), np.uint8)
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

    # Ensure body is white
    if np.sum(img_thresh == 255) < np.sum(img_thresh == 0):
        img_thresh = cv2.bitwise_not(img_thresh)

    # Resize and normalize
    img_resized = cv2.resize(img_thresh, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0

    img_input = img_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return img_input

# 🔷 Main Route
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        # Get uploaded file
        file = request.files.get("image")

        if file is None or file.filename == "":
            return render_template("index.html", error="No file uploaded")

        file_path = os.path.join("static", "uploaded.png")
        file.save(file_path)

        # Preprocess
        img_input = preprocess_image(file_path)

        if img_input is None:
            return render_template("index.html", error="Invalid image")

        # Prediction
        pred_whr = model.predict(img_input)[0][0]
        category = classify_whr(pred_whr)

        return render_template(
            "index.html",
            prediction=round(float(pred_whr), 3),
            category=category,
            image=file_path
        )

    return render_template("index.html")

# 🔷 Run App
if __name__ == "__main__":
    app.run(debug=True)


# @app.route("/")
# def test():
#     return "Hello World"