
from flask import Flask, render_template, request, redirect, send_file
from database import create_database, save_prediction
import tensorflow as tf
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import json
import os
import cv2
import sqlite3
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)
create_database()

# =========================
# Load Model
# =========================

model = tf.keras.models.load_model(
    "model/crop_disease_model.keras"
)

# =========================
# Load Class Names
# =========================

with open("model/class_names.json", "r") as f:
    classes = json.load(f)

print("Total Classes:", len(classes))


# =========================
# Disease Information
# =========================

disease_info = {

    "Apple___Apple_scab": {
        "symptoms": "Olive-green to dark spots appear on apple leaves and fruits.",
        "management": "Remove infected leaves and fruits and maintain good air circulation."
    },

    "Apple___Black_rot": {
        "symptoms": "Brown or black spots may appear on leaves, branches, and fruits.",
        "management": "Remove infected plant parts and maintain good orchard sanitation."
    },

    "Apple___Cedar_apple_rust": {
        "symptoms": "Yellow-orange spots appear on apple leaves.",
        "management": "Remove infected leaves and maintain good orchard hygiene."
    },

    "Apple___healthy": {
        "symptoms": "The apple leaf appears healthy.",
        "management": "Maintain proper watering, nutrition and regular monitoring."
    },

    "Blueberry___healthy": {
        "symptoms": "The blueberry leaf appears healthy.",
        "management": "Maintain proper watering, nutrition and regular monitoring."
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "symptoms": "White powder-like growth appears on leaves and shoots.",
        "management": "Improve air circulation and remove severely affected plant parts."
    },

    "Cherry_(including_sour)___healthy": {
        "symptoms": "The cherry leaf appears healthy.",
        "management": "Maintain proper watering, nutrition and regular monitoring."
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "symptoms": "Gray or brown rectangular spots develop on maize leaves.",
        "management": "Maintain field hygiene and follow suitable disease-management practices."
    },

    "Corn_(maize)___Common_rust_": {
        "symptoms": "Reddish-brown rust-colored spots appear on maize leaves.",
        "management": "Monitor the crop regularly and follow suitable disease-management practices."
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "symptoms": "Long gray-green or brown lesions appear on maize leaves.",
        "management": "Maintain field hygiene and follow suitable disease-management practices."
    },

    "Corn_(maize)___healthy": {
        "symptoms": "The maize leaf appears healthy.",
        "management": "Maintain proper irrigation, nutrition and regular monitoring."
    },

    "Grape___Black_rot": {
        "symptoms": "Brown circular spots and dark lesions may appear on leaves and fruits.",
        "management": "Remove infected plant material and maintain vineyard sanitation."
    },

    "Grape___Esca_(Black_Measles)": {
        "symptoms": "Leaves may develop yellow or brown areas.",
        "management": "Remove severely affected plant material and maintain vineyard hygiene."
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "symptoms": "Dark brown or reddish spots appear on grape leaves.",
        "management": "Remove infected leaves and improve air circulation."
    },

    "Grape___healthy": {
        "symptoms": "The grape leaf appears healthy.",
        "management": "Maintain proper irrigation, nutrition and regular monitoring."
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "symptoms": "Leaves may show uneven yellowing and mottled green patterns.",
        "management": "Monitor plants regularly and manage insect vectors appropriately."
    },

    "Peach___Bacterial_spot": {
        "symptoms": "Small dark spots appear on peach leaves and fruits.",
        "management": "Remove infected plant material and maintain orchard sanitation."
    },

    "Peach___healthy": {
        "symptoms": "The peach leaf appears healthy.",
        "management": "Maintain proper watering, nutrition and regular monitoring."
    },

    "Pepper,_bell___Bacterial_spot": {
        "symptoms": "Small dark spots may appear on pepper leaves and fruits.",
        "management": "Remove infected plant material and maintain field sanitation."
    },

    "Pepper,_bell___healthy": {
        "symptoms": "The pepper leaf appears healthy.",
        "management": "Maintain proper irrigation, nutrition and regular monitoring."
    },

    "Potato___Early_blight": {
        "symptoms": "Brown circular spots with concentric rings appear on potato leaves.",
        "management": "Remove infected plant material and maintain field hygiene."
    },

    "Potato___Late_blight": {
        "symptoms": "Dark irregular spots appear on potato leaves.",
        "management": "Remove affected plant material and follow suitable local disease-management practices."
    },

    "Potato___healthy": {
        "symptoms": "The potato leaf appears healthy.",
        "management": "Maintain proper irrigation, nutrition and regular monitoring."
    },

    "Raspberry___healthy": {
        "symptoms": "The raspberry leaf appears healthy.",
        "management": "Maintain proper watering, nutrition and regular monitoring."
    },

    "Soybean___healthy": {
        "symptoms": "The soybean leaf appears healthy.",
        "management": "Maintain proper irrigation, nutrition and crop monitoring."
    },

    "Squash___Powdery_mildew": {
        "symptoms": "White powder-like patches appear on squash leaves.",
        "management": "Improve air circulation and remove severely affected leaves."
    },

    "Strawberry___Leaf_scorch": {
        "symptoms": "Dark purple to brown spots may appear on strawberry leaves.",
        "management": "Remove affected leaves and maintain field sanitation."
    },

    "Strawberry___healthy": {
        "symptoms": "The strawberry leaf appears healthy.",
        "management": "Maintain proper irrigation, nutrition and regular monitoring."
    },

    "Tomato___Bacterial_spot": {
        "symptoms": "Small dark spots appear on tomato leaves and fruits.",
        "management": "Remove infected plant material and maintain field sanitation."
    },

    "Tomato___Early_blight": {
        "symptoms": "Brown circular spots with concentric rings develop on tomato leaves.",
        "management": "Remove infected leaves and maintain proper plant spacing."
    },

    "Tomato___Late_blight": {
        "symptoms": "Dark irregular lesions appear on tomato leaves and stems.",
        "management": "Remove affected plant material and avoid overhead watering."
    },

    "Tomato___Leaf_Mold": {
        "symptoms": "Yellow patches may appear on leaves with olive-green growth underneath.",
        "management": "Improve air circulation and reduce excessive humidity."
    },

    "Tomato___Septoria_leaf_spot": {
        "symptoms": "Small circular spots with dark borders appear on tomato leaves.",
        "management": "Remove infected leaves and maintain field sanitation."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "symptoms": "Leaves may develop small yellow or pale spots.",
        "management": "Monitor leaves regularly and follow suitable pest-management practices."
    },

    "Tomato___Target_Spot": {
        "symptoms": "Brown circular lesions with concentric rings appear on tomato leaves.",
        "management": "Remove affected leaves and improve air circulation."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "symptoms": "Leaves may curl upward and become yellow.",
        "management": "Monitor and manage whitefly populations and follow local agricultural guidance."
    },

    "Tomato___Tomato_mosaic_virus": {
        "symptoms": "Leaves may show light and dark green mosaic patterns.",
        "management": "Remove infected plants where appropriate and sanitize tools."
    },

    "Tomato___healthy": {
        "symptoms": "The tomato leaf appears healthy.",
        "management": "Maintain proper watering, nutrition and regular monitoring."
    }
}


# =========================
# Home Page
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# Prediction
# =========================
# =========================
# Prediction
# =========================
def check_image_quality(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # Low score = very blurry image
    if blur_score < 50:
        return False

    return True
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =========================
        # Get Uploaded File
        # =========================

        file = request.files.get("image")
        allowed_extensions = {
            "jpg",
            "jpeg",
            "png"
        }

        file_extension = (
            file.filename.rsplit(".", 1)[-1].lower()
            if "." in file.filename
            else ""
        )

        if file_extension not in allowed_extensions:
            return "Please upload a JPG, JPEG, or PNG image."

        if file is None or file.filename == "":
            return "Please select an image."

        # =========================
        # Save Uploaded Image
        # =========================

        os.makedirs("static/uploads", exist_ok=True)

        filename = secure_filename(file.filename)

        image_path = os.path.join(
            "static",
            "uploads",
            filename
        )

        file.save(image_path)

        # =========================
        # Image Quality Check
        # =========================

        if not check_image_quality(image_path):

            return render_template(
                "result.html",
                disease="Image Quality Too Low",
                confidence=0,
                symptoms="The uploaded image appears blurry or unclear.",
                management="Please upload a clear crop leaf image with good lighting.",
                image_path=image_path,
                top_predictions=[]
            )

        # =========================
        # Open Image
        # =========================

        image = Image.open(
            image_path
        ).convert("RGB")

        # =========================
        # Resize Image
        # =========================

        input_size = model.input_shape[1:3]

        image = image.resize(input_size)

        # =========================
        # Convert to Array
        # =========================

        img = np.array(
            image,
            dtype=np.float32
        )

        img = np.expand_dims(
            img,
            axis=0
        )

        # =========================
        # Prediction
        # =========================

        prediction = model.predict(
            img,
            verbose=0
        )

        # =========================
        # Top 3 Predictions
        # =========================

        top_indices = np.argsort(
            prediction[0]
        )[-3:][::-1]

        top_predictions = []

        for i in top_indices:

            disease_name = classes[i]

            confidence_value = (
                float(prediction[0][i]) * 100
            )

            display_name = (
                disease_name
                .replace("___", " — ")
                .replace("_", " ")
            )

            top_predictions.append({
                "disease": display_name,
                "confidence": round(
                    confidence_value,
                    2
                )
            })

        # =========================
        # Main Prediction
        # =========================

        disease = classes[top_indices[0]]

        confidence = (
            float(
                prediction[0][top_indices[0]]
            ) * 100
        )

        display_disease = (
            disease
            .replace("___", " — ")
            .replace("_", " ")
        )

        # =========================
        # Disease Information
        # =========================

        info = disease_info.get(
            disease,
            {
                "symptoms": "Information not available.",
                "management": "Please consult an agricultural expert."
            }
        )

        # =========================
        # Save Prediction
        # =========================

        save_prediction(
            filename,
            display_disease,
            round(confidence, 2)
        )

        # =========================
        # Show Result
        # =========================

        return render_template(
            "result.html",
            disease=display_disease,
            confidence=round(confidence, 2),
            symptoms=info["symptoms"],
            management=info["management"],
            image_path=image_path,
            top_predictions=top_predictions
        )

    except Exception as e:

        print("ERROR:", e)

        return f"Error occurred: {e}"
    # =========================
# Dashboard
# =========================
# =========================
# Disease Knowledge Center
# =========================

@app.route("/knowledge")
def knowledge():
    return render_template(
        "knowledge.html",
        disease_info=disease_info
    )
# =========================
# Generate PDF Report
# =========================

@app.route("/download_report")
def download_report():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT disease, confidence, date_time
        FROM predictions
        ORDER BY id DESC
        LIMIT 1
    """)

    prediction_data = cursor.fetchone()

    conn.close()
    filepath = os.path.join(
        "static",
        "crop_disease_report.pdf"
    )

    c = canvas.Canvas(
        filepath,
        pagesize=A4
    )

    width, height = A4

    c.setFont("Helvetica-Bold", 20)

    c.drawCentredString(
        width / 2,
        height - 60,
        "AI Crop Disease Prediction Report"
    )

    c.setFont("Helvetica", 12)

    c.drawString(
        60,
        height - 110,
        "Generated by AI Crop Disease Prediction System"
    )

    c.drawString(
        60,
        height - 150,
        "Preliminary AI-based crop disease analysis."
    )
    if prediction_data:

     disease = prediction_data[0]
    confidence = prediction_data[1]
    date_time = prediction_data[2]
    info = {
            "symptoms": "Information not available.",
            "management": "Please consult an agricultural expert."
        }

    for key, value in disease_info.items():
            display_key = key.replace("___", " — ").replace("_", " ")

            if display_key == disease:
                info = value
                break

    c.setFont("Helvetica-Bold", 14)

    c.drawString(
        60,
        height - 200,
        "Prediction Details"
    )

    c.setFont("Helvetica", 12)

    c.drawString(
        60,
        height - 230,
        f"Disease: {disease}"
    )

    c.drawString(
        60,
        height - 255,
        f"Confidence: {confidence}%"
    )

    c.drawString(
        60,
        height - 280,
        f"Date & Time: {date_time}"
    )
    c.setFont("Helvetica-Bold", 14)

    c.drawString(
            60,
            height - 320,
            "Symptoms"
        )

    c.setFont("Helvetica", 11)

    c.drawString(
            60,
            height - 345,
            info["symptoms"]
        )

    c.setFont("Helvetica-Bold", 14)

    c.drawString(
            60,
            height - 385,
            "Management & Prevention"
        )

    c.setFont("Helvetica", 11)

    c.drawString(
            60,
            height - 410,
            info["management"]
        )
    

    c.save()

    return send_file(
        filepath,
        as_attachment=True
    )
@app.route("/dashboard")
def dashboard():


    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE LOWER(disease) LIKE '%healthy%'
    """)
    healthy = cursor.fetchone()[0]

    diseased = total - healthy

    health_data = {
        "healthy": healthy,
        "diseased": diseased
    }
    cursor.execute("""
        SELECT AVG(confidence)
        FROM predictions
    """)

    avg_confidence = cursor.fetchone()[0]

    cursor.execute("""
        SELECT disease, COUNT(*)
        FROM predictions
        GROUP BY disease
        ORDER BY COUNT(*) DESC
    """)

    disease_stats = cursor.fetchall()
    cursor.execute("""
        SELECT image_name, disease, confidence, date_time
        FROM predictions
        ORDER BY id DESC
        LIMIT 5
    """)

    recent_predictions = cursor.fetchall()

    conn.close()
    return render_template(
        "dashboard.html",
        total=total,
        healthy=healthy,
        diseased=diseased,
        health_data=health_data,
        avg_confidence=round(avg_confidence or 0, 2),
        disease_stats=disease_stats,
        recent_predictions=recent_predictions
    )
# =========================
# Prediction History
# =========================

@app.route("/history")
def history():

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """)

    predictions = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        predictions=predictions
    )
    # =========================
# Run App
# =========================

if __name__ == "__main__":
    app.run(debug=True)