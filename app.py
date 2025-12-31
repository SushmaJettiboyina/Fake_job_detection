import os
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Reduce TensorFlow logging noise
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)

# -------------------- LOAD TOKENIZER --------------------
print("Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded!")

# -------------------- LOAD FULL TF MODEL (.h5) --------------------
print("Loading full TensorFlow model...")
model = tf.keras.models.load_model("fake_job_lstm_model.h5")
print("Model loaded!")

# -------------------- CONSTANTS --------------------
MAX_SEQUENCE_LENGTH = 200

# -------------------- TEXT PREPROCESSING --------------------
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return padded

# -------------------- ROUTES --------------------
@app.route("/", methods=["GET"])
def home():
    # Provide an empty form_values mapping so template can access form fields safely on GET
    return render_template("index.html", form_values={})

@app.route("/predict", methods=["POST"])
def predict():
    # Accept separate fields (fall back to legacy 'combined_text' if provided)
    title = request.form.get("title", "").strip()
    company = request.form.get("company", "").strip()
    location = request.form.get("location", "").strip()
    description = request.form.get("description", "").strip()
    requirements = request.form.get("requirements", "").strip()
    how_to_apply = request.form.get("how_to_apply", "").strip()
    legacy = request.form.get("combined_text", "").strip()

    # Build combined text (use legacy if provided)
    if legacy:
        combined_text = legacy
    else:
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if company:
            parts.append(f"Company: {company}")
        if location:
            parts.append(f"Location: {location}")
        if description:
            parts.append(f"Description: {description}")
        if requirements:
            parts.append(f"Requirements: {requirements}")
        if how_to_apply:
            parts.append(f"How to Apply: {how_to_apply}")
        combined_text = "\n".join(parts)

    if not combined_text.strip():
        return render_template(
            "index.html",
            prediction="Please enter job details.",
            form_values=request.form
        )

    # Tokenization and token-count debug info
    sequence = tokenizer.texts_to_sequences([combined_text])
    nonzero_tokens = sum(1 for w in sequence[0] if w != 0)

    # Preprocess input
    input_data = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    # Make prediction using FULL TensorFlow model
    prediction = float(model.predict(input_data)[0][0])

    # Classification threshold and human-friendly labels
    label = "Fake" if prediction > 0.7 else "Real"

    # Only return the classification label (no probability, tokens, or full text)
    return render_template(
        "index.html",
        prediction=f"The job post is {label}",
        form_values=request.form
    )

# -------------------- MAIN --------------------
if __name__ == "__main__":
    app.run(debug=True)
