import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Explanation helper
from explainers import generate_explanations

MAX_SEQUENCE_LENGTH = 200

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = tf.keras.models.load_model("fake_job_lstm_model.h5")

examples = {
    "legit_example": "Title: Software Engineer\nCompany: Goldman Sachs\nLocation: Bengaluru, India\nDescription: Develop scalable software, architect low-latency infrastructure, and leverage machine learning to turn data into action. Requirements: Bachelor's in Computer Science or related field. How to Apply: Visit the Goldman Sachs careers page: https://www.goldmansachs.com/careers/our-firm/engineering",
    "fraud_example": "Title: Junior Developer - Immediate Hiring\nCompany: Future AI Solutions\nLocation: Remote\nDescription: Seeking enthusiastic developers to work on AI projects with international clients. No experience needed. Requirements: No experience required; basic Python preferred. How to Apply: Apply via careers@futureaisolutions.com. A refundable ₹4,999 deposit is required for training materials.",
    "user_example": "Add your own job post here"
}

for name, text in examples.items():
    seq = tokenizer.texts_to_sequences([text])
    nonzero = sum([1 for w in seq[0] if w != 0])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)[0][0]
    probability_pct = round(pred * 100, 1)
    label = 'Fraudulent' if pred > 0.7 else 'Legitimate'

    print(f"-- {name} --")
    print(f"Non-zero tokens: {nonzero}/{len(seq[0])}")
    print(f"Probability (fake): {probability_pct}%")
    print(f"Prediction label: {label} (threshold>0.7 => Fraudulent)")

    # Visual bar mapping for quick inspection
    # Display bar uses 100 - probability so high fake probability yields low fill
    bar_pct = round(100 - probability_pct, 1)
    if label == 'Fraudulent':
        bar_label = f"{probability_pct}% Fraudulent (display {bar_pct}% fill)"
    else:
        bar_label = f"{bar_pct}% Legitimate"
    print(f"Display bar: {bar_label} (fill {bar_pct}%)")

    # Explanations
    explanations = generate_explanations(text, description=text)
    if explanations:
        print("Explanations:")
        for e in explanations:
            print(f"- {e}")
    print()

print('Done.')
