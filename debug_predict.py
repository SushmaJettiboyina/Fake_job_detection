import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    print(f"-- {name} --")
    print(f"Non-zero tokens: {nonzero}/{len(seq[0])}")
    print(f"Raw prediction probability: {pred:.4f}")
    print(f"Label (threshold>0.7 => Fraudulent): {'Fraudulent' if pred>0.7 else 'Legitimate'}")
    print()

print('Done.')
