
import pickle
from pathlib import Path


# Locate the directory containing this Python file
BASE_DIR = Path(__file__).resolve().parent


def load_artifacts():
    """Load the trained model, vectorizer and decision threshold."""

    with open(BASE_DIR / "spam_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open(BASE_DIR / "vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    with open(BASE_DIR / "threshold.pkl", "rb") as file:
        threshold = pickle.load(file)

    return model, vectorizer, threshold


def predict_message(message, model, vectorizer, threshold):
    """Classify an SMS message as spam or ham."""

    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please enter a valid SMS message.")

    # Convert the message into TF-IDF features
    message_vector = vectorizer.transform([message])

    # Obtain the model's predicted probabilities
    probabilities = model.predict_proba(message_vector)[0]

    # Find the probability corresponding to the spam class
    spam_index = list(model.classes_).index("spam")
    spam_probability = probabilities[spam_index]

    # Apply the selected decision threshold
    prediction = "spam" if spam_probability >= threshold else "ham"

    return {
        "prediction": prediction,
        "spam_probability": float(spam_probability),
        "threshold": threshold
    }

