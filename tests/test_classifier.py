
import pytest

from classifier import load_artifacts, predict_message


# Load the trained model and supporting files
model, vectorizer, threshold = load_artifacts()


def test_spam_message():
    """Check that a known spam message is classified as spam."""

    result = predict_message(
        "Congratulations! You have won a free prize. Call now!",
        model,
        vectorizer,
        threshold
    )

    assert result["prediction"] == "spam"


def test_ham_message():
    """Check that a legitimate message is classified as ham."""

    result = predict_message(
        "Hi, are you coming for class today?",
        model,
        vectorizer,
        threshold
    )

    assert result["prediction"] == "ham"


def test_empty_message():
    """Check that an empty message raises a ValueError."""

    with pytest.raises(ValueError):
        predict_message(
            "",
            model,
            vectorizer,
            threshold
        )


def test_whitespace_message():
    """Check that a message containing only spaces is rejected."""

    with pytest.raises(ValueError):
        predict_message(
            "   ",
            model,
            vectorizer,
            threshold
        )


def test_prediction_probability_range():
    """Check that the spam probability is between 0 and 1."""

    result = predict_message(
        "Hello, how are you doing today?",
        model,
        vectorizer,
        threshold
    )

    assert 0.0 <= result["spam_probability"] <= 1.0


def test_prediction_threshold():
    """Check that the prediction follows the configured threshold."""

    result = predict_message(
        "Congratulations! You have won a free prize. Call now!",
        model,
        vectorizer,
        threshold
    )

    if result["spam_probability"] >= threshold:
        assert result["prediction"] == "spam"
    else:
        assert result["prediction"] == "ham"
def test_artifacts_load_successfully():
    """Verify that all three model artifacts load correctly."""

    model, vectorizer, threshold = load_artifacts()

    assert model is not None
    assert vectorizer is not None
    assert threshold == 0.20
