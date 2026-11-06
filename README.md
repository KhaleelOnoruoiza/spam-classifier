# SMS Guard – Spam SMS Detection

**Live Demo:** [Try SMS Guard](https://spam-classifier-rzpsvf3hwgdvfjxov6kbef.streamlit.app/)

## Overview

SMS Guard is a machine learning application that classifies SMS messages as spam or legitimate (ham) using TF-IDF and Multinomial Naive Bayes.

## Dataset

Trained on 5,572 SMS messages:
- 4,825 legitimate messages
- 747 spam messages

## Model Performance

Tested on 837 unseen messages:

- **Accuracy:** 98%
- **Spam Precision:** 100%
- **Spam Recall:** 84%
- **Spam F1-score:** 91%

A decision threshold of 0.20 was selected through validation.

## Tech Stack

Python, Scikit-learn, Pandas, NumPy, TF-IDF and Streamlit.

## Run Locally

```bash
git clone https://github.com/KhaleelOnoruoiza/spam-classifier.git
cd spam-classifier
pip install -r requirements.txt
streamlit run app.py
```

Ensure `spam_model.pkl`, `vectorizer.pkl` and `threshold.pkl` are in the project directory.

## Author

**Khaleel Sadiq Onoruoiza**

[GitHub](https://github.com/KhaleelOnoruoiza) | [Live App](https://spam-classifier-rzpsvf3hwgdvfjxov6kbef.streamlit.app/)
