```python
import streamlit as st
from classifier import load_artifacts, predict_message


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SMS Guard | Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# LOAD MODEL ARTIFACTS
# --------------------------------------------------

@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    model, vectorizer, threshold = get_artifacts()

except FileNotFoundError as error:
    st.error(f"Required model file not found: {error.filename}")
    st.info("Please ensure all three model files are in the repository.")
    st.stop()


# --------------------------------------------------
# DARK DASHBOARD STYLING
# --------------------------------------------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #0B1120;
    color: #E2E8F0;
}

/* Main content width */
.block-container {
    max-width: 1150px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111C30;
    border-right: 1px solid #24344F;
}

section[data-testid="stSidebar"] * {
    color: #CBD5E1;
}

/* Header */
.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -1px;
}

.subtitle {
    color: #94A3B8;
    font-size: 1rem;
    margin-top: 0.5rem;
    margin-bottom: 2rem;
}

.section-title {
    color: #E2E8F0;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

/* Accent text */
.accent {
    color: #38BDF8;
}

/* Text input */
.stTextArea textarea
