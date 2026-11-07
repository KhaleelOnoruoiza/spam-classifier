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

/* Main heading */
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

.accent {
    color: #38BDF8;
}

/* Text input */
.stTextArea textarea {
    background-color: #111C30 !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
}

.stTextArea textarea:focus {
    border: 1px solid #38BDF8 !important;
    box-shadow: 0 0 0 1px #38BDF8 !important;
}

/* Submit button */
div[data-testid="stFormSubmitButton"] button {
    background-color: #0284C7;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.7rem 1rem;
    font-weight: 700;
    width: 100%;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #0369A1;
    color: white;
    border: none;
}

/* General buttons */
div.stButton > button {
    background-color: #0284C7;
    color: white;
    border: none;
    border-radius: 9px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #111C30;
    border: 1px solid #24344F;
    border-radius: 12px;
    padding: 18px;
}

div[data-testid="stMetric"] label {
    color: #94A3B8 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #F8FAFC !important;
}

/* Prediction panels */
.result-spam {
    background-color: #351923;
    border: 1px solid #9F3347;
    border-left: 5px solid #EF4444;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
}

.result-ham {
    background-color: #102D2A;
    border: 1px solid #237A65;
    border-left: 5px solid #22C55E;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
}

.result-spam h3 {
    color: #FCA5A5;
    margin-top: 0;
}

.result-ham h3 {
    color: #86EFAC;
    margin-top: 0;
}

.result-spam p,
.result-ham p {
    color: #E2E8F0;
    margin-bottom: 0;
}

/* General text */
.stMarkdown p {
    color: #CBD5E1;
}

hr {
    border-color: #24344F;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🛡️ SMS Guard")

    st.caption("INTELLIGENT MESSAGE SECURITY")

    st.divider()

    st.subheader("About this project")

    st.write(
        "SMS Guard uses machine learning to identify "
        "potential spam messages and distinguish them "
        "from legitimate SMS messages."
    )

    st.divider()

    st.subheader("Model information")

    st.write("**Algorithm:** Multinomial Naive Bayes")
    st.write("**Feature extraction:** TF-IDF")
    st.write(f"**Decision threshold:** {threshold:.2f}")

    st.divider()

    st.subheader("Technology")

    st.write("Python")
    st.write("Scikit-learn")
    st.write("Streamlit")

    st.divider()

    st.caption("Developed by Khaleel Onoruoiza")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    'SMS Message <span class="accent">Analysis</span>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyse SMS messages and identify potential spam '
    'using a trained machine learning model.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# MESSAGE INPUT
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Enter your message</div>',
    unsafe_allow_html=True
)

with st.form("message_form"):

    msg = st.text_area(
        "SMS message",
        placeholder="Paste or type your SMS message here...",
        height=180,
        label_visibility="collapsed"
    )

    submitted = st.form_submit_button(
        "🔍 Analyse Message",
        use_container_width=True
    )


# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------

if submitted:

    try:
        result = predict_message(
            msg,
            model,
            vectorizer,
            threshold
        )

        prediction = result["prediction"]
        spam_probability = result["spam_probability"]

        st.divider()

        st.subheader("Classification result")

        if prediction == "spam":

            st.markdown(
                """
                <div class="result-spam">
                    <h3>⚠️ Potential Spam Detected</h3>
                    <p>
                        The model has classified this message
                        as spam.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-ham">
                    <h3>✓ Message Classified as Ham</h3>
                    <p>
                        The model has classified this message
                        as ham (not spam).
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Analysis details
        st.subheader("Analysis details")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Estimated spam probability",
                f"{spam_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Decision threshold",
                f"{threshold * 100:.0f}%"
            )

        st.write("Spam probability")

        st.progress(float(spam_probability))

        st.caption(
            "The probability is the model's estimate, not a "
            "guarantee. A message classified as ham may still "
            "be unwanted or fraudulent."
        )

    except ValueError as error:
        st.warning(str(error))


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()

st.subheader("Model performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Test accuracy", "~98%")

with col2:
    st.metric("Spam recall", "~84%")

with col3:
    st.metric("Spam precision", "100%")

st.caption(
    "Performance figures are from the reported held-out test "
    "evaluation. Actual results may vary on new messages."
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align: center; color: #64748B;">
        SMS Guard | Machine Learning Spam Detection
    </div>
    """,
    unsafe_allow_html=True
)
```
