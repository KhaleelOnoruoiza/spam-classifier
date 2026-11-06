

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
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F5F7FB;
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 750;
    color: #172B4D;
    letter-spacing: -1px;
}

.subtitle {
    color: #64748B;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.section-title {
    color: #172B4D;
    font-size: 1.15rem;
    font-weight: 650;
}

div.stButton > button {
    background-color: #1677D2;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 0.65rem 1rem;
    font-weight: 600;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #105DA8;
    color: white;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
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

    st.caption("Developed by Khaleel Onoruoiza")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">SMS Message Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Check whether an SMS message is potentially spam '
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
        height=160,
        label_visibility="collapsed"
    )

    submitted = st.form_submit_button(
        "Analyse Message"
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

        if prediction == "spam":

            st.error("Potential Spam Detected")

            st.write(
                "The model has classified this message as spam."
            )

        else:

            st.success("Message Classified as Ham")

            st.write(
                "The model has classified this message as "
                "ham (not spam)."
            )

        # Display probability information
        st.divider()

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
