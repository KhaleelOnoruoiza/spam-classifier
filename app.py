import streamlit as st
import pickle

st.set_page_config(page_title="Spam Classifier", page_icon="📩")
#Load
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f: 
    vectorizer = pickle.load(f)
st.title("📩 Spam SMS Detector") 
st.write("Built by Khaleel Onoruoiza | Accuracy: 96.5%")
st.write("---")

msg = st.text_area("Enter SMS message:", height=120)
if st.button("Check"):
    if msg.strip() == "": 
        st.warning("Type a message first")
    else: 
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)[0] 
        if str(pred).lower() == "spam" or pred == 1:
            st.error("🚨 SPAM")
        else: st.success("✅ HAM (Not Spam)")
st.caption("Model: TF-IDF + Multinomial Naive Bayes")
