import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Classifier")
st.write("Enter a news headline or article text to check if it's Real or Fake.")

user_input = st.text_area("News Text:")

if st.button("Classify"):
    if user_input.strip():
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        result = "✅ Real News" if prediction == 1 else "❌ Fake News"
        st.subheader(result)
    else:
        st.warning("Please enter some text.")
