import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Simple preprocessing function (without NLTK stemmer)
def clean_text(content):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', content)
    # Lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    text = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return ' '.join(text)

# Streamlit UI
st.title("Twitter Sentiment Analysis")
st.subheader("Predict if a Tweet is Positive or Negative")

tweet = st.text_area("✍️ Enter a tweet:")

if st.button("Analyze Sentiment"):
    if tweet.strip() != "":
        processed_text = clean_text(tweet)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 0:
            st.error("❌ Negative Tweet")
        else:
            st.success("✅ Positive Tweet")
    else:
        st.warning("⚠️ Please enter a tweet before analyzing.")

