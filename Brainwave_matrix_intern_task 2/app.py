import streamlit as st
import pickle
import re
from sklearn.feature_extraction import text

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Use sklearn's built-in stopwords
STOPWORDS = text.ENGLISH_STOP_WORDS

# Preprocessing function
def clean_text(content):
    text_cleaned = re.sub('[^a-zA-Z]', ' ', content)  # remove non-letters
    text_cleaned = text_cleaned.lower()
    text_cleaned = [word for word in text_cleaned.split() if word not in STOPWORDS]
    return ' '.join(text_cleaned)

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


