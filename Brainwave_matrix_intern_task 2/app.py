import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Stemmer
port_stem = PorterStemmer()

# Preprocessing function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in ENGLISH_STOP_WORDS]
    return ' '.join(stemmed_content)

# Streamlit UI
st.title("Twitter Sentiment Analysis")
st.subheader("Predict if a Tweet is Positive or Negative")

tweet = st.text_area("✍️ Enter a tweet:")

if st.button("Analyze Sentiment"):
    if tweet.strip() != "":
        processed_text = stemming(tweet)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 0:
            st.error("❌ Negative Tweet")
        else:
            st.success("✅ Positive Tweet")
    else:
        st.warning("⚠️ Please enter a tweet before analyzing.")
