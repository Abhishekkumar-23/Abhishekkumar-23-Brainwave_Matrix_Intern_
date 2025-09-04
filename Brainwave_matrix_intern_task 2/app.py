import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load saved model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Stemmer
port_stem = PorterStemmer()

# Text preprocessing function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Streamlit UI
st.title("Twitter Sentiment Analysis")
st.subheader("Check if a tweet is Positive or Negative")

# User input
user_input = st.text_area("Enter a tweet here...")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        processed_text = stemming(user_input)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 0:
            st.error("❌ Negative Tweet")
        else:
            st.success("✅ Positive Tweet")
    else:
        st.warning("⚠️ Please enter a tweet.")
