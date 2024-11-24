      
import os
import re
import nltk
import gensim
import joblib
import numpy as np
import pandas as pd 
import streamlit as st
import sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # For progress bar 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



# Define paths to model files (assuming they're in the same directory as app.py)
WORD2VEC_MODEL_PATH = "word2vec_model.joblib"
XGB_MODEL_PATH = "best_model.joblib"
ENCODER_PATH = "label_encoder.joblib"
SCALER_PATH = "scaler.joblib"

# Load pre-trained models and scaler
try:
    word2vec_model = joblib.load(WORD2VEC_MODEL_PATH)
    model = joblib.load(XGB_MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    class_names = encoder.classes_
    vector_size = word2vec_model.vector_size
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()



# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)         # Tokenize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)      # Return cleaned text as a string



def get_avg_word2vec(tokens, model, vector_size):
    valid_tokens = [token for token in tokens if token in model.wv.key_to_index]
    if not valid_tokens:
        return np.zeros(vector_size) # Return zero vector if no tokens are found.  Consider a better default.
    return np.mean([model.wv[token] for token in valid_tokens], axis = 0)



# Streamlit UI
st.title('Binary Review Classification')
st.write('Enter a review, and the model will classify it.')

review_text = st.text_area('Enter the review here:')



if review_text:
    with st.spinner('Processing...'): # Progress indicator
        cleaned_review = clean_text(review_text)
        st.write(f"**Cleaned Review:** {cleaned_review}")
        tokens = cleaned_review.split()
        review_embedding = get_avg_word2vec(tokens, word2vec_model, vector_size)
        review_embedding_scaled = scaler.transform([review_embedding])
        prediction = model.predict(review_embedding_scaled)
        predicted_label = encoder.inverse_transform(prediction)[0]

    st.write(f"The predicted label is: **{predicted_label}**")

    st.write("Class Mapping:")
    st.table({i: label for i, label in enumerate(class_names)}) 