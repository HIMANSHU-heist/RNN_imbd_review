
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


model = load_model("rnn_model_pred.keras")

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []
    for word in words:
        idx = word_index.get(word, 2)
        # 2 = unknown, 1 = start, 0 = padding - he reserved ahet
        encoded_review.append(idx + 3)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    score = prediction[0][0]
    
    
    if score >= 0.6:
        sentiment = '😊 Positive'
    elif score <= 0.4:
        sentiment = '😞 Negative'
    else:
        sentiment = '😐 Neutral (uncertain)'
    
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Score: {score:.4f}')
    st.progress(float(score))
