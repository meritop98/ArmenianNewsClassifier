#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import joblib

# Load the MLP model weights
mlp = joblib.load('mlp_model_weights.pkl')

# Load the vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app
st.title("Text Classification App")

# User input
text_input = st.text_area("Enter text", "")

if st.button("Predict"):
    # Vectorize the input text
    text_vector = vectorizer.transform([text_input])

    # Make predictions using the loaded MLP model
    prediction = mlp.predict(text_vector)

    # Display the predicted category
    st.write("Predicted Category:", prediction[0])

