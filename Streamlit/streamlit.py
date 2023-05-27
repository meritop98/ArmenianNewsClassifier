#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import joblib
import matplotlib.pyplot as plt

# Load the MLP model weights
mlp = joblib.load('mlp_model.pkl')

# Load the vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Load the model score
model_score = joblib.load('f1_score.pkl')

# Load the misclassification plot
misclassification_plot = 'misclassification_plot.jpg'

# App description
st.title("News Classification Model")

# video_file = open('vecteezy_world-breaking-news-digital-earth-hud-rotating-globe-rotating_6299370_777.mp4', 'rb')
# video_bytes = video_file.read()
# st.video(video_bytes)
plot="robot.jpg"
st.image(plot, use_column_width=True)



st.subheader("Problem")

st.write("<p style='font-size: 18px; line-height: 1.5;'>The goal of the project is to collect different types of news  from the Internet (through web scraping), process and obtain a model on this set of data, which will make it possible to predict what the given news is about.</p>",unsafe_allow_html=True)


st.markdown("---")
st.subheader("Data")
df = pd.read_csv('data_cleaned.csv')
st.dataframe(df)
st.write("Dimensions of the Data: {} rows, {} columns".format(df.shape[0], df.shape[1]))

plot="category_counts_plot.jpg"
st.markdown("---")
st.subheader("Category Counts")
st.image(plot, use_column_width=True)
# Example categories
st.subheader("Example Categories:")
st.write("<p style='font-size: 18px; line-height: 1.5;'>You can expect to see categories like:.</p>".format(model_score * 100), unsafe_allow_html=True)


col1, col2 = st.columns(2)

left_categories = ["0 - Կորոնավիրուս", "1 - Մշակույթ", "2 - Սպորտ"]
right_categories = ["3 - ՏՏ ոլորտ", "4 - Տնտեսություն", "5 - Քաղաքականություն"]

with col1:
    for category in left_categories:
        st.write("- " + category)

with col2:

    for category in right_categories:
        st.write("- " + category)
# User input
st.markdown("---")
st.subheader("Predict with the Model:")
text_input = st.text_area("Enter text", "")

if st.button("Predict"):
    if text_input.strip() != "":
        try:
            # Vectorize the input text
            text_vector = vectorizer.transform([text_input])

            # Make predictions using the loaded MLP model
            prediction = mlp.predict(text_vector)

            # Display the predicted category
            st.write("Predicted Category:", prediction[0])
        except Exception as e:
            st.error("An error occurred during prediction. Please try again.")
            st.error(str(e))
    else:
        st.warning("Please enter some text.")

st.markdown("---")
st.subheader("Model Information:")

# Larger text using CSS styling
st.write("<p style='font-size: 18px; line-height: 1.5;'>This classification model is based on a Multi-Layer Perceptron (MLP) neural network.</p>", unsafe_allow_html=True)
st.write("<p style='font-size: 18px; line-height: 1.5;'>It was trained on a dataset of armenian news articles and can predict the category of a given article.</p>", unsafe_allow_html=True)
st.write("<p style='font-size: 18px; line-height: 1.5;'>The model achieved an accuracy of {:.2f}% during training.</p>".format(model_score * 100), unsafe_allow_html=True)

st.markdown("---")
# Display the misclassification plot

st.subheader("Most Frequently Misclassified Categories")
st.image(misclassification_plot, use_column_width=True)


# In[ ]:




