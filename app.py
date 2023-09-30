import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the trained model
model = load_model("C:/Users/ajayi/OneDrive/Desktop/Sample_deployment/Sentiment_Analysis/model.h5")

# Load the saved tokenizer
with open('C:/Users/ajayi/OneDrive/Desktop/Sample_deployment/Sentiment_Analysis/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the preprocess_text function
def preprocess_text(text):
    # Tokenize the input text
    tokens = tokenizer.texts_to_sequences([text])
    
    # Pad the sequences to a fixed length (use the same sequence length as during training)
    #max_sequence_length = ...  # Replace with the sequence length used during training
    padded_tokens = pad_sequences(tokens, maxlen = 17)
    
    return padded_tokens[0]


st.title('Sentiment Analysis App')

# Create a text input widget for user input
user_input = st.text_area('Enter text for sentiment analysis', '')

# Create a button to trigger sentiment analysis
if st.button('Analyze Sentiment'):
    # Preprocess the user input (you should customize this based on your model's requirements)
    # For example, tokenize, pad sequences, and convert text to numerical data
    # Ensure that your preprocessing matches what was done during training

    # Preprocess the user input
    processed_input = preprocess_text(user_input)

    # Make sentiment prediction using the loaded model
    prediction = model.predict(np.array([processed_input]))

    # Map the prediction to sentiment labels
    # sentiment = 'Positive' if prediction[0] > 0.5 else 'Negative'
    # Calculate the average prediction value
    average_prediction = prediction.mean()

    # Make a decision based on the average_prediction value
    if average_prediction > 0.5:
        sentiment = 'Negative'
    else:
        sentiment = 'Positive'


    # Display the sentiment prediction
    st.write(f'Sentiment: {sentiment}')
