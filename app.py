from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model("model.h5")

# Load the saved tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the preprocess_text function
def preprocess_text(text):
    # Tokenize the input text
    tokens = tokenizer.texts_to_sequences([text])
    
    # Pad the sequences to a fixed length (use the same sequence length as during training)
    #max_sequence_length = ...  # Replace with the sequence length used during training
    padded_tokens = pad_sequences(tokens, maxlen=100)
    
    return padded_tokens[0]

@app.route("/predict", methods=["GET", "POST"])
def predict():
    sentiment = ""

    if request.method == "POST":
        user_input = request.form["user_input"]

        # Preprocess the user input
        processed_input = preprocess_text(user_input)

        # Make sentiment prediction using the loaded model
        prediction = model.predict(np.array([processed_input]))

        # Map the prediction to sentiment labels
        average_prediction = prediction.mean()

        # Make a decision based on the average_prediction value
        if average_prediction > 0.5:
            sentiment = 'Negative'
        else:
            sentiment = 'Positive'

    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run()
