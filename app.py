import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
import json
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

app = Flask(__name__)

# Load the trained model
try:
    trained_model = tf.keras.models.load_model(
        r'C:\Users\user\OneDrive\Desktop\BSC PROJ\textualsentimentanalysis\trainmodel.keras'
    )
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file not found.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

# Load the tokenizer
try:
    with open(
        r'C:\Users\user\OneDrive\Desktop\BSC PROJ\textualsentimentanalysis\tokenizer.json', 'r'
    ) as f:
        tokenizer_config = json.load(f)
    tokenizer2 = tokenizer_from_json(tokenizer_config)
    print("Tokenizer loaded successfully.")
except FileNotFoundError:
    print("Tokenizer file not found.")
except Exception as e:
    print(f"An error occurred while loading the tokenizer: {e}")

# Initialize the stemmer
stemmer = PorterStemmer()

# Define the max_sequence_length (match your training setup)
max_sequence_length = 66  # Adjust based on training

def preprocess_text(text):
    stemmed_text = [stemmer.stem(word) for word in text.split()]
    print("Stemmed Text:", stemmed_text)
    sequence = tokenizer2.texts_to_sequences([' '.join(stemmed_text)])
    print("Tokenized Sequence:", sequence)
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    print("Padded Sequence:", padded_sequence)
    return padded_sequence

def predict_emotion(sentence):
    processed_text = preprocess_text(sentence)
    prediction = trained_model.predict(processed_text)
    print("Raw Prediction:", prediction)

    predicted_class = np.argmax(prediction, axis=1)
    print("Predicted Class:", predicted_class)

    possible_emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    emojis = {
        'sadness': '😢',
        'joy': '😊',
        'love': '❤️',
        'anger': '😠',
        'fear': '😨',
        'surprise': '😲'
    }

    predicted_emotion = possible_emotions[predicted_class[0]]
    predicted_emoji = emojis[predicted_emotion]

    return f"{predicted_emotion} {predicted_emoji}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'message' in request.form:
        message = request.form['message']
        print("Received message:", message)
        prediction = predict_emotion(message)
        return redirect(url_for('result', prediction=prediction))
    else:
        return "No message provided", 400

@app.route('/result/<prediction>')
def result(prediction):
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
