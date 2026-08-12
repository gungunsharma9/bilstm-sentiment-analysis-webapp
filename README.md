# Textual Sentiment Analysis Web Application using BiLSTM

## Project Overview
This project is a complete end-to-end machine learning solution for Textual Sentiment Analysis. It utilizes a Bidirectional Long Short-Term Memory (BiLSTM) neural network to classify text into six distinct emotional categories. To make the model accessible and interactive, a web application was developed using the Flask framework, allowing users to input text and instantly receive emotional sentiment predictions along with corresponding emojis.

## Emotions Detected
The model accurately classifies text into the following six emotions:
*   **Sadness** 😢
*   **Joy** 😊
*   **Love** ❤️
*   **Anger** 😠
*   **Fear** 😨
*   **Surprise** 😲

## Key Features
*   **Deep Learning Model:** Built using TensorFlow/Keras with an Embedding layer, Bidirectional LSTM, and Dense layers for accurate sequence processing and classification.
*   **Text Preprocessing Pipeline:** Utilizes NLTK's `PorterStemmer` to reduce words to their base forms, followed by tokenization and padding to ensure uniform input shapes (max sequence length of 66).
*   **Interactive UI:** A beautifully designed web interface with a glass-morphism effect, custom background, and intuitive design for seamless user experience.
*   **Flask Backend:** A lightweight and efficient server that loads the pre-trained `.keras` model and `tokenizer.json` to process real-time HTTP POST requests.

## Project Structure
```text
├── bilstm_emotion_classifier.ipynb   # BiLSTM emotion classifier for text sentiment analysis
├── app.py                        # Main Flask application script
├── trainmodel.keras              # Saved trained BiLSTM model (not included because it increases GitHub size limitations)
├── tokenizer.json                # Saved tokenizer dictionary
├── static/
│   └── bg.jpeg                   # Background image for the web app
└── templates/
    ├── index.html                # Home page UI with text input form
    └── result.html               # Result page displaying the predicted emotion

```

## Technologies Used

* Programming Language: Python 

* Machine Learning Framework: TensorFlow & Keras

* Natural Language Processing: NLTK (Stemming), Keras Tokenizer

* Data Analysis & EDA: Pandas, NumPy, Matplotlib, Seaborn

* Web Framework: Flask

* Frontend: HTML5, CSS3
