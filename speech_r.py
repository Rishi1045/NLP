import pandas as pd
import numpy as np
import re
import spacy
import speech_recognition as sr  # Import the speech recognition library
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the SpaCy model
nlp = spacy.load("en_core_web_md")

# Load the dataset
data = pd.read_csv('dataset.csv')

# Data Preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '{amount}', text)  # Replace digits with placeholder
    text = re.sub(r'rs|rupees|dollars', '{currency}', text)  # Replace currencies with placeholder
    text = re.sub(r'\b[a-zA-Z]+\b', '{receiver}', text)  # Replace receiver names with placeholder
    return text

data['cleaned_sentence'] = data['sentence'].apply(preprocess_text)

# Features and Labels
X = data['cleaned_sentence']
y = data['label']

# Convert text data to numerical features using embeddings
X_embeddings = []
for sentence in X:
    doc = nlp(sentence)
    X_embeddings.append(doc.vector)

# Convert list to numpy array
X_embeddings = np.vstack(X_embeddings)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Function to process new transaction requests
def process_transaction(input_sentence):
    # Check if the command is valid
    input_cleaned = preprocess_text(input_sentence)
    input_vector = nlp(input_cleaned).vector
    prediction = model.predict([input_vector])[0]
    
    # Check if valid transaction
    if prediction != 1:
        return "Invalid transaction."
    
    # Extract the amount and receiver name
    amount_match = re.search(r'(rs|rupees|dollars)?\s*(\d+)', input_sentence, re.IGNORECASE)  
    receiver_match = re.search(r'to\s+([a-zA-Z]+)', input_sentence)  # Extract receiver after 'to'
    
    if not amount_match or not receiver_match:
        return "Invalid transaction: Missing amount or receiver."

    # Extracting amount and receiver (numeric value only for amount)
    amount_str = amount_match.group(2)  # Extract numeric part of amount only
    receiver_str = receiver_match.group(1)  # This is the receiver's name

    return f"Amount = {amount_str}\nReceiver = {receiver_str}\nTransaction processed."

# Function for voice recognition
def listen_to_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for your transaction request...")
        audio = recognizer.listen(source)
        try:
            # Recognize the speech in the audio
            input_sentence = recognizer.recognize_google(audio)
            print(f"You said: {input_sentence}")
            return input_sentence
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return None

# Main loop for user input
while True:
    input_sentence = listen_to_voice()
    if input_sentence is None:
        continue  # Skip this loop if there was an error in recognition
    if input_sentence.lower() == 'exit':
        break
    result = process_transaction(input_sentence)
    print(result)
