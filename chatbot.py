import json
import numpy as np
import pickle
import random
import pymongo
from datetime import datetime
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

# Kết nối tới MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['chatbot_database']
collection = db['user_feedback']

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Tải intents
with open('data.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_vector = bow(sentence)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list):
    tag = intents_list[0]['intent']
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

def save_unanswered(user_input):
    feedback_data = {
        "user_input": user_input,
        "predicted_tag": "none",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    collection.insert_one(feedback_data)

def chatbot_response(user_input):
    intents_list = predict_class(user_input)
    if len(intents_list) == 0:
        save_unanswered(user_input)
        return "Xin lỗi, tôi chưa hiểu câu hỏi này."
    else:
        return get_response(intents_list)
