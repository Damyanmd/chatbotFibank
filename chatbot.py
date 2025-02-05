import random
import json
import pickle
import numpy as np
import language
import nltk

from keras._tf_keras.keras.models import load_model

intents = json.loads(open('intents.json', encoding='utf-8').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    cleaned_words = []
    for word in sentence_words:
        lang = language.detect_language(word)
        cleaned_words.append(language.stem_word(word.lower(), lang))
    return cleaned_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    result_list = []
    for r in result:
        result_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return result_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Bot is running! Say Hello!")

while True:
    try:
        message = input("")
        ints = predict_class(message)
        res = get_response(ints, intents)
        print(res)
    except KeyboardInterrupt:
        print("Conversation is over")
    except Exception as e:
        print(f"Error: {str(e)}")