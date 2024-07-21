import random
import json
import pickle
import numpy as np
import language
import nltk

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')

intents = json.loads(open('intents.json', encoding='utf-8').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process the intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)

        # Apply appropriate stemming for each word based on language
        processed_words = []
        for word in word_list:
            word_lower = word.lower()
            if word_lower not in ignore_letters:
                lang = language.detect_language(word_lower)
                processed_word = language.stem_word(word_lower, lang)
                processed_words.append(processed_word)

        words.extend(processed_words)
        documents.append((processed_words, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Remove duplicates and sort
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

#in here we feed the NN with numerical values using bag of words
#Preprocessing the daata
for document in documents:
    bag = []
    word_patterns = document[0]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object) # Use dtype=object to handle heterogeneous lists

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
#to prevent overfitting
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.keras', hist)
print('Done')