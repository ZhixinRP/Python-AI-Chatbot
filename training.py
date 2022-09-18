from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from nltk.stem import WordNetLemmatizer
import random
import json
import pickle
import numpy as np

import nltk
# Make Sure to Install all this packages
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


wnl = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(words)
words = [wnl.lemmatize(word.lower())
         for word in words if word not in ignore_letters]
words = sorted(set(words))
print(words)
classes = sorted(set(classes))

# Save the files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(words, open('classes.pkl', 'wb'))

# bag of words

training = []
output_empty = [0] * len(classes)

for documents in documents:
    bag = []
