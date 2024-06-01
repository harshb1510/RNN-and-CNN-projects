import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, Input
from tensorflow.keras.optimizers import RMSprop

# Download the text data
filepath = tf.keras.utils.get_file("shakespeare.txt", 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read the text data
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# Slice the text
text = text[300000:800000]

# Create character-to-index and index-to-character mappings
characters = sorted(set(text))
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

# Prepare the data for training
SEQ_LENGTH = 40
STEP_SIZE = 3
sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i+SEQ_LENGTH])
    next_characters.append(text[i+SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

# Check dataset size
print(f"Number of sequences: {len(sentences)}")

# Build the model
model = Sequential()
model.add(Input(shape=(SEQ_LENGTH, len(characters))))
model.add(LSTM(128))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# Train the model
model.fit(x, y, epochs=10, batch_size=128)

# Save the model
model.save('textgenerator.model')
