import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data for training
input_texts = ['What is your name?', 'How are you?', 'Who are you?']
target_texts = ['I am a chatbot.', 'I am doing well.', 'I am a friendly AI.']

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_texts + target_texts)

input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

input_sequences_padded = pad_sequences(input_sequences)
target_sequences_padded = pad_sequences(target_sequences)

# Define the model
embedding_dim = 16
units = 32

encoder_input = tf.keras.layers.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, embedding_dim)(encoder_input)
encoder, encoder_state = tf.keras.layers.LSTM(units, return_state=True)(encoder_embedding)

decoder_input = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, embedding_dim)(decoder_input)
decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_lstm(decoder_embedding, initial_state=encoder_state)

decoder_dense = tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Compile the model
model = tf.keras.models.Model([encoder_input, decoder_input], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([input_sequences_padded, target_sequences_padded[:, :-1]],
          target_sequences_padded[:, 1:],
          epochs=50,
          batch_size=1)

# Save the model
model.save('chatbot_model.py')

# Inference model for chatbot
encoder_model = tf.keras.models.Model(encoder_input, encoder_state)

decoder_state_input = tf.keras.layers.Input(shape=(units,))
decoder_outputs, decoder_state = decoder_lstm(
    decoder_embedding, initial_state=decoder_state_input)
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = tf.keras.models.Model(
    [decoder_input] + [decoder_state_input],
    [decoder_outputs] + [decoder_state])

# Function to generate response
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq_padded = pad_sequences(input_seq)
    state_value = encoder_model.predict(input_seq_padded)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h = decoder_model.predict([target_seq] + [state_value])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word[sampled_token_index]

        if sampled_word != '<end>':
            decoded_sentence += ' ' + sampled_word

        if sampled_word == '<end>' or len(decoded_sentence.split()) > 20:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        state_value = h

    return decoded_sentence.strip()

# Example usage
user_input = 'How are you?'
response = generate_response(user_input)
print(f'User: {user_input}')
print(f'Chatbot: {response}')