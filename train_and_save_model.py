import tensorflow as tf
from tensorflow import keras

# Load the IMDb movie reviews dataset
imdb = keras.datasets.imdb
(train_data, train_labels), (_, _) = imdb.load_data(num_words=10000)

# Preprocess the data by padding the sequences
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=0, padding='post', maxlen=256)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Embedding(10000, 16, input_length=256),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_data,
    train_labels,
    epochs=10,
    batch_size=512,
    validation_split=0.2
)

# Save the trained model
model.save('moviereview.h5')
