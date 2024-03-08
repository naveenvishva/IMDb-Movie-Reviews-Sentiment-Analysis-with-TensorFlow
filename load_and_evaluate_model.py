import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the IMDb movie reviews dataset
imdb = keras.datasets.imdb
(_, _), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Preprocess the test data by padding the sequences
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=0, padding='post', maxlen=256)

# Load the saved model
loaded_model = keras.models.load_model('moviereview.h5')

# Use the loaded model for inference
predictions = loaded_model.predict(test_data)
predicted_labels = np.round(predictions).flatten()

# Create a confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

# Define class labels
class_labels = ['Negative', 'Positive']

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

# Set labels, title, and ticks
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(class_labels)
ax.yaxis.set_ticklabels(class_labels)

# Display the plot
plt.show()
