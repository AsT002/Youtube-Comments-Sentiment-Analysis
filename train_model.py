from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Define parameters
max_length = 256  # Maximum length of input sequences (for padding)
vocab_size = 10000  # Vocabulary size (top most frequent words to consider)
embedding_dim = 100  # Dimension of the embedding vectors

# Load the IMDb dataset with a limited vocabulary size
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure consistent input length
x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post')

# Build the sentiment analysis model
model = Sequential([
    Embedding(vocab_size, embedding_dim),  # Embedding layer: converts word indices to dense vectors of size `embedding_dim`
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),  # Dense layer with ReLU activation: introduces non-linearity and learns features
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation: outputs probability for binary classification
])

# Compile the model
model.compile(
    optimizer='adam',  # Optimizer: Adam algorithm for gradient-based optimization
    loss='binary_crossentropy',  # Loss function: binary cross-entropy for binary classification problems
    metrics=['accuracy']  # Evaluation metric: accuracy
)

# Train the model
epochs = int(input("Enter number of epochs for training: "))  # User input for the number of epochs
history = model.fit(
    x_train, y_train,  # Training data and labels
    epochs=epochs,  # Number of epochs for training
    batch_size=64,  # Number of samples per gradient update
    validation_data=(x_test, y_test),  # Validation data to evaluate the model during training
    verbose=2  # Verbosity mode: 2 for detailed logging
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Save the trained model
save_path = "./sentiment_analysis_model.keras"  # Path to save the trained model
model.save(save_path)  # Save the model
print(f"Model saved at {save_path}.")