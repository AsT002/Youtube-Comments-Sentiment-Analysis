import tensorflow as tf
import requests
from dotenv import load_dotenv
import os
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
import json

# Load the IMDb word index
word_index = imdb.get_word_index()

# Adjust the word index to match Keras's vocabulary and include special tokens
word_index = {k: (v + 1) for k, v in word_index.items()}
word_index["<UNK>"] = 0

# Load environment variables from the .env file
load_dotenv()

# Path to the trained sentiment analysis model
model_path = 'sentiment_analysis_model.keras'

# Check if the trained model file exists
if not os.path.isfile(model_path):
    print(f"Model ({model_path}) doesn't exist. Please run train_model.py first.")
    exit(1)

# Retrieve API key from environment variables
api_key = os.getenv('API_KEY')
video_id = input("Youtube Video ID: ")

# Load the trained sentiment analysis model
model = tf.keras.models.load_model(model_path)
print("Model loaded from", model_path)

# Parameters for text preprocessing
max_len = 256  # Maximum length of input sequences
vocab_size = 10000  # Vocabulary size used during model training

def get_youtube_comments(api_key, video_id):
    """
    Retrieve comments from a specified YouTube video.
    
    Args:
        api_key (str): The API key for YouTube Data API.
        video_id (str): The ID of the YouTube video.
        
    Returns:
        list: A list of comments from the video.
    """

    base_url = 'https://www.googleapis.com/youtube/v3/commentThreads'
    comments = []  # List to store comments
    next_page_token = None  # Token for paginating through results

    # Check if cache exists and load comments from cache
    cache_path = f"./cache/{video_id}.json"
    if os.path.isfile(cache_path):
        with open(cache_path, 'r') as json_file:
            comments = json.load(json_file)
            print("Retrieved comments from cache.")
            return comments

    # Fetch comments from YouTube API
    while True:
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'key': api_key,
            'textFormat': 'plainText',
            'pageToken': next_page_token,
            'maxResults': 100  # Fetch up to 100 comments per page
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Check for HTTP errors
            data = response.json()  # Parse response JSON
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            exit(1)

        # Extract comments from the response
        for item in data.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
            comments.append(comment)

        # Check for the next page of comments
        next_page_token = data.get('nextPageToken')
        if not next_page_token:
            break  # Exit loop if there are no more pages
    
    # Cache the comments
    os.makedirs("./cache", exist_ok=True)
    with open(cache_path, "w") as json_file:
        json.dump(comments, json_file)
    
    print(f"Comments for {video_id} have been cached.")
    return comments

def preprocess_data(data, max_len, vocab_size):
    """
    Preprocess the input data by padding sequences and ensuring indices are within vocabulary size.

    Args:
        data (list of list of int): The input data to preprocess.
        max_len (int): The maximum length for padding sequences.
        vocab_size (int): The maximum index for the vocabulary.

    Returns:
        numpy.ndarray: Padded sequences ready for model input.
    """
    # Ensure that indices are within the vocabulary size range
    data = [[min(index, vocab_size - 1) for index in sequence] for sequence in data]
    
    # Pad the sequences
    return pad_sequences(data, maxlen=max_len, padding='post')

# Get comments from the specified YouTube video
all_comments = get_youtube_comments(api_key, video_id)

# Convert comments to sequences
sequences = [[word_index.get(word, 0) for word in review.lower().split()] for review in all_comments]

# Preprocess the data
padded_sequences = preprocess_data(sequences, max_len, vocab_size)

# Predict sentiment
sample_predictions = model.predict(padded_sequences)
sample_predicted_classes = (sample_predictions > 0.5).astype(int)

# Calculate sentiment statistics
Positive = sum(1 for x in sample_predicted_classes if x[0] == 1)
Negative = len(sample_predicted_classes) - Positive
sentiment_sum = sum(x[0] for x in sample_predictions)

print(f"Positive sentiment comments: {Positive}")
print(f"Negative sentiment comments: {Negative}")
print(f"% of comments that were positive: {round((Positive / (Positive + Negative)) * 100, 2)}%")
print(f"% of comments that were negative: {round((Negative / (Positive + Negative)) * 100, 2)}%")
print(f"Overall sentiment: {sentiment_sum / (Positive + Negative)}")
