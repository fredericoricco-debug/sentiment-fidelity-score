from helpers import clean_text
from transformers import pipeline, BertTokenizer, BertModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch

# Check for MPS support and set the device accordingly
try:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
except:
    device = torch.device("cpu")
    print("MPS device not found. Using CPU instead.")

# TRANSFORMER
# Define your sentiment labels and their corresponding scores
labels = ['negative', 'neutral', 'positive']
mapping = {'negative': -1, 'neutral': 0, 'positive': 1}

# INITIALISING THE DIFFERENT CLASSIFIERS from Hugging Face
# Initialize the zero-shot classification pipeline
classifier_zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Initialise the roberta sentiment analysis model
# Load the tokenizer and model
roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment").to(device)

# Initialize the LiYuan amazon review sentiment analysis model
amazon_tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
amazon_model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis").to(device)

# Initialize the distilbert sentiment analysis model
distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
distilbert_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(device)

# VADER
# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# WORD EMBEDDING
# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Function to calculate softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Sentiment analysis function using zero-shot classification using transformer model
def zero_shot_perform_sentiment_analysis(text):
    # Clean and prepare the text for analysis
    text = clean_text(text)  # Assuming clean_text is defined elsewhere

    # Perform sentiment analysis on the cleaned text
    sentiments = classifier_zero_shot(text, candidate_labels=labels)
    cumulative_score = sum(score * mapping[label] for label, score in zip(sentiments["labels"], sentiments["scores"]))
    
    # Determine the sentiment category based on the final score
    final_score = cumulative_score / len(labels)  # Simplified approach
    closest_score = min(mapping.values(), key=lambda x: abs(x - final_score))
    category = [key for key, value in mapping.items() if value == closest_score][0]

    return final_score, category

# Sentiment analysis function using roberta model
def roberta_perform_sentiment_analysis(text):
    # Clean and prepare the text for analysis
    text = clean_text(text)  # Assuming clean_text is defined elsewhere

    # Perform sentiment analysis on the cleaned text
    inputs = roberta_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
    outputs = roberta_model(**inputs)
    logits = outputs.logits.cpu().detach().numpy()  # Move logits back to CPU
    scores = softmax(logits[0])
    final_score = sum(score * mapping[label] for label, score in zip(labels, scores))
    
    # Determine the sentiment category based on the final score
    closest_score = min(mapping.values(), key=lambda x: abs(x - final_score))
    category = [key for key, value in mapping.items() if value == closest_score][0]

    return final_score, category

# Sentiment analysis function using amazon review model
def amazon_perform_sentiment_analysis(text):
    # Clean and prepare the text for analysis
    text = clean_text(text)  # Assuming clean_text is defined elsewhere

    # Perform sentiment analysis on the cleaned text
    inputs = amazon_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
    outputs = amazon_model(**inputs)
    logits = outputs.logits.cpu().detach().numpy()  # Move logits back to CPU
    scores = softmax(logits[0])
    final_score = sum(score * mapping[label] for label, score in zip(labels, scores))
    
    # Determine the sentiment category based on the final score
    closest_score = min(mapping.values(), key=lambda x: abs(x - final_score))
    category = [key for key, value in mapping.items() if value == closest_score][0]

    return final_score, category

# Sentiment analysis function using distilbert model
def distilbert_perform_sentiment_analysis(text):
    # Clean and prepare the text for analysis
    text = clean_text(text)  # Assuming clean_text is defined elsewhere

    # Perform sentiment analysis on the cleaned text
    inputs = distilbert_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
    outputs = distilbert_model(**inputs)
    logits = outputs.logits.cpu().detach().numpy()  # Move logits back to CPU
    scores = softmax(logits[0])
    final_score = sum(score * mapping[label] for label, score in zip(labels, scores))
    
    # Determine the sentiment category based on the final score
    closest_score = min(mapping.values(), key=lambda x: abs(x - final_score))
    category = [key for key, value in mapping.items() if value == closest_score][0]

    return final_score, category

# Sentiment analysis function using VADER
def vader_perform_sentiment_analysis(text):
    # Clean and prepare the text for analysis
    text = clean_text(text)  # Assuming clean_text is defined elsewhere

    # Perform sentiment analysis on the cleaned text
    sentiment_scores = analyzer.polarity_scores(text)
    final_score = sentiment_scores["compound"]
    
    # Determine the sentiment category based on the final score
    if final_score < -0.10:
        category = "negative"
    elif final_score > 0.10:
        category = "positive"
    else:
        category = "neutral"

    return final_score, category

# Function to get sentence embedding
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
    outputs = model(**inputs)
    # Use the mean of all token embeddings as the sentence representation
    embeddings = outputs.last_hidden_state.mean(1).cpu()  # Move embeddings back to CPU
    return embeddings

def loop():
    text = input("Paste text to get data:")
    # Transformer sentiment analysis
    score, category = zero_shot_perform_sentiment_analysis(text)
    print(f"ZERO SHOT: The sentiment score is: {score}")
    print(f"ZERO SHOT: The sentiment category is: {category}")
    # RoBERTa sentiment analysis
    score, category = roberta_perform_sentiment_analysis(text)
    print(f"RoBERTa: The sentiment score is: {score}")
    print(f"RoBERTa: The sentiment category is: {category}")
    # Amazon Review sentiment analysis
    score, category = amazon_perform_sentiment_analysis(text)
    print(f"AMAZON: The sentiment score is: {score}")
    print(f"AMAZON: The sentiment category is: {category}")
    # DistilBERT sentiment analysis
    score, category = distilbert_perform_sentiment_analysis(text)
    print(f"DISTILBERT: The sentiment score is: {score}")
    print(f"DISTILBERT: The sentiment category is: {category}")
    # VADER sentiment analysis
    score, category = vader_perform_sentiment_analysis(text)
    print(f"VADER: The sentiment score is: {score}")
    print(f"VADER: The sentiment category is: {category}")
    # WORD EMBEDDING
    #embedding = get_embedding(text)
    #print(f"WORD EMBEDDING: The embedding is: {embedding}")

if __name__ == "__main__":
    while True:
        loop()
        print("\n")