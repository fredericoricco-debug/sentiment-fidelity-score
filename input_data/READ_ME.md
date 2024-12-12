# Sentiment Fidelity Analysis - Data Set

This project analyzes how well sentiment is preserved when text is rephrased. It uses Amazon fine food reviews as a baseline dataset, processing them through multiple sentiment analysis models (ROBERTA, VADER, Zero-shot, Amazon, and DistilBERT) to establish ground truth sentiments. The system then uses different LLMs to rephrase these reviews and analyzes how the sentiment shifts across multiple rephrasings, measuring the "fidelity" of sentiment preservation using metrics certain metrics: RMSE, absolute change, and standard deviation. This helps understand how reliable AI-powered text rephrasing is at maintaining the original emotional content of text.

The dataset used was: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
It contains 568,454 reviews of fine foods from Amazon.

## Data Processing Pipeline

The project processes the Amazon reviews through two main scripts:

1. `get_dataset_sentiments.py`:
   - Takes the raw Amazon reviews and samples 5000 reviews for each star rating (1-5 stars)
   - Performs sentiment analysis using multiple models:
     - ROBERTA
     - VADER
     - Zero-shot
     - Amazon
     - DistilBERT
   - Outputs: `base_sentiment.csv`

2. `convert_dataset.py`:
   - Converts the analyzed data into a processed format
   - Features:
     - Normalizes sentiment scores
     - Bins sentiments into 5 categories
     - Provides two sampling methods:
       - By product
       - By sample count
     - Options for:
       - Random or sequential sampling
       - Matching user scores with sentiment bins
       - Forcing complete product ID sets

## Output Data Format

The processed data includes:
- ID
- product_id
- bin_sentiment (1-5)
- user_score
- text
- sentiment
- sentiment_score
- normalized_score

## Usage

1. First run `get_dataset_sentiments.py` to generate the base sentiment analysis
2. Then run `convert_dataset.py` to process and sample the data as needed

The script will prompt for sampling preferences and generate timestamped output files in the `input_data/data_samples/` directory.