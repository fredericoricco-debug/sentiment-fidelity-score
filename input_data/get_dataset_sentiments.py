import sys, os

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# imports
import pandas as pd
from natural_language_processing import roberta_perform_sentiment_analysis, amazon_perform_sentiment_analysis, distilbert_perform_sentiment_analysis, vader_perform_sentiment_analysis, zero_shot_perform_sentiment_analysis
from tqdm import tqdm

# Number of reviews to get for each star rating
number_of_reviews = 5000

# Open raw_amazon_reviews.csv and convert to a dataframe
raw_amazon_reviews = pd.read_csv("input_data/datasets/raw_amazon_reviews.csv")

# Initialize dataframes for each star rating
reviews_with_1_star = pd.DataFrame(columns=["Id", "ProductId", "Score", "Summary", "Text"])
reviews_with_2_star = pd.DataFrame(columns=["Id", "ProductId", "Score", "Summary", "Text"])
reviews_with_3_star = pd.DataFrame(columns=["Id", "ProductId", "Score", "Summary", "Text"])
reviews_with_4_star = pd.DataFrame(columns=["Id", "ProductId", "Score", "Summary", "Text"])
reviews_with_5_star = pd.DataFrame(columns=["Id", "ProductId", "Score", "Summary", "Text"])

# Create the dataframe which will store the base_sentiment
base_sentiment = pd.DataFrame(columns=["Id", "ProductId", "Score", "Summary", "Text", "Sentiment_ROBERTA"])

# Iterate through the raw amazon reviews, taking number_of_reviews reviews for each star rating
for i in range(len(raw_amazon_reviews)):
    score = raw_amazon_reviews.loc[i, "Score"]
    if score == 1 and len(reviews_with_1_star) < number_of_reviews:
        reviews_with_1_star = reviews_with_1_star._append(raw_amazon_reviews.loc[i], ignore_index=True)
    elif score == 2 and len(reviews_with_2_star) < number_of_reviews:
        reviews_with_2_star = reviews_with_2_star._append(raw_amazon_reviews.loc[i], ignore_index=True)
    elif score == 3 and len(reviews_with_3_star) < number_of_reviews:
        reviews_with_3_star = reviews_with_3_star._append(raw_amazon_reviews.loc[i], ignore_index=True)
    elif score == 4 and len(reviews_with_4_star) < number_of_reviews:
        reviews_with_4_star = reviews_with_4_star._append(raw_amazon_reviews.loc[i], ignore_index=True)
    elif score == 5 and len(reviews_with_5_star) < number_of_reviews:
        reviews_with_5_star = reviews_with_5_star._append(raw_amazon_reviews.loc[i], ignore_index=True)

# Print the length of each dataframe
print(f"1 star: {len(reviews_with_1_star)}")
print(f"2 star: {len(reviews_with_2_star)}")
print(f"3 star: {len(reviews_with_3_star)}")
print(f"4 star: {len(reviews_with_4_star)}")
print(f"5 star: {len(reviews_with_5_star)}")

# Iterate through the reviews and perform sentiment analysis, storing the results in the base_sentiment dataframe
for i in tqdm(range(len(reviews_with_1_star))):
    # Perform sentiment analysis using models
    sentiment_vader = vader_perform_sentiment_analysis(reviews_with_1_star.loc[i, "Text"])
    sentiment_zero_shot = zero_shot_perform_sentiment_analysis(reviews_with_1_star.loc[i, "Text"])
    sentiment_roberta = roberta_perform_sentiment_analysis(reviews_with_1_star.loc[i, "Text"])
    sentiment_amazon = amazon_perform_sentiment_analysis(reviews_with_1_star.loc[i, "Text"])
    sentiment_distilbert = distilbert_perform_sentiment_analysis(reviews_with_1_star.loc[i, "Text"])
    # Append the results to the base_sentiment dataframe
    base_sentiment = base_sentiment._append({
        "Id": reviews_with_1_star.loc[i, "Id"],
        "ProductId": reviews_with_1_star.loc[i, "ProductId"],
        "Score": reviews_with_1_star.loc[i, "Score"],
        "Summary": reviews_with_1_star.loc[i, "Summary"],
        "Text": reviews_with_1_star.loc[i, "Text"],
        "Sentiment_ROBERTA": sentiment_roberta,
    }, ignore_index=True)

for i in tqdm(range(len(reviews_with_2_star))):
    # Perform sentiment analysis using models
    sentiment_vader = vader_perform_sentiment_analysis(reviews_with_2_star.loc[i, "Text"])
    sentiment_zero_shot = zero_shot_perform_sentiment_analysis(reviews_with_2_star.loc[i, "Text"])
    sentiment_roberta = roberta_perform_sentiment_analysis(reviews_with_2_star.loc[i, "Text"])
    sentiment_amazon = amazon_perform_sentiment_analysis(reviews_with_2_star.loc[i, "Text"])
    sentiment_distilbert = distilbert_perform_sentiment_analysis(reviews_with_2_star.loc[i, "Text"])
    # Append the results to the base_sentiment dataframe
    base_sentiment = base_sentiment._append({
        "Id": reviews_with_2_star.loc[i, "Id"],
        "ProductId": reviews_with_2_star.loc[i, "ProductId"],
        "Score": reviews_with_2_star.loc[i, "Score"],
        "Summary": reviews_with_2_star.loc[i, "Summary"],
        "Text": reviews_with_2_star.loc[i, "Text"],
        "Sentiment_ROBERTA": sentiment_roberta,
    }, ignore_index=True)

for i in tqdm(range(len(reviews_with_3_star))):
    # Perform sentiment analysis using models
    sentiment_vader = vader_perform_sentiment_analysis(reviews_with_3_star.loc[i, "Text"])
    sentiment_zero_shot = zero_shot_perform_sentiment_analysis(reviews_with_3_star.loc[i, "Text"])
    sentiment_roberta = roberta_perform_sentiment_analysis(reviews_with_3_star.loc[i, "Text"])
    sentiment_amazon = amazon_perform_sentiment_analysis(reviews_with_3_star.loc[i, "Text"])
    sentiment_distilbert = distilbert_perform_sentiment_analysis(reviews_with_3_star.loc[i, "Text"])
    # Append the results to the base_sentiment dataframe
    base_sentiment = base_sentiment._append({
        "Id": reviews_with_3_star.loc[i, "Id"],
        "ProductId": reviews_with_3_star.loc[i, "ProductId"],
        "Score": reviews_with_3_star.loc[i, "Score"],
        "Summary": reviews_with_3_star.loc[i, "Summary"],
        "Text": reviews_with_3_star.loc[i, "Text"],
        "Sentiment_ROBERTA": sentiment_roberta,
    }, ignore_index=True)


for i in tqdm(range(len(reviews_with_4_star))):
    # Perform sentiment analysis using models
    sentiment_vader = vader_perform_sentiment_analysis(reviews_with_4_star.loc[i, "Text"])
    sentiment_zero_shot = zero_shot_perform_sentiment_analysis(reviews_with_4_star.loc[i, "Text"])
    sentiment_roberta = roberta_perform_sentiment_analysis(reviews_with_4_star.loc[i, "Text"])
    sentiment_amazon = amazon_perform_sentiment_analysis(reviews_with_4_star.loc[i, "Text"])
    sentiment_distilbert = distilbert_perform_sentiment_analysis(reviews_with_4_star.loc[i, "Text"])
    # Append the results to the base_sentiment dataframe
    base_sentiment = base_sentiment._append({
        "Id": reviews_with_4_star.loc[i, "Id"],
        "ProductId": reviews_with_4_star.loc[i, "ProductId"],
        "Score": reviews_with_4_star.loc[i, "Score"],
        "Summary": reviews_with_4_star.loc[i, "Summary"],
        "Text": reviews_with_4_star.loc[i, "Text"],
        "Sentiment_ROBERTA": sentiment_roberta,
    }, ignore_index=True)

for i in tqdm(range(len(reviews_with_5_star))):
    # Perform sentiment analysis using models
    sentiment_vader = vader_perform_sentiment_analysis(reviews_with_5_star.loc[i, "Text"])
    sentiment_zero_shot = zero_shot_perform_sentiment_analysis(reviews_with_5_star.loc[i, "Text"])
    sentiment_roberta = roberta_perform_sentiment_analysis(reviews_with_5_star.loc[i, "Text"])
    sentiment_amazon = amazon_perform_sentiment_analysis(reviews_with_5_star.loc[i, "Text"])
    sentiment_distilbert = distilbert_perform_sentiment_analysis(reviews_with_5_star.loc[i, "Text"])
    # Append the results to the base_sentiment dataframe
    base_sentiment = base_sentiment._append({
        "Id": reviews_with_5_star.loc[i, "Id"],
        "ProductId": reviews_with_5_star.loc[i, "ProductId"],
        "Score": reviews_with_5_star.loc[i, "Score"],
        "Summary": reviews_with_5_star.loc[i, "Summary"],
        "Text": reviews_with_5_star.loc[i, "Text"],
        "Sentiment_ROBERTA": sentiment_roberta,
    }, ignore_index=True)

# Save the base_sentiment dataframe to a csv file
base_sentiment.to_csv("input_data/datasets/base_sentiment.csv", index=False)