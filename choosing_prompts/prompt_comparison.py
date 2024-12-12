import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from rephraser import run_rephrase_no_history
from natural_language_processing import vader_perform_sentiment_analysis, transformer_perform_sentiment_analysis, get_embedding
from scipy.spatial.distance import cosine
from tqdm.auto import tqdm
from datetime import datetime

iterations_per_prompt = 33

# open the file prompt_candidates.csv
with open("choosing_prompts/prompt_candidates_new.csv") as f:
    # read the file
    prompts = pd.read_csv(f)

# retrieve initial texts
initial_texts_file = "input_data/testing_datasets/sample_data_9_samples_random_1_1_1.csv"
with open(initial_texts_file) as f:
    initial_texts = pd.read_csv(f)

# create new data frame with columns: id, prompt_id, prompt, prompt_theme, input_text_id, input_text, completion_text, input_text_sentiment, completion_text_sentiment, sentiment difference, cosine similarity
output_data = pd.DataFrame(columns=["id", "prompt_id", "prompt", "prompt_theme", "input_text_id", "input_text", "completion_text", "input_text_sentiment", "completion_text_sentiment", "sentiment_difference", "cosine_similarity"])

# iterate through the number of iterations
for _ in tqdm(range(iterations_per_prompt), desc="Iterations per Prompt"):
    # iterate through the prompts
    for i in tqdm(range(len(prompts)), desc="Prompts"):
        prompt_id = prompts["id"][i] # get the prompt id
        prompt = prompts["prompt"][i] # get the prompt
        prompt_theme = prompts["theme"][i] # get the prompt theme
        # iterate through the initial texts
        for j in tqdm(range(len(initial_texts)), desc="Initial Texts"):
            # Existing data
            input_text_id = j # get the input text id
            input_text = initial_texts["text"][j] # get the input text
            input_text_sentiment = initial_texts["sentiment"][j] # get the input text sentiment
            
            # Get the rephrased text
            completion_text = run_rephrase_no_history(input_text, prompt)
            
            # Perform sentiment analysis with both VADER and the transformer
            input_text_sentiment_vader, _ = vader_perform_sentiment_analysis(input_text)
            completion_text_sentiment_vader, _ = vader_perform_sentiment_analysis(completion_text)
            input_text_sentiment_transformer, _ = transformer_perform_sentiment_analysis(input_text)
            completion_text_sentiment_transformer, _ = transformer_perform_sentiment_analysis(completion_text)
            # Get the average sentiment
            input_text_sentiment = (input_text_sentiment_vader + input_text_sentiment_transformer) / 2
            completion_text_sentiment = (completion_text_sentiment_vader + completion_text_sentiment_transformer) / 2

            # Calculate the sentiment difference
            sentiment_difference = abs(input_text_sentiment - completion_text_sentiment)

            # Get the embeddings for the input and completion texts
            input_embedding = get_embedding(input_text)
            completion_embedding = get_embedding(completion_text)
            # Calculate the cosine similarity
            cosine_similarity = 1 - cosine(input_embedding.detach().numpy().flatten(), completion_embedding.detach().numpy().flatten())

            # Append the data to the output data
            output_data = output_data._append({"id": len(output_data), "prompt_id": prompt_id, "prompt": prompt, "prompt_theme": prompt_theme, "input_text_id": input_text_id, "input_text": input_text, "completion_text": completion_text, "input_text_sentiment": input_text_sentiment, "completion_text_sentiment": completion_text_sentiment, "sentiment_difference": sentiment_difference, "cosine_similarity": cosine_similarity}, ignore_index=True)


# Get the datetime
now = datetime.now()
output_file = f"choosing_prompts/prompt_comparison_data_{now}.csv"

# Save the output data
output_data.to_csv(output_file, index=False)
