import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine

from natural_language_processing import roberta_perform_sentiment_analysis, get_embedding
from rephraser import run_rephrase_no_history

initial_conversation = [{"role": "system", "content": "Your job is to rephrase text without changing its meaning. Only rephrase the text, do not add any additional information. Your response should be the rephrased text ONLY."}]

def make_directory():
    now = datetime.now()
    folder_name = now.strftime("%Y-%m-%d-%H-%M-%S")
    directory = "output_data/" + folder_name
    os.mkdir(directory)
    return directory 

def save_initial_report(directory, data, input_file):
    with open(directory + "/initial_report.txt", "w") as file:
        llm = input("Enter the LLM model used: ")
        file.write("Initial Report\n\n")
        file.write(f"LLM Model: {llm}\n")
        file.write(f"Input File: {input_file}\n")
        file.write(f"Total Number of Samples: {len(data)}\n\n")
        for bin in range(1, 6):
            file.write(f"Number of Sentiment Bin {bin}: {len(data[data['bin_sentiment'] == bin])}\n")

def get_binned_text_data(directory):
    data = pd.read_csv(directory)
    return data

def perform_sentiment_analysis(text):
    polarity, _ = roberta_perform_sentiment_analysis(text)
    return polarity

def calculate_cosine_similarity(original_embedding, rephrased_embedding):
    return 1 - cosine(original_embedding.detach().numpy().flatten(), rephrased_embedding.detach().numpy().flatten())

def normalize_sentiment(sentiment):
    return (sentiment + 1) / 2  # Normalize from [-1, 1] to [0, 1]

def get_current_bin(normalized_sentiment):
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = [1, 2, 3, 4, 5]
    return pd.cut([normalized_sentiment], bins=bins, labels=labels, include_lowest=True)[0]

def create_dataframe(original_text, original_embedding, original_bin, original_user_score, product_id):
    df = pd.DataFrame(columns=["product_id", "text", "polarity", "cosine_similarity", "original_bin", "original_user_score", "current_bin"])
    original_polarity = perform_sentiment_analysis(original_text)
    normalized_sentiment = normalize_sentiment(original_polarity)
    current_bin = get_current_bin(normalized_sentiment)
    df.loc[0] = [product_id, original_text, original_polarity, 1.0, original_bin, original_user_score, current_bin]
    return df

def rephrase_and_analyze(text, original_embedding, iterations, original_bin, original_user_score, product_id):
    df = create_dataframe(text, original_embedding, original_bin, original_user_score, product_id)
    for i in tqdm(range(iterations)):
        rephrased_text = run_rephrase_no_history(text)
        rephrased_embedding = get_embedding(rephrased_text)
        cosine_similarity = calculate_cosine_similarity(original_embedding, rephrased_embedding)
        polarity = perform_sentiment_analysis(rephrased_text)
        normalized_sentiment = normalize_sentiment(polarity)
        current_bin = get_current_bin(normalized_sentiment)
        df.loc[i+1] = [product_id, rephrased_text, polarity, cosine_similarity, original_bin, original_user_score, current_bin]
        text = rephrased_text
    return df

def plot_metric(data, metric, bin_directory, index, color):
    plt.figure(figsize=(10, 6))
    
    if isinstance(data, pd.Series):
        y = data
        x = range(len(data))
    else:  # DataFrame
        y = data[metric]
        x = range(len(y))
    
    plt.scatter(x, y, label=f'{metric.replace("_", " ").title()} over Iterations', color=color, alpha=0.6)

    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * np.array(x) + intercept
    plt.plot(x, trend_line, label=f'Trend Line (y={slope:.4f}x + {intercept:.4f})', color='red')

    correlation_coefficient = np.corrcoef(x, y)[0, 1]
    plt.text(0, min(y), f"Correlation Coefficient (R): {correlation_coefficient:.4f}", fontsize=12)

    plt.xlabel("Iterations")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Number of Rephrases vs. {metric.replace('_', ' ').title()}")
    plt.legend()
    plt.savefig(f"{bin_directory}/{index}_{metric}.png")
    plt.close()

def process_binned_data(data, directory, num_iterations, rephrase_iterations):
    for bin in range(1, 6):
        bin_directory = os.path.join(directory, f"bin_{bin}")
        os.mkdir(bin_directory)
        bin_data = data[data['bin_sentiment'] == bin]
        
        for i, (_, row) in enumerate(bin_data.iterrows()):
            if i >= num_iterations:
                break
            text = row['text']
            product_id = row['product_id']  # Get the product ID
            print(f"Processing Bin {bin}, Sample {i+1}, Product ID {product_id}: {text[:50]}...")
            original_embedding = get_embedding(text)
            df = rephrase_and_analyze(text, original_embedding, rephrase_iterations, bin, row['user_score'], product_id)
            df.to_csv(f"{bin_directory}/{i}.csv", index=False)
            plot_metric(df, "polarity", bin_directory, i, 'blue')
            plot_metric(df["cosine_similarity"][1:], "cosine_similarity", bin_directory, i, 'green')

def main():
    rephrase_iterations = 50
    input_file = "input_data/data_samples/2024-07-11_00-16-13_SAMPLED_BY_SAMPLE_100_sequential_matched_product_data_subset.csv"
    directory = make_directory()

    all_data = get_binned_text_data(input_file)
    save_initial_report(directory, all_data, input_file)

    num_iterations = min(len(all_data[all_data['bin_sentiment'] == bin]) for bin in range(1, 6))

    process_binned_data(all_data, directory, num_iterations, rephrase_iterations)

if __name__ == '__main__':
    main()