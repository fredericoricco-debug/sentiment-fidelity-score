import pandas as pd
import random
from datetime import datetime
from collections import Counter
import numpy as np

def convert_raw_to_processed_data(raw_data_path, processed_data_path):
    # Load the data
    df = pd.read_csv(raw_data_path)

    # Select the required columns
    df_final = df[['Id', 'ProductId', 'Score', 'Text', 'Sentiment_ROBERTA']]   
    # Rename the columns
    df_final.columns = ['ID', 'product_id', 'user_score', 'text', 'Sentiment_ROBERTA']

    # Decouple the ROBERTA score and sentiment label (stored as a tuple)
    df_final['sentiment'] = df_final['Sentiment_ROBERTA'].apply(lambda x: x.split(',')[1].strip(')'))
    df_final['sentiment_score'] = df_final['Sentiment_ROBERTA'].apply(lambda x: x.split(',')[0].strip('('))

    # Drop the original ROBERTA column
    df_final.drop('Sentiment_ROBERTA', axis=1, inplace=True)

    # Save the new dataframe to a CSV file
    df_final.to_csv(processed_data_path, index=False)

    print(f"Data processing complete and saved to '{processed_data_path}'.")

def produce_binned_sentiment(processed_data_path):
    # Load the processed data
    df = pd.read_csv(processed_data_path)

    # Calculate the min and max for normalization
    min_score = df['sentiment_score'].min()
    max_score = df['sentiment_score'].max()

    # Normalize the sentiment scores using min-max normalization
    df['normalized_score'] = (df['sentiment_score'] - min_score) / (max_score - min_score)

    # Infinitesimmal value to ensure the min value is included in the first bin
    df['normalized_score'] = df['normalized_score'].apply(lambda x: x + 0.0001 if x == 0.0 else x)
    # Bin the normalized scores into 5 bins
    df['bin_sentiment'] = pd.cut(df['normalized_score'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])

    # Save the updated dataframe
    df.to_csv(processed_data_path, index=False)

    print(f"Sentiment scores binned and saved to '{processed_data_path}'.")

def produce_subset_by_product(data, save_file, num_products, random_sample, match_score):
    # Load the processed data
    df = pd.read_csv(data)

    # Create a new dataframe to store the final subset
    final_df = pd.DataFrame(columns=df.columns)

    # Get unique product IDs
    unique_products = df['product_id'].unique()

    if random_sample:
        np.random.shuffle(unique_products)
    else:
        # Calculate balance score for each product
        product_balance_scores = []
        for product_id in unique_products:
            product_rows = df[df['product_id'] == product_id]
            if match_score:
                product_rows = product_rows[product_rows['user_score'] == product_rows['bin_sentiment']]
            bin_counts = Counter(product_rows['bin_sentiment'])
            
            # Calculate entropy as a measure of balance
            total_samples = sum(bin_counts.values())
            if total_samples > 0:
                probabilities = [count / total_samples for count in bin_counts.values()]
                entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probabilities)
                
                # Penalize for missing bins
                missing_bins_penalty = (5 - len(bin_counts)) * 0.1
                
                balance_score = entropy - missing_bins_penalty
            else:
                balance_score = -np.inf
            
            product_balance_scores.append((product_id, balance_score, total_samples))
        
        # Sort products by balance score in descending order, then by total samples
        unique_products = [product for product, _, _ in sorted(product_balance_scores, 
                                                               key=lambda x: (x[1], x[2]), 
                                                               reverse=True)]

    # Counter for selected products
    selected_product_count = 0

    # Iterate through all products
    for product_id in unique_products:
        if selected_product_count >= num_products:
            break

        product_rows = df[df['product_id'] == product_id]
        
        # Apply match_score check if required
        if match_score:
            matching_rows = product_rows[product_rows['user_score'] == product_rows['bin_sentiment']]
            if not matching_rows.empty:
                final_df = pd.concat([final_df, matching_rows], ignore_index=True)
                selected_product_count += 1
        else:
            final_df = pd.concat([final_df, product_rows], ignore_index=True)
            selected_product_count += 1

    # Place the columns in the correct order
    final_df = final_df[['ID', 'product_id', 'bin_sentiment', 'user_score', 'text', 'sentiment', 'sentiment_score', 'normalized_score']]

    # Save the final dataframe to the save_file
    final_df.to_csv(save_file, index=False)
    
    print(f"Subset of data created and saved to '{save_file}'.")
    print(f"Total unique products: {len(final_df['product_id'].unique())}")
    print(f"Total samples: {len(final_df)}")
    print(f"Samples per bin:")
    print(final_df['bin_sentiment'].value_counts().sort_index())

    if len(final_df['product_id'].unique()) < num_products:
        print(f"\nWarning: Only {len(final_df['product_id'].unique())} unique products were found that met the criteria.")
        print("Consider relaxing the match_score condition or increasing the sample size.")

    # Print distribution for each product
    for product_id in final_df['product_id'].unique():
        product_distribution = final_df[final_df['product_id'] == product_id]['bin_sentiment'].value_counts().sort_index()
        print(f"\nProduct {product_id} distribution:")
        print(product_distribution)

def produce_subset_by_sample(data, save_file, num_samples, random_sample, match_score, force_product_id):
    # Load the processed data
    df = pd.read_csv(data)

    # Create a new dataframe to store the final subset
    final_df = pd.DataFrame(columns=df.columns)

    # Group the data by bin_sentiment
    grouped = df.groupby('bin_sentiment')

    # Dictionary to keep track of selected product_ids
    selected_product_ids = set()

    for bin_sentiment, group in grouped:
        samples_collected = 0
        
        # If random sampling is required, shuffle the group
        if random_sample:
            group = group.sample(frac=1).reset_index(drop=True)
        
        for _, row in group.iterrows():
            if samples_collected >= num_samples:
                break
            
            # Check if the user_score matches the bin_sentiment when match_score is True
            if match_score and int(row['user_score']) != int(row['bin_sentiment']):
                continue
            
            # Add the current row to the final dataframe
            final_df = pd.concat([final_df, pd.DataFrame([row])], ignore_index=True)
            samples_collected += 1
            
            # If force_product_id is True, add all rows with the same product_id
            if force_product_id and row['product_id'] not in selected_product_ids:
                selected_product_ids.add(row['product_id'])
                product_rows = df[df['product_id'] == row['product_id']]
                
                # Apply match_score check to product rows if required
                if match_score:
                    product_rows = product_rows[product_rows['user_score'] == product_rows['bin_sentiment']]
                
                final_df = pd.concat([final_df, product_rows], ignore_index=True)

    # Remove duplicates that might have been introduced when forcing product_ids
    final_df.drop_duplicates(inplace=True)

    # Truncate the final dataframe to the required number of samples
    # each bin should have num_samples samples
    final_df = final_df.groupby('bin_sentiment').head(num_samples)

    # Place the columns in the correct order
    final_df = final_df[['ID', 'product_id', 'bin_sentiment', 'user_score', 'text', 'sentiment', 'sentiment_score', 'normalized_score']]

    # Save the final dataframe to the save_file
    final_df.to_csv(save_file, index=False)

    print(f"Subset of data created and saved to '{save_file}'.")
    print(f"Total samples: {len(final_df)}")
    print(f"Samples per bin:")
    print(final_df['bin_sentiment'].value_counts().sort_index())

if __name__ == "__main__":
    # Define the paths to the raw and processed data
    raw_data_path = "input_data/datasets/base_sentiment.csv"
    processed_data_path = "input_data/datasets/formatted_data.csv"
    
    # Convert the raw data to formatted data
    convert_raw_to_processed_data(raw_data_path, processed_data_path)

    # Produce binned sentiment scores
    produce_binned_sentiment(processed_data_path)

    print("Data conversion and binning complete.")

    # Final parent path for save data files
    parent_path = "input_data/data_samples/"
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ################# USER SETTINGS #################
    # Define the number of samples to extract per bin
    num_samples = 2
    # Define if the data should be randomly sampled
    random_sample = False
    # Define if the user_score needs to match the bin_sentiment
    match_score = True
    # Force all product IDs to be gathered
    force_product_id = True
    #################################################

    # Ask the user if they want to sample by product or sample
    sample_by_product = input(f"The number of samples is: {num_samples}.\n\nWould you like to sample by product? (y/n): ") == 'y'

    # Check if sampling by product is required
    if sample_by_product:
        # Create filename
        filename = parent_path + date_time + f"_SAMPLED_BY_PRODUCT_{num_samples}_{'random' if random_sample else 'sequential'}_{'matched' if match_score else 'NOT_matched'}_data_subset.csv"
        # Produce a new dataset with a subset of the data
        produce_subset_by_product(processed_data_path, filename, num_samples, random_sample, match_score)
    else:
        # Create filename
        filename = parent_path + date_time + f"_SAMPLED_BY_SAMPLE_{num_samples}_{'random' if random_sample else 'sequential'}_{'matched' if match_score else 'NOT_matched'}_{'product' if force_product_id else 'NOT_product'}_data_subset.csv"
        # Produce a new dataset with a subset of the data
        produce_subset_by_sample(processed_data_path, filename, num_samples, random_sample, match_score, force_product_id)


