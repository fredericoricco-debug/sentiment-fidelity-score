import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_process_files(file_paths):
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        print(f"Loaded file: {file_path}")
        print(f"Shape: {df.shape}")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Combined shape: {combined_df.shape}")
    print(f"Unique prompt_ids: {combined_df['prompt_id'].nunique()}")

    return combined_df

def analyze_prompts(df):
    prompt_counts = df['prompt_id'].value_counts()
    print(f"Prompt counts:\n{prompt_counts}")

    grouped_df = df.groupby('prompt_id').agg(
        average_sentiment_diff=('sentiment_difference', 'mean'),
        std_sentiment_diff=('sentiment_difference', 'std'),
        average_cosine_similarity=('cosine_similarity', 'mean'),
        std_cosine_similarity=('cosine_similarity', 'std'),
        count=('sentiment_difference', 'count')
    ).reset_index()

    # Calculate standard error
    grouped_df['sem_sentiment_diff'] = grouped_df['std_sentiment_diff'] / np.sqrt(grouped_df['count'])
    grouped_df['sem_cosine_similarity'] = grouped_df['std_cosine_similarity'] / np.sqrt(grouped_df['count'])

    # Calculate 95% confidence intervals
    grouped_df['ci95_sentiment_diff'] = 1.96 * grouped_df['sem_sentiment_diff']
    grouped_df['ci95_cosine_similarity'] = 1.96 * grouped_df['sem_cosine_similarity']

    # Drop unnecessary columns
    grouped_df = grouped_df.drop(columns=['std_sentiment_diff', 'std_cosine_similarity', 'count'])

    print(f"Analyzed data shape: {grouped_df.shape}")
    print(grouped_df.head())

    return grouped_df

def reorganize_prompt_candidates(prompt_candidates_path):
    prompt_candidates = pd.read_csv(prompt_candidates_path)
    
    # Sort by theme and then by original prompt_id
    prompt_candidates_sorted = prompt_candidates.sort_values(['theme', 'prompt_id'])
    
    # Assign new IDs
    prompt_candidates_sorted['new_prompt_id'] = range(len(prompt_candidates_sorted))
    
    # Create a mapping from old to new IDs
    id_mapping = dict(zip(prompt_candidates_sorted['prompt_id'], prompt_candidates_sorted['new_prompt_id']))
    
    # Save the reorganized prompt candidates
    prompt_candidates_sorted.to_csv('choosing_prompts/prompt_candidates_organised.csv', index=False)
    
    return prompt_candidates_sorted, id_mapping

def plot_results(grouped_df, prompt_candidates_sorted, id_mapping):
    # Map the old prompt_ids to new_prompt_ids in grouped_df
    grouped_df['new_prompt_id'] = grouped_df['prompt_id'].map(id_mapping)
    
    # Merge grouped_df with reorganized prompt_candidates
    merged_df = pd.merge(grouped_df, prompt_candidates_sorted, left_on='new_prompt_id', right_on='new_prompt_id', how='left')

    fig, ax = plt.subplots(figsize=(30, 13))  # Slightly shorter figure

    # Bar plot for average sentiment difference with 95% confidence intervals
    sentiment_bars = ax.bar(merged_df['new_prompt_id'] - 0.2, merged_df['average_sentiment_diff'], width=0.4, 
                            yerr=merged_df['ci95_sentiment_diff'], capsize=5, label='Sentiment Difference', color='blue')

    # Bar plot for average cosine similarity with 95% confidence intervals
    cosine_bars = ax.bar(merged_df['new_prompt_id'] + 0.2, merged_df['average_cosine_similarity'], width=0.4, 
                         yerr=merged_df['ci95_cosine_similarity'], capsize=5, label='Cosine Similarity', color='orange')

    # Set transparency for 'sentiment' theme
    for i, theme in enumerate(merged_df['theme']):
        if theme == 'sentiment':
            sentiment_bars[i].set_alpha(0.3)
            cosine_bars[i].set_alpha(0.3)

    # Find the highest cosine similarity for each theme
    max_cosine_none = merged_df[merged_df['theme'] == 'none']['average_cosine_similarity'].idxmax()
    max_cosine_sentiment = merged_df[merged_df['theme'] == 'sentiment']['average_cosine_similarity'].idxmax()

    # Highlight the bars with the highest cosine similarity for each theme
    cosine_bars[max_cosine_none].set_color('red')
    cosine_bars[max_cosine_none].set_edgecolor('black')
    cosine_bars[max_cosine_none].set_linewidth(2)

    cosine_bars[max_cosine_sentiment].set_color('purple')
    cosine_bars[max_cosine_sentiment].set_edgecolor('black')
    cosine_bars[max_cosine_sentiment].set_linewidth(2)

    ax.set_xlabel('Prompt ID', fontsize=44)  # Increased from 40
    ax.set_ylabel('Values', fontsize=44)  # Increased from 40
    ax.legend(fontsize=36)  # Increased from 32

    plt.xticks(merged_df['new_prompt_id'], merged_df['new_prompt_id'], rotation=45, ha='right', fontsize=40)  # Increased from 36
    plt.yticks(fontsize=40)  # Increased from 36
    
    plt.tight_layout()
    plt.savefig('choosing_prompts/prompt_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('choosing_prompts/prompt_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    file_paths = [
        'choosing_prompts/prompt_comparison_data_2024-06-12 19:33:38.467920.csv',
        'choosing_prompts/prompt_comparison_data_2024-06-16 17:10:06.847615.csv'
    ]
    prompt_candidates_path = 'choosing_prompts/prompt_candidates.csv'

    # Reorganize prompt candidates
    prompt_candidates_sorted, id_mapping = reorganize_prompt_candidates(prompt_candidates_path)

    combined_df = load_and_process_files(file_paths)
    results = analyze_prompts(combined_df)
    plot_results(results, prompt_candidates_sorted, id_mapping)