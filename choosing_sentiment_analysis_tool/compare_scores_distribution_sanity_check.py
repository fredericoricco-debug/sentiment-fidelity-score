import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

# Increase font sizes by 1.5x
plt.rcParams.update({
    'font.size': 36,
    'axes.titlesize': 39,
    'axes.labelsize': 36,
    'xtick.labelsize': 33,
    'ytick.labelsize': 33,
    'legend.fontsize': 33,
})

# Load the sentiment data
base_sentiment = pd.read_csv("input_data/datasets/merged_base_sentiment.csv")

# Extract the sentiment scores from the tuples
def extract_sentiment_score(sentiment):
    return eval(sentiment)[0]

def calculate_jsd(dist1, dist2):
    # Ensure the distributions have the same number of bins
    bins = max(len(dist1), len(dist2))
    dist1 = np.pad(dist1, (0, bins - len(dist1)), 'constant')
    dist2 = np.pad(dist2, (0, bins - len(dist2)), 'constant')
    
    # Normalize the distributions
    dist1 = dist1 / np.sum(dist1)
    dist2 = dist2 / np.sum(dist2)
    
    return jensenshannon(dist1, dist2)

# Extract sentiment scores for VADER, Zero_shot
base_sentiment['Sentiment_VADER_Score'] = base_sentiment['Sentiment_VADER'].apply(extract_sentiment_score)
base_sentiment['Sentiment_Zero_Shot_Score'] = base_sentiment['Sentiment_Zero_Shot'].apply(extract_sentiment_score)
base_sentiment['Sentiment_ROBERTA_Score'] = base_sentiment['Sentiment_ROBERTA'].apply(extract_sentiment_score)
base_sentiment['Sentiment_Amazon_Score'] = base_sentiment['Sentiment_Amazon'].apply(extract_sentiment_score)
base_sentiment['Sentiment_DisitlBERT_Score'] = base_sentiment['Sentiment_DisitlBERT'].apply(extract_sentiment_score)

# Calculate min and max for normalization
min_vader = base_sentiment['Sentiment_VADER_Score'].min()
max_vader = base_sentiment['Sentiment_VADER_Score'].max()
min_zero_shot = base_sentiment['Sentiment_Zero_Shot_Score'].min()
max_zero_shot= base_sentiment['Sentiment_Zero_Shot_Score'].max()
min_roberta = base_sentiment['Sentiment_ROBERTA_Score'].min()
max_roberta = base_sentiment['Sentiment_ROBERTA_Score'].max()
min_amazon = base_sentiment['Sentiment_Amazon_Score'].min()
max_amazon = base_sentiment['Sentiment_Amazon_Score'].max()
min_distilbert = base_sentiment['Sentiment_DisitlBERT_Score'].min()
max_distilbert = base_sentiment['Sentiment_DisitlBERT_Score'].max()

# Normalize the sentiment scores using min-max normalization
base_sentiment['Normalized_Sentiment_VADER'] = (base_sentiment['Sentiment_VADER_Score'] - min_vader) / (max_vader - min_vader)
base_sentiment['Normalized_Sentiment_Zero_Shot'] = (base_sentiment['Sentiment_Zero_Shot_Score'] - min_zero_shot) / (max_zero_shot - min_zero_shot)
base_sentiment['Normalized_Sentiment_ROBERTA'] = (base_sentiment['Sentiment_ROBERTA_Score'] - min_roberta) / (max_roberta - min_roberta)
base_sentiment['Normalized_Sentiment_Amazon'] = (base_sentiment['Sentiment_Amazon_Score'] - min_amazon) / (max_amazon - min_amazon)
base_sentiment['Normalized_Sentiment_DisitlBERT'] = (base_sentiment['Sentiment_DisitlBERT_Score'] - min_distilbert) / (max_distilbert - min_distilbert)

# Bin the normalized scores into 5 bins
base_sentiment['Sentiment_VADER_Bin'] = pd.cut(base_sentiment['Normalized_Sentiment_VADER'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])
base_sentiment['Sentiment_Zero_Shot_Bin'] = pd.cut(base_sentiment['Normalized_Sentiment_Zero_Shot'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])
base_sentiment['Sentiment_ROBERTA_Bin'] = pd.cut(base_sentiment['Normalized_Sentiment_ROBERTA'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])
base_sentiment['Sentiment_Amazon_Bin'] = pd.cut(base_sentiment['Normalized_Sentiment_Amazon'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])
base_sentiment['Sentiment_DisitlBERT_Bin'] = pd.cut(base_sentiment['Normalized_Sentiment_DisitlBERT'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])

# Save the updated dataframe
base_sentiment.to_csv("input_data/datasets/base_sentiment_binned.csv", index=False)

# Calculate the distribution of actual star ratings
star_dist = base_sentiment['Score'].value_counts().sort_index().values

# Calculate JSD for each sentiment method
jsd_vader = calculate_jsd(star_dist, base_sentiment['Sentiment_VADER_Bin'].value_counts().sort_index().values)
jsd_zero_shot = calculate_jsd(star_dist, base_sentiment['Sentiment_Zero_Shot_Bin'].value_counts().sort_index().values)
jsd_roberta = calculate_jsd(star_dist, base_sentiment['Sentiment_ROBERTA_Bin'].value_counts().sort_index().values)
jsd_amazon = calculate_jsd(star_dist, base_sentiment['Sentiment_Amazon_Bin'].value_counts().sort_index().values)
jsd_distilbert = calculate_jsd(star_dist, base_sentiment['Sentiment_DisitlBERT_Bin'].value_counts().sort_index().values)

# Produces a dictionary of the proportion of each sentiment bin which originally had a given star rating
def get_proportions(df, sentiment_col, star_col):
    proportions = {}
    for sentiment_bin in range(1, 6):
        proportions[sentiment_bin] = {}
        for star_rating in range(1, 6):
            proportions[sentiment_bin][star_rating] = len(df[(df[sentiment_col] == sentiment_bin) & (df[star_col] == star_rating)]) / len(df[df[sentiment_col] == sentiment_bin])
    return proportions

# Get the proportions for each sentiment method
proportions_vader = get_proportions(base_sentiment, 'Sentiment_VADER_Bin', 'Score')
proportions_zero_shot = get_proportions(base_sentiment, 'Sentiment_Zero_Shot_Bin', 'Score')
proportions_roberta = get_proportions(base_sentiment, 'Sentiment_ROBERTA_Bin', 'Score')
proportions_amazon = get_proportions(base_sentiment, 'Sentiment_Amazon_Bin', 'Score')
proportions_distilbert = get_proportions(base_sentiment, 'Sentiment_DisitlBERT_Bin', 'Score')

fig, axes = plt.subplots(2, 3, figsize=(30, 15), sharey=True)

# Color map for star ratings
star_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']

# Actual star ratings
star_counts = base_sentiment['Score'].value_counts().sort_index()
bars = axes[0, 0].bar(star_counts.index, star_counts.values, color=star_colors)
axes[0, 0].text(0.5, 0.95, 'Actual Star Ratings', 
                horizontalalignment='center', verticalalignment='top', 
                transform=axes[0, 0].transAxes, fontsize=33, fontweight='bold')
axes[0, 0].set_ylabel('Count')

# Function to plot stacked bars for each sentiment method
def plot_sentiment_bins(ax, sentiment_col, proportions, title, jsd, title_position='top'):
    bin_counts = base_sentiment[sentiment_col].value_counts().sort_index()
    for bin in range(1, 6):
        sorted_props = sorted([(star, proportions[bin][star]) for star in range(1, 6)], 
                              key=lambda x: x[1], reverse=True)
        bottom = 0
        for star, prop in sorted_props:
            height = prop * bin_counts[bin]
            ax.bar(bin, height, bottom=bottom, color=star_colors[star-1], alpha=0.7)
            bottom += height
    ax.set_xticks(range(1, 6))
    
    # Add title inside the plot
    if title_position == 'top':
        ax.text(0.5, 0.95, f'{title}\n(JSD: {jsd:.4f})', 
                horizontalalignment='center', verticalalignment='top', 
                transform=ax.transAxes, fontsize=33, fontweight='bold')
    else:  # 'left'
        ax.text(0.05, 0.95, f'{title}\n(JSD: {jsd:.4f})', 
                horizontalalignment='left', verticalalignment='top', 
                transform=ax.transAxes, fontsize=28, fontweight='bold')

# Plot sentiment bins
plot_sentiment_bins(axes[0, 1], 'Sentiment_VADER_Bin', proportions_vader, 'VADER', jsd_vader)
plot_sentiment_bins(axes[0, 2], 'Sentiment_Zero_Shot_Bin', proportions_zero_shot, 'Zero Shot', jsd_zero_shot)
plot_sentiment_bins(axes[1, 0], 'Sentiment_ROBERTA_Bin', proportions_roberta, 'RoBERTa', jsd_roberta)
plot_sentiment_bins(axes[1, 1], 'Sentiment_Amazon_Bin', proportions_amazon, 'Amazon', jsd_amazon, title_position='left')
plot_sentiment_bins(axes[1, 2], 'Sentiment_DisitlBERT_Bin', proportions_distilbert, 'DistilBERT', jsd_distilbert)

# Remove x-axis labels for top row
for ax in axes[0]:
    ax.set_xticklabels([])

# Add x-axis label only to bottom row
for ax in axes[1]:
    ax.set_xlabel('Sentiment Bin')

# Add y-axis label only to left column
axes[0, 0].set_ylabel('Count')
axes[1, 0].set_ylabel('Count')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0)  # Decrease vertical space between rows

# Add a legend with reduced space between it and the graphs
handles = [plt.Rectangle((0,0),1,1, color=color) for color in star_colors]
fig.legend(handles, ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'], 
           loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Adjust the layout to leave less space for the legend
plt.savefig("choosing_sentiment_analysis_tool/sentiment_bins_distribution_with_jsd_colored_sorted_2x3.png", bbox_inches='tight')
plt.savefig("choosing_sentiment_analysis_tool/sentiment_bins_distribution_with_jsd_colored_sorted_2x3.pdf", bbox_inches='tight')
plt.show()