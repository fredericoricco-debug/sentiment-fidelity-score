import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from collections import defaultdict, Counter
import imageio
from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
##########################################################################################################################
def get_model_name():
    return input("Enter the name of the model used: ")
######################################## OLD NEGATIVE POSITIVE SENTIMENT ANALYSIS ########################################
def load_csv_files_negative_positive(folder_path):
    csv_files = {'negative': [], 'neutral': [], 'positive': []}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                sentiment = os.path.basename(root)
                if sentiment in csv_files:
                    csv_files[sentiment].append(os.path.join(root, file))
    return csv_files

def normalize_series(series):
    return (series - series.min()) / (series.max() - series.min())

def calculate_trend_negative_positive(csv_files):
    trends = {'negative': [], 'neutral': [], 'positive': []}

    for sentiment, files in csv_files.items():
        all_dfs = []
        for file in sorted(files):
            df = pd.read_csv(file)
            df['normalized_polarity'] = normalize_series(df['polarity'])
            all_dfs.append(df)
        
        # Ensure all dataframes have the same number of rows
        min_rows = min(df.shape[0] for df in all_dfs)
        all_dfs = [df.iloc[:min_rows] for df in all_dfs]
        
        # Calculate average normalized polarity across all files
        avg_normalized_polarity = pd.concat([df['normalized_polarity'] for df in all_dfs], axis=1).mean(axis=1)
        
        trends[sentiment] = pd.DataFrame({
            'timestep': range(len(avg_normalized_polarity)),
            'avg_normalized_polarity': avg_normalized_polarity
        })
    
    return trends

def plot_boxplot_trends_negative_positive(trend_dfs, output_folder, num_bins=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for sentiment, df in trend_dfs.items():
        if df.empty:
            print(f"No data for {sentiment} sentiment.")
            continue

        # Create time bins
        df['time_bin'] = pd.cut(df['timestep'], bins=num_bins)
        
        X = np.arange(num_bins).reshape(-1, 1)
        y = df.groupby('time_bin')['avg_normalized_polarity'].mean().values
        
        model = LinearRegression().fit(X, y)
        trendline = model.predict(X)
        r_coefficient = np.corrcoef(X.flatten(), y)[0, 1]
        
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='time_bin', y='avg_normalized_polarity', data=df)
        plt.plot(X, trendline, color='red', label='Trend Line')
        plt.title(f'Trend of Average Normalized Polarity Distribution for {sentiment.capitalize()} Sentiment\nR Coefficient: {r_coefficient:.2f}')
        plt.xlabel('Time Bins')
        plt.ylabel('Average Normalized Polarity')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{sentiment}_trend_boxplot.png'))
        plt.close()

def calculate_avg_trend_negative_positive(csv_files):
    trends = {'negative': [], 'neutral': [], 'positive': []}

    for sentiment, files in csv_files.items():
        all_dfs = []
        for file in sorted(files):
            df = pd.read_csv(file)
            df['normalized_polarity'] = normalize_series(df['polarity'])
            all_dfs.append(df)
        
        # Ensure all dataframes have the same number of rows
        min_rows = min(df.shape[0] for df in all_dfs)
        all_dfs = [df.iloc[:min_rows] for df in all_dfs]
        
        # Calculate average normalized polarity across all files
        avg_normalized_polarity = pd.concat([df['normalized_polarity'] for df in all_dfs], axis=1).mean(axis=1)
        
        trends[sentiment] = pd.DataFrame({
            'timestep': range(len(avg_normalized_polarity)),
            'avg_normalized_polarity': avg_normalized_polarity
        })
    
    return trends

def plot_scatter_trends_negative_positive(trend_dfs, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for sentiment, df in trend_dfs.items():
        if df.empty:
            print(f"No data for {sentiment} sentiment.")
            continue

        X = df['timestep'].values.reshape(-1, 1)
        y = df['avg_normalized_polarity'].values
        
        model = LinearRegression().fit(X, y)
        trendline = model.predict(X)
        r_coefficient = np.corrcoef(df['timestep'], df['avg_normalized_polarity'])[0, 1]
        
        plt.figure(figsize=(15, 8))
        plt.scatter(df['timestep'], df['avg_normalized_polarity'], marker='o', label='Average Normalized Polarity')
        plt.plot(df['timestep'], trendline, color='red', label='Trend Line')
        plt.title(f'Trend of Average Normalized Polarity for {sentiment.capitalize()} Sentiment\nR Coefficient: {r_coefficient:.2f}')
        plt.xlabel('Timestep')
        plt.ylabel('Average Normalized Polarity')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{sentiment}_trend_scatter.png'))
        plt.close()
##########################################################################################################################
######################################## SINGLE BIN ANALYSIS FUNCTIONS ###################################################
def load_csv_files_single_bin(folder_path):
    csv_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(folder_path, file))
    return sorted(csv_files, key=lambda x: int(os.path.basename(x).split('.')[0]))

def normalize_series(series):
    return (series - series.min()) / (series.max() - series.min())

def calculate_trend_single_bin(csv_files):
    # Read and normalize all CSV files
    normalized_dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['normalized_polarity'] = normalize_series(df['polarity'])
        normalized_dfs.append(df)
    
    # Ensure all dataframes have the same number of rows
    min_rows = min(df.shape[0] for df in normalized_dfs)
    normalized_dfs = [df.iloc[:min_rows] for df in normalized_dfs]
    
    # Calculate the average normalized polarity across all files for each timestep
    average_normalized_polarities = pd.concat([df['normalized_polarity'] for df in normalized_dfs], axis=1).mean(axis=1)
    
    # Create a dataframe with timesteps and average normalized polarities
    trend_df = pd.DataFrame({
        'timestep': range(len(average_normalized_polarities)),
        'average_normalized_polarity': average_normalized_polarities
    })
    
    return trend_df

def plot_scatter_trend_single_bin(trend_df, output_folder, bin_number, model_name, font_size=28):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plt.figure(figsize=(15, 8))
    
    # Create color gradient
    colors = LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    color = colors((int(bin_number) - 1) / 4)  # Normalize to [0, 1]
    
    plt.scatter(trend_df['timestep'], trend_df['average_normalized_polarity'], 
                marker='o', label='Average Normalized Polarity', color=color,
                edgecolors='black', linewidth=0.5, s=45)
    
    # Fit trendline
    X = trend_df['timestep'].values.reshape(-1, 1)
    y = trend_df['average_normalized_polarity'].values
    model = LinearRegression().fit(X, y)
    trendline = model.predict(X)
    r_coefficient = np.corrcoef(trend_df['timestep'], trend_df['average_normalized_polarity'])[0, 1]
    
    plt.plot(trend_df['timestep'], trendline, color='black', label='Trend Line', linewidth=2)
    
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('Average Normalized Polarity', fontsize=font_size)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    plt.tick_params(axis='both', which='major', labelsize=font_size-2)
    
    # Add legend with R coefficient
    plt.legend(fontsize=font_size-2, title=f'R = {r_coefficient:.2f}', title_fontsize=font_size-2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'bin_{bin_number}_trend_scatter.png'))
    plt.savefig(os.path.join(output_folder, f'bin_{bin_number}_trend_scatter.pdf'))
    plt.close()

def plot_boxplot_trend_single_bin(normalized_dfs, output_folder, bin_number, num_bins, model_name, font_size=28):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Combine all normalized dataframes
    combined_df = pd.concat(normalized_dfs, axis=0, ignore_index=True)
    
    # Create time bins
    combined_df['time_bin'] = pd.cut(combined_df.index % normalized_dfs[0].shape[0], bins=num_bins)
    
    plt.figure(figsize=(15, 8))
    
    # Create color gradient
    colors = LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    color = colors((int(bin_number) - 1) / 4)  # Normalize to [0, 1]
    
    sns.boxplot(x='time_bin', y='normalized_polarity', data=combined_df, color=color)
    
    plt.xlabel('Iterations', fontsize=font_size)
    plt.ylabel('Normalized Polarity', fontsize=font_size)
    plt.ylim(0 - 0.05, 1 + 0.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize x-axis labels to show 0 to 50
    x_ticks = range(num_bins)
    x_labels = [f"{int(i * 50 / (num_bins - 1))}" for i in range(num_bins)]
    plt.xticks(x_ticks, x_labels, rotation=45, fontsize=font_size-15)
    
    plt.tick_params(axis='y', which='major', labelsize=font_size-15)
    
    # Add R-Coefficient text box
    r_coefficient = input(f"Enter the R-Coefficient for Bin {bin_number}: ")
    if int(bin_number) == 1:
        box_position = (0.05, 0.95)  # Top left
        va = 'top'
    elif int(bin_number) == 5:
        box_position = (0.05, 0.05)  # Bottom left
        va = 'bottom'
    else:
        box_position = (0.05, 0.95)  # Top left for other bins
        va = 'top'
    
    plt.text(box_position[0], box_position[1], f'R-Coefficient: {r_coefficient}',
             transform=plt.gca().transAxes, fontsize=font_size-4,
             verticalalignment=va, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'bin_{bin_number}_trend_boxplot.png'))
    plt.savefig(os.path.join(output_folder, f'bin_{bin_number}_trend_boxplot.pdf'))
    plt.close()

def plot_violin_trend_single_bin(normalized_dfs, output_folder, bin_number):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Combine all normalized dataframes
    combined_df = pd.concat(normalized_dfs, axis=1)
    combined_df.columns = [f'file_{i}' for i in range(len(normalized_dfs))]
    
    # Melt the dataframe to long format
    melted_df = combined_df.reset_index().melt(id_vars='index', var_name='file', value_name='normalized_polarity')
    melted_df['iteration'] = melted_df['index']
    
    # Create the violin plot
    plt.figure(figsize=(25, 15))
    sns.violinplot(x='iteration', y='normalized_polarity', data=melted_df, cut=0, scale='width', palette='viridis')
    plt.title(f'Violin Plot of Normalized Polarity Distribution for Bin {bin_number}')
    # Set y-axis limits to -0.5 and 1.5
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Polarity')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'bin_{bin_number}_trend_violin.png'))
    plt.close()
##########################################################################################################################
######################################## SENTIMENT BINS ANALYSIS FUNCTIONS ##############################################
def plot_scatter_trends_all_bins(parent_folder, output_folder, model_name, font_size=12):
    bin_folders = sorted([f for f in os.listdir(parent_folder) if f.startswith('bin_') and os.path.isdir(os.path.join(parent_folder, f))])
    
    os.makedirs(output_folder, exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    
    colors = LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    
    legend_handles = []
    legend_labels = []
    
    for bin_folder in bin_folders:
        bin_path = os.path.join(parent_folder, bin_folder)
        csv_files = load_csv_files_single_bin(bin_path)
        trend_df = calculate_trend_single_bin(csv_files)
        bin_number = int(bin_folder.split('_')[1])
        
        color = colors((bin_number - 1) / 4)  # Normalize to [0, 1]
        scatter = plt.scatter(trend_df['timestep'], trend_df['average_normalized_polarity'], 
                    marker='o', color=color, alpha=0.6, edgecolors='black', linewidth=0.5, s=45)
        
        # Fit trendline
        X = trend_df['timestep'].values.reshape(-1, 1)
        y = trend_df['average_normalized_polarity'].values
        model = LinearRegression().fit(X, y)
        trendline = model.predict(X)
        
        line, = plt.plot(trend_df['timestep'], trendline, color=color, linestyle='--', linewidth=2)
        
        # Add to legend lists
        legend_handles.append(scatter)
        legend_labels.append(f'Bin {bin_number}')
    
    plt.title(f'Trend of Average Normalized Polarity for All Bins\nModel: {model_name}', fontsize=font_size+2)
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('Average Normalized Polarity', fontsize=font_size)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    plt.tick_params(axis='both', which='major', labelsize=font_size-2)
    
    # Add legend in the correct order
    plt.legend(legend_handles, legend_labels, fontsize=font_size-2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'all_bins_trend_scatter.png'))
    plt.close()

def plot_boxplot_trends_all_bins(parent_folder, output_folder, num_bins, model_name, font_size=12):
    bin_folders = sorted([f for f in os.listdir(parent_folder) if f.startswith('bin_') and os.path.isdir(os.path.join(parent_folder, f))])
    
    os.makedirs(output_folder, exist_ok=True)
    
    colors = LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    
    for bin_folder in bin_folders:
        bin_path = os.path.join(parent_folder, bin_folder)
        csv_files = load_csv_files_single_bin(bin_path)
        
        normalized_dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            df['normalized_polarity'] = normalize_series(df['polarity'])
            normalized_dfs.append(df[['normalized_polarity']])
        
        bin_number = int(bin_folder.split('_')[1])
        color = colors((bin_number - 1) / 4)  # Normalize to [0, 1]
        
        # Combine all normalized dataframes
        combined_df = pd.concat(normalized_dfs, axis=0, ignore_index=True)
        
        # Create time bins
        combined_df['time_bin'] = pd.cut(combined_df.index % normalized_dfs[0].shape[0], bins=num_bins)
        
        plt.figure(figsize=(15, 8))
        
        sns.boxplot(x='time_bin', y='normalized_polarity', data=combined_df, color=color)
        
        plt.title(f'Trend of Normalized Polarity Distribution for Bin {bin_number}\nModel: {model_name}', 
                  fontsize=font_size+2)
        plt.xlabel('Time Bins', fontsize=font_size)
        plt.ylabel('Normalized Polarity', fontsize=font_size)
        plt.ylim(0 - 0.05, 1+0.05)
        plt.xticks(rotation=45)
        plt.tick_params(axis='both', which='major', labelsize=font_size-2)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'bin_{bin_number}_trend_boxplot.png'))
        plt.close()

    print(f"Box plots for all bins saved in {output_folder}")
##########################################################################################################################
######################################## PRODUCT SCORE DISTRIBUTION FUNCTIONS ############################################
def perform_product_analysis(output_data_folder, specific_product_id=None):
    print(f"Starting product analysis for folder: {output_data_folder}")
    print(f"Specific product ID: {specific_product_id}")

    # Create the product_analysis folder
    analysis_folder = os.path.join(output_data_folder.replace('output_data', 'output_graphs'), 'product_analysis')
    os.makedirs(analysis_folder, exist_ok=True)
    print(f"Analysis folder created: {analysis_folder}")

    # Dictionary to store data for each product
    product_data = defaultdict(lambda: defaultdict(list))
    product_file_count = Counter()

    # Iterate through all bin folders
    bin_folders = [f for f in os.listdir(output_data_folder) if os.path.isdir(os.path.join(output_data_folder, f))]
    print(f"Found {len(bin_folders)} bin folders")

    for bin_folder in bin_folders:
        bin_path = os.path.join(output_data_folder, bin_folder)
        print(f"\nProcessing bin folder: {bin_folder}")
        
        csv_files = [f for f in os.listdir(bin_path) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files in {bin_folder}")

        for file in csv_files:
            file_path = os.path.join(bin_path, file)
            df = pd.read_csv(file_path)
            
            product_id = df['product_id'].iloc[0]
            if specific_product_id and str(product_id) != str(specific_product_id):
                continue

            product_file_count[product_id] += 1

            for iteration, row in df.iterrows():
                product_data[product_id][iteration].append(row['current_bin'])
            
            print(f"Processed file: {file}, Product ID: {product_id}, Iterations: {len(df)}")

    print(f"\nTotal products found: {len(product_data)}")

    # Create graphs for each product
    for product_id, iterations in tqdm(product_data.items()):
        product_folder = os.path.join(analysis_folder, f'product_{product_id}')
        os.makedirs(product_folder, exist_ok=True)
        
        # Find the maximum frequency across all iterations
        max_frequency = max(max(Counter(bins).values()) for bins in iterations.values())
        
        images = []
        for iteration, bins in sorted(iterations.items()):
            plt.figure(figsize=(10, 6))
            plt.hist(bins, bins=5, range=(1, 6), align='left', rwidth=0.8)
            plt.title(f'Product {product_id} - Iteration {iteration}\nAverage Score: {sum(bins)/len(bins):.2f}')
            plt.xlabel('Current Bin')
            plt.ylabel('Frequency')
            plt.xticks(range(1, 6))
            plt.ylim(0, max_frequency + 1)  # Set consistent y-axis limit
            
            # Save the plot
            plot_path = os.path.join(product_folder, f'iteration_{iteration}.png')
            plt.savefig(plot_path)
            images.append(imageio.imread(plot_path))
            plt.close()
        
        # Create GIF
        gif_path = os.path.join(product_folder, f'product_{product_id}_evolution.gif')
        imageio.mimsave(gif_path, images, duration=0.5)

    # Generate and save the report
    report_path = os.path.join(analysis_folder, 'product_file_count_report.txt')
    with open(report_path, 'w') as report_file:
        report_file.write("Product File Count Report\n")
        report_file.write("=========================\n\n")
        report_file.write("Product ID | File Count\n")
        report_file.write("------------|------------\n")
        for product_id, count in sorted(product_file_count.items(), key=lambda x: x[1], reverse=True):
            report_file.write(f"{product_id:10} | {count:12}\n")

    print(f"Product analysis completed. Results saved in {analysis_folder}")
    print(f"Product file count report saved as {report_path}")
##########################################################################################################################
############################################## RESULT ANALYSIS FUNCTION ##################################################
def perform_result_analysis(parent_folder, output_folder, model_name):
    results = []
    bin_folders = [f for f in os.listdir(parent_folder) if f.startswith('bin_') and os.path.isdir(os.path.join(parent_folder, f))]

    for bin_folder in bin_folders:
        bin_path = os.path.join(parent_folder, bin_folder)
        csv_files = load_csv_files_single_bin(bin_path)
        bin_number = int(bin_folder.split('_')[1])

        all_polarities = []
        for file in csv_files:
            df = pd.read_csv(file)
            all_polarities.append(df['polarity'])

        all_polarities = pd.concat(all_polarities, axis=1)

        initial_polarities = all_polarities.iloc[0]
        
        for iteration in range(len(all_polarities)):
            current_polarities = all_polarities.iloc[iteration]
            
            avg_initial_sentiment = initial_polarities.mean()
            avg_current_sentiment = current_polarities.mean()
            sentiment_change = avg_current_sentiment - avg_initial_sentiment
            absolute_change = abs(sentiment_change)
            
            std_dev = current_polarities.std()
            confidence_interval = 1.96 * std_dev / np.sqrt(len(current_polarities))
            
            mae = np.mean(np.abs(current_polarities - initial_polarities))
            rmse = np.sqrt(np.mean((current_polarities - initial_polarities)**2))
            
            results.append({
                'model': model_name,  # Add this line
                'bin': bin_number,
                'iteration': iteration,
                'avg_initial_sentiment': avg_initial_sentiment,
                'avg_current_sentiment': avg_current_sentiment,
                'sentiment_change': sentiment_change,
                'absolute_change': absolute_change,
                'std_dev': std_dev,
                'confidence_interval': confidence_interval,
                'mae': mae,
                'rmse': rmse
            })

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_folder, 'analysis_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Analysis results saved to {csv_path}")

    return results_df

def plot_sentiment_trajectory(results_df, output_folder, model_name, font_size=12):
    plt.figure(figsize=(15, 10))
    
    # Create a custom colormap
    colors = LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    
    # Get unique bins and sort them
    bins = sorted(results_df['bin'].unique())
    
    for bin_num in bins:
        bin_data = results_df[results_df['bin'] == bin_num]
        color = colors((bin_num - 1) / (len(bins) - 1))  # Normalize to [0, 1]
        plt.scatter(bin_data['iteration'], bin_data['avg_current_sentiment'], 
                    label=f'Bin {bin_num}', color=color, edgecolors='black', linewidth=0.5,
                    s=45)  # s is the size of the dots
    
    plt.title(f'Sentiment Trajectory Across Iterations\nModel: {model_name}', fontsize=font_size+4)
    plt.xlabel('Iteration', fontsize=font_size+2)
    plt.ylabel('Average Sentiment', fontsize=font_size+2)
    plt.legend(fontsize=font_size, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-1, 1)  # Assuming sentiment ranges from -1 to 1
    
    # Add subtle background color
    plt.gca().set_facecolor('#f0f0f0')
    
    # Customize tick parameters
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'sentiment_trajectory_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sentiment trajectory plot saved to {plot_path}")

def plot_sentiment_heatmap(results_df, output_folder):
    # Pivot the dataframe to create a matrix suitable for a heatmap
    heatmap_data = results_df.pivot(index='bin', columns='iteration', values='avg_current_sentiment')
    
    # Calculate the range of values for normalization
    vmin = heatmap_data.min().min()
    vmax = heatmap_data.max().max()
    vcenter = 0  # Assuming 0 is our neutral sentiment
    
    # Create a custom normalization to emphasize values around the center
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    
    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_data, cmap='RdBu_r', norm=norm, center=0,
                annot=False, cbar_kws={'label': 'Average Sentiment'})
    
    plt.title('Heatmap of Sentiment Change Across Bins and Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Sentiment Bin')
    
    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'sentiment_heatmap.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Sentiment heatmap saved to {plot_path}")

import numpy as np

def calculate_sentiment_fidelity(results_df, output_folder, weights=(0.4, 0.4, 0.2)):
    def safe_normalize(series):
        # For now just return the series as is
        return series

    # Define weights
    w1, w2, w3 = weights

    # Group by bin and iteration
    grouped = results_df.groupby(['bin', 'iteration'])

    # Calculate fidelity score for each group
    def calculate_fidelity(group):
        normalized_rmse = safe_normalize(group['rmse'])
        normalized_absolute_change = safe_normalize(group['absolute_change'])
        normalized_std_dev = safe_normalize(group['std_dev'])
        
        denominator = w1 * normalized_rmse + w2 * normalized_absolute_change + w3 * normalized_std_dev
        
        if denominator.iloc[0] == 0:
            return pd.Series({'fidelity_score': 1.0})  # Perfect fidelity if all normalized values are 0
        else:
            return pd.Series({'fidelity_score': 1 / denominator.iloc[0]})

    fidelity_scores = grouped.apply(calculate_fidelity).reset_index()

    # Save fidelity scores to CSV
    csv_path = os.path.join(output_folder, 'sentiment_fidelity_scores.csv')
    temp_fidelity_scores = fidelity_scores.copy()
    # Save all the fidelity scores to the CSV file, except for iteration 0
    temp_fidelity_scores = temp_fidelity_scores[temp_fidelity_scores['iteration'] != 0]
    temp_fidelity_scores.to_csv(csv_path, index=False)
    print(f"Sentiment fidelity scores saved to {csv_path}")

    # Prepare the output
    output = []
    output.append("Sentiment Fidelity Scores\n")
    output.append("==========================\n\n")

    # Scores for each bin
    for bin_num in fidelity_scores['bin'].unique():
        bin_scores = fidelity_scores[fidelity_scores['bin'] == bin_num]
        output.append(f"Bin {bin_num}:\n")
        output.append("Iteration\tFidelity Score\n")
        for _, row in bin_scores.iterrows():
            if (row['iteration'] in [5] or row['iteration'] % 10 == 0 and row['iteration'] != 0) or row['iteration'] == 1:
                output.append(f"{row['iteration']}\t\t{row['fidelity_score']:.4f}\n")
        output.append("\n")

    # Overall scores
    output.append("Overall Scores (average across all bins):\n")
    output.append("Iteration\tFidelity Score\n")
    overall_scores = fidelity_scores.groupby('iteration')['fidelity_score'].mean()
    for iteration, score in overall_scores.items():
        if (iteration in [5] or iteration % 10 == 0 and iteration != 0) or iteration == 1:
            output.append(f"{iteration}\t\t{score:.4f}\n")
    output.append("\n")

    # Formula and explanation
    output.append("Sentiment Fidelity Score Formula:\n")
    output.append("Fidelity Score = 1 / (w1 * RMSE + w2 * absolute_change + w3 * std_dev)\n\n")
    
    output.append(f"Weights: w1 = {w1}, w2 = {w2}, w3 = {w3}\n\n")
    
    output.append("Explanation of terms:\n")
    output.append("1. RMSE (Root Mean Square Error): Measures the overall error in sentiment prediction. A higher RMSE indicates lower fidelity.\n")
    output.append("   Justification: RMSE captures the magnitude of sentiment shifts, penalizing larger errors more heavily.\n\n")
    
    output.append("2. Absolute Change: Represents the magnitude of change from the initial sentiment. Larger changes indicate lower fidelity.\n")
    output.append("   Justification: This directly measures how much the sentiment has shifted from its original value.\n\n")
    
    output.append("3. Standard Deviation: Indicates the consistency of sentiment changes across samples. Higher variability suggests lower fidelity.\n")
    output.append("   Justification: This captures how consistently the sentiment changes across different samples, with more consistent changes indicating higher fidelity.\n\n")
    
    output.append("The score is inverted (1 / ...) so that higher values indicate better fidelity. Each term is normalized to ensure they contribute equally to the score.\n")
    output.append("Note: In cases where all values are the same for a particular metric, that metric is considered to have perfect fidelity and is assigned a normalized value of 1.\n")
    output.append("If all normalized values are 0, indicating perfect fidelity across all metrics, the fidelity score is set to 1 (maximum fidelity).\n")

    # Write to file
    with open(os.path.join(output_folder, 'sentiment_fidelity_report.txt'), 'w') as f:
        f.writelines(output)

    print(f"Sentiment fidelity report saved to {os.path.join(output_folder, 'sentiment_fidelity_report.txt')}")

    return fidelity_scores

def plot_fidelity_scores(fidelity_scores, output_folder, model_name, font_size=12):
    plt.figure(figsize=(15, 10))
    
    # Create a custom colormap
    colors = LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    
    # Get unique bins and sort them
    bins = sorted(fidelity_scores['bin'].unique())
    
    for bin_num in bins:
        bin_data = fidelity_scores[fidelity_scores['bin'] == bin_num]
        color = colors((bin_num - 1) / (len(bins) - 1))  # Normalize to [0, 1]
        plt.scatter(bin_data['iteration'], bin_data['fidelity_score'], 
                    label=f'Bin {bin_num}', color=color, edgecolors='black', linewidth=0.5,
                    s=45, alpha=0.7)
        
        # Plot a dotted line connecting the points
        plt.plot(bin_data['iteration'], bin_data['fidelity_score'], color=color, linestyle='--', linewidth=1)
    
    plt.title(f'Sentiment Fidelity Scores Across Iterations\nModel: {model_name}', fontsize=font_size+4)
    plt.xlabel('Iteration', fontsize=font_size+2)
    plt.ylabel('Fidelity Score', fontsize=font_size+2)
    plt.legend(fontsize=font_size, loc='upper right', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis to start from 0
    plt.ylim(0, 27)
    
    # Add subtle background color
    plt.gca().set_facecolor('#f0f0f0')
    
    # Customize tick parameters
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    
    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'fidelity_scores_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Fidelity scores plot saved to {plot_path}")

def compare_fidelity_scores(folder_paths, output_folder, folder_names):
    plt.figure(figsize=(20, 12))
    
    # Create a custom colormap
    colors = plt.cm.get_cmap('tab10')  # Using a different colormap for distinguishing models
    
    for i, folder in enumerate(folder_paths):
        csv_path = os.path.join(folder, 'sentiment_fidelity_scores.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            overall_scores = df.groupby('iteration')['fidelity_score'].median()
            
            color = colors(i)
            # Plot scatter points
            plt.scatter(overall_scores.index, overall_scores.values, 
                        label=folder_names[folder], color=color, 
                        edgecolors='black', linewidth=0.5, s=50, alpha=0.7)
            
            # Add dotted line connecting the scatter points
            plt.plot(overall_scores.index, overall_scores.values, 
                     color=color, linestyle=':', linewidth=1.5, alpha=0.7)
        else:
            print(f"No fidelity scores found for folder: {csv_path}")
    
    plt.xlabel('Iteration', fontsize=36)
    plt.ylabel('Median Fidelity Score', fontsize=36)
    plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    # Set background color to white
    plt.gca().set_facecolor('white')
    
    # Customize tick parameters
    plt.tick_params(axis='both', which='major', labelsize=30)
    
    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'fidelity_scores_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    # Save the plot as a PDF
    pdf_path = os.path.join(output_folder, 'fidelity_scores_comparison.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fidelity scores comparison plot saved as PDF to {pdf_path}")
    plt.close()
    
    print(f"Fidelity scores comparison plot saved to {plot_path}")

def plot_extrapolated_sentiment_trajectory(results_df, output_folder, model_name, extrapolation_steps=50, font_size=12):
    plt.figure(figsize=(20, 12))
    
    max_iteration = results_df['iteration'].max()
    extrapolated_iterations = np.arange(0, max_iteration + extrapolation_steps + 1)
    
    # Create a custom colormap
    colors = LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    
    # Get unique bins and sort them
    bins = sorted(results_df['bin'].unique())
    
    linear_final = []
    for bin_num in bins:
        bin_data = results_df[results_df['bin'] == bin_num]
        color = colors((bin_num - 1) / (len(bins) - 1))  # Normalize to [0, 1]
        
        # Plot original data
        plt.scatter(bin_data['iteration'], bin_data['avg_current_sentiment'], 
                    label=f'Bin {bin_num} (Original)', color=color, edgecolors='black', 
                    linewidth=0.5, s=15, alpha=0.7)
        
        # Perform linear regression
        model = LinearRegression().fit(bin_data['iteration'].values.reshape(-1, 1),
                                       bin_data['avg_current_sentiment'].values)
        
        # Extrapolate
        y_extrapolated = model.predict(extrapolated_iterations.reshape(-1, 1))
        
        # Plot extrapolated data
        plt.plot(extrapolated_iterations, y_extrapolated, 
                 label=f'Bin {bin_num} (Extrapolated)', color=color, linestyle='--', linewidth=2)
        
        linear_final.append(y_extrapolated[-1])
    
    plt.title(f'Extrapolated Sentiment Trajectory Across Iterations (Linear)\nModel: {model_name}', fontsize=font_size*2.5)
    plt.xlabel('Iteration', fontsize=font_size*2)
    plt.ylabel('Average Sentiment', fontsize=font_size*2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-1, 1)  # Assuming sentiment ranges from -1 to 1
    
    # Add subtle background color
    plt.gca().set_facecolor('#f0f0f0')
    
    # Customize tick parameters
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'linear_extrapolation_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Linear extrapolation plot saved to {plot_path}")
    print(f"Linear extrapolation final range: {min(linear_final):.2f} to {max(linear_final):.2f}")
    print(f"Final iteration: {extrapolated_iterations[-1]}")

    return linear_final, extrapolated_iterations[-1]

def calculate_intersection_point(results_df, extrapolation_steps):
    max_iteration = results_df['iteration'].max()
    extrapolated_iterations = np.arange(0, max_iteration + extrapolation_steps + 1)
    
    models = {}
    for bin_num in results_df['bin'].unique():
        bin_data = results_df[results_df['bin'] == bin_num]
        X = bin_data['iteration'].values.reshape(-1, 1)
        y = bin_data['avg_current_sentiment'].values
        model = LinearRegression().fit(X, y)
        models[bin_num] = model
    
    # Check for intersections
    for i in range(len(extrapolated_iterations)):
        sentiments = [model.predict([[extrapolated_iterations[i]]])[0] for model in models.values()]
        if max(sentiments) - min(sentiments) < 0.01:  # Consider them intersected if within 0.01
            return (extrapolated_iterations[i], np.mean(sentiments))
    
    return None  # No clear intersection found

def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def plot_exponential_extrapolation(results_df, output_folder, extrapolation_steps=50):
    plt.figure(figsize=(20, 12))
    
    max_iteration = results_df['iteration'].max()
    extrapolated_iterations = np.arange(0, max_iteration + extrapolation_steps + 1)
    
    exp_final = []
    for bin_num in results_df['bin'].unique():
        bin_data = results_df[results_df['bin'] == bin_num]
        
        # Plot original data
        plt.plot(bin_data['iteration'], bin_data['avg_current_sentiment'], 
                 label=f'Bin {bin_num} (Original)', alpha=0.7)
        
        # Exponential extrapolation
        popt, _ = curve_fit(exponential_decay, bin_data['iteration'], bin_data['avg_current_sentiment'],
                            p0=[1, 0.1, np.mean(bin_data['avg_current_sentiment'])], maxfev=10000)
        y_exp = exponential_decay(extrapolated_iterations, *popt)
        plt.plot(extrapolated_iterations, y_exp, 
                 linestyle='--', label=f'Bin {bin_num} (Extrapolated)')
        
        exp_final.append(y_exp[-1])
    
    plt.title('Extrapolated Sentiment Trajectory (Exponential Fit)')
    plt.xlabel('Iteration')
    plt.ylabel('Average Sentiment')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'exponential_extrapolation_plot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"Exponential extrapolation plot saved to {plot_path}")
    print(f"Exponential extrapolation final range: {min(exp_final):.2f} to {max(exp_final):.2f}")
    print(f"Final iteration: {extrapolated_iterations[-1]}")

    return exp_final, extrapolated_iterations[-1]

def compare_positive_negative_shifts(folder_paths, folder_names, output_folder):
    shift_counts = {}

    for folder_path in folder_paths:
        positive_shifts = 0
        negative_shifts = 0

        bin_folders = [f for f in os.listdir(folder_path) if f.startswith('bin_') and os.path.isdir(os.path.join(folder_path, f))]

        for bin_folder in bin_folders:
            bin_path = os.path.join(folder_path, bin_folder)
            csv_files = [f for f in os.listdir(bin_path) if f.endswith('.csv')]

            for csv_file in csv_files:
                file_path = os.path.join(bin_path, csv_file)
                df = pd.read_csv(file_path)
                
                first_sentiment = df['polarity'].iloc[0]
                last_sentiment = df['polarity'].iloc[-1]
                
                if last_sentiment > first_sentiment:
                    positive_shifts += 1
                elif last_sentiment < first_sentiment:
                    negative_shifts += 1
                # If they're equal, we don't count it as a shift

        shift_counts[folder_names[folder_path]] = (positive_shifts, negative_shifts)

    # Plotting
    plt.figure(figsize=(12, 8))
    
    max_count = max(max(abs(pos), abs(neg)) for pos, neg in shift_counts.values())
    y_limit = max_count + (max_count * 0.1)  # Add 10% padding

    x = np.arange(len(shift_counts))
    width = 0.35

    for i, (folder_name, (pos, neg)) in enumerate(shift_counts.items()):
        plt.bar(i - width/2, pos, width, label=f'{folder_name} Positive', color='green')
        plt.bar(i + width/2, -neg, width, label=f'{folder_name} Negative', color='red')

    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.ylabel('Number of Shifts')
    plt.title('Comparison of Positive and Negative Sentiment Shifts')
    plt.xticks(x, shift_counts.keys())
    plt.ylim(-y_limit, y_limit)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'positive_negative_shifts_comparison.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Positive/Negative shifts comparison plot saved to {plot_path}")

    # Print results
    print("\nShift Counts:")
    for folder_name, (pos, neg) in shift_counts.items():
        print(f"{folder_name}:")
        print(f"  Positive shifts: {pos}")
        print(f"  Negative shifts: {neg}")
        print(f"  Total shifts: {pos + neg}")
        print(f"  Net shift: {pos - neg}")
        print()

import matplotlib.pyplot as plt
import numpy as np

def calculate_shift_magnitude(folder_paths, folder_names, output_folder):
    shift_magnitudes = {}
    bin_magnitudes = {}

    for folder_path in folder_paths:
        total_magnitude = 0
        bin_magnitudes[folder_names[folder_path]] = {}

        bin_folders = sorted([f for f in os.listdir(folder_path) if f.startswith('bin_') and os.path.isdir(os.path.join(folder_path, f))])

        for bin_folder in bin_folders:
            bin_path = os.path.join(folder_path, bin_folder)
            csv_files = [f for f in os.listdir(bin_path) if f.endswith('.csv')]

            bin_magnitude = 0
            for csv_file in csv_files:
                file_path = os.path.join(bin_path, csv_file)
                df = pd.read_csv(file_path)
                
                first_sentiment = df['polarity'].iloc[0]
                last_sentiment = df['polarity'].iloc[-1]
                
                magnitude = abs(last_sentiment - first_sentiment)
                bin_magnitude += magnitude

            total_magnitude += bin_magnitude
            bin_magnitudes[folder_names[folder_path]][bin_folder] = bin_magnitude

        shift_magnitudes[folder_names[folder_path]] = total_magnitude

    # Plotting
    plt.figure(figsize=(12, 8))
    
    max_magnitude = max(shift_magnitudes.values())
    y_limit = max_magnitude + (max_magnitude * 0.1)  # Add 10% padding

    colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(bin_folders)))
    
    bottom = np.zeros(len(shift_magnitudes))
    for i, bin_folder in enumerate(bin_folders):
        values = [bin_magnitudes[folder][bin_folder] for folder in shift_magnitudes.keys()]
        plt.bar(shift_magnitudes.keys(), values, bottom=bottom, color=colors[i], label=bin_folder)
        bottom += values

    plt.ylabel('Total Shift Magnitude')
    plt.title('Comparison of Total Sentiment Shift Magnitudes by Bin')
    plt.ylim(0, y_limit)
    plt.legend(title='Bins', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'shift_magnitude_comparison_by_bin.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    print(f"Shift magnitude comparison plot saved to {plot_path}")

    # Print results
    print("\nTotal Shift Magnitudes:")
    for folder_name, magnitude in shift_magnitudes.items():
        print(f"{folder_name}: {magnitude:.4f}")
        print("  Breakdown by bin:")
        for bin_folder, bin_magnitude in bin_magnitudes[folder_name].items():
            print(f"    {bin_folder}: {bin_magnitude:.4f}")
        print()

def calculate_shift_magnitude_with_bin_selection(folder_paths, folder_names, output_folder):
    shift_magnitudes = {}
    bin_magnitudes = {}

    # First, identify all unique bins across all folders
    all_bins = set()
    for folder_path in folder_paths:
        bin_folders = [f for f in os.listdir(folder_path) if f.startswith('bin_') and os.path.isdir(os.path.join(folder_path, f))]
        all_bins.update(bin_folders)

    # Sort the bins
    all_bins = sorted(list(all_bins))

    # Ask user which bins to include
    print("Available bins:")
    for i, bin_folder in enumerate(all_bins):
        print(f"{i+1}. {bin_folder}")
    
    selected_indices = input("Enter the numbers of the bins you want to include (comma-separated), or press Enter to include all: ")
    if selected_indices.strip():
        selected_indices = [int(i.strip()) - 1 for i in selected_indices.split(',')]
        selected_bins = [all_bins[i] for i in selected_indices]
    else:
        selected_bins = all_bins

    for folder_path in folder_paths:
        total_magnitude = 0
        bin_magnitudes[folder_names[folder_path]] = {}

        for bin_folder in selected_bins:
            bin_path = os.path.join(folder_path, bin_folder)
            if not os.path.exists(bin_path):
                bin_magnitudes[folder_names[folder_path]][bin_folder] = 0
                continue

            csv_files = [f for f in os.listdir(bin_path) if f.endswith('.csv')]

            bin_magnitude = 0
            for csv_file in csv_files:
                file_path = os.path.join(bin_path, csv_file)
                df = pd.read_csv(file_path)
                
                first_sentiment = df['polarity'].iloc[0]
                last_sentiment = df['polarity'].iloc[-1]
                
                magnitude = abs(last_sentiment - first_sentiment)
                bin_magnitude += magnitude

            total_magnitude += bin_magnitude
            bin_magnitudes[folder_names[folder_path]][bin_folder] = bin_magnitude

        shift_magnitudes[folder_names[folder_path]] = total_magnitude

    # Plotting
    plt.figure(figsize=(12, 8))
    
    max_magnitude = max(shift_magnitudes.values())
    y_limit = max_magnitude + (max_magnitude * 0.1)  # Add 10% padding

    colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(selected_bins)))
    
    bottom = np.zeros(len(shift_magnitudes))
    for i, bin_folder in enumerate(selected_bins):
        values = [bin_magnitudes[folder][bin_folder] for folder in shift_magnitudes.keys()]
        plt.bar(shift_magnitudes.keys(), values, bottom=bottom, color=colors[i], label=bin_folder)
        bottom += values

    plt.ylabel('Total Shift Magnitude')
    plt.title('Comparison of Total Sentiment Shift Magnitudes by Selected Bins')
    plt.ylim(0, y_limit)
    plt.legend(title='Bins', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'shift_magnitude_comparison_selected_bins.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    print(f"Shift magnitude comparison plot saved to {plot_path}")

    # Print results
    print("\nTotal Shift Magnitudes (for selected bins):")
    for folder_name, magnitude in shift_magnitudes.items():
        print(f"{folder_name}: {magnitude:.4f}")
        print("  Breakdown by selected bin:")
        for bin_folder in selected_bins:
            print(f"    {bin_folder}: {bin_magnitudes[folder_name][bin_folder]:.4f}")
        print()

def plot_average_sentiment_trajectory(output_folder, font_size=12):
    plt.figure(figsize=(20, 15))
    
    models = ['Phi2 3B', 'GPT4o', 'Mistral 7B Instruct v0.2', 'Llama3 8B Instruct']
    file_paths = {}
    
    print("Please provide the file paths for each model's analysis results:")
    for model in models:
        while True:
            path = input(f"Enter the file path for {model} analysis results: ")
            if os.path.exists(path):
                file_paths[model] = path
                break
            else:
                print("File not found. Please enter a valid file path.")
    
    legend_lines = []
    legend_labels = []
    
    for i, model in enumerate(models):
        ax = plt.subplot(2, 2, i+1)
        
        try:
            results_df = pd.read_csv(file_paths[model])
            
            for bin_num in range(1, 6):
                bin_data = results_df[results_df['bin'] == bin_num]
                line, = plt.plot(bin_data['iteration'], bin_data['avg_current_sentiment'], 
                         marker='o', markersize=3, linestyle=':', linewidth=1)
                
                if i == 0:  # Only add to legend for the first subplot
                    legend_lines.append(line)
                    legend_labels.append(f'Bin {bin_num}')
            
            plt.title(model, fontsize=font_size+2)
            if i in [2, 3]:  # Only add x-label for bottom plots
                plt.xlabel('Iteration', fontsize=font_size)
            if i in [0, 2]:  # Only add y-label for left plots
                plt.ylabel('Average Normalized Polarity', fontsize=font_size)
            plt.ylim(-1, 1)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tick_params(axis='both', which='major', labelsize=font_size-2)
            
            # Remove x-axis labels for top plots
            if i in [0, 1]:
                ax.set_xticklabels([])
        
        except Exception as e:
            print(f"Error processing data for {model}: {e}")
            plt.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=plt.gca().transAxes)

    # Add a single legend to the first subplot
    first_ax = plt.gcf().axes[0]
    first_ax.legend(legend_lines, legend_labels, fontsize=font_size-2, loc='lower right')

    plt.tight_layout()
    plt.savefig(f"{output_folder}/average_sentiment_trajectory.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_folder}/average_sentiment_trajectory.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Average sentiment trajectory plot saved to {output_folder}/average_sentiment_trajectory.png")

def plot_linear_extrapolation(output_folder, extrapolation_steps=200, font_size=12):
    plt.figure(figsize=(20, 15))
    
    models = ['Phi2 3B', 'GPT4o', 'Mistral 7B Instruct v0.2', 'Llama3 8B Instruct']
    file_paths = {}
    
    print("Please provide the file paths for each model's analysis results:")
    for model in models:
        while True:
            path = input(f"Enter the file path for {model} analysis results: ")
            if os.path.exists(path):
                file_paths[model] = path
                break
            else:
                print("File not found. Please enter a valid file path.")
    
    legend_lines = []
    legend_labels = []
    
    for i, model in enumerate(models):
        ax = plt.subplot(2, 2, i+1)
        
        try:
            results_df = pd.read_csv(file_paths[model])
            max_iteration = results_df['iteration'].max()
            extrapolated_iterations = np.arange(0, max_iteration + extrapolation_steps + 1)
            
            for bin_num in range(1, 6):
                bin_data = results_df[results_df['bin'] == bin_num]
                
                X = bin_data['iteration'].values.reshape(-1, 1)
                y = bin_data['avg_current_sentiment'].values
                
                lin_reg = LinearRegression().fit(X, y)
                y_extrapolated = lin_reg.predict(extrapolated_iterations.reshape(-1, 1))
                
                line, = plt.plot(extrapolated_iterations, y_extrapolated)
                plt.scatter(bin_data['iteration'], bin_data['avg_current_sentiment'], s=10)
                
                if i == 0:  # Only add to legend for the first subplot
                    legend_lines.append(line)
                    legend_labels.append(f'Bin {bin_num}')
            
            plt.title(model, fontsize=font_size+2)
            if i in [2, 3]:  # Only add x-label for bottom plots
                plt.xlabel('Iteration', fontsize=font_size)
            if i in [0, 2]:  # Only add y-label for left plots
                plt.ylabel('Average Sentiment', fontsize=font_size)
            plt.ylim(-1, 1)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tick_params(axis='both', which='major', labelsize=font_size-2)
            
            # Remove x-axis labels for top plots
            if i in [0, 1]:
                ax.set_xticklabels([])
        
        except Exception as e:
            print(f"Error processing data for {model}: {e}")
            plt.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=plt.gca().transAxes)

    # Add a single legend to the first subplot
    first_ax = plt.gcf().axes[0]
    first_ax.legend(legend_lines, legend_labels, fontsize=font_size-2, loc='lower right')

    plt.tight_layout()
    plt.savefig(f"{output_folder}/linear_extrapolation.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_folder}/linear_extrapolation.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Linear extrapolation plot saved to {output_folder}/linear_extrapolation.png")

def plot_sentiment_fidelity_scores(output_folder, font_size=12):
    plt.figure(figsize=(20, 13))
    
    models = ['Phi2 3B', 'GPT4o', 'Mistral 7B Instruct v0.2', 'Llama3 8B Instruct']
    file_paths = {}
    
    print("Please provide the file paths for each model's analysis results:")
    for model in models:
        while True:
            path = input(f"Enter the file path for {model} analysis results: ")
            if os.path.exists(path):
                file_paths[model] = path
                break
            else:
                print("File not found. Please enter a valid file path.")
    
    legend_lines = []
    legend_labels = []
    max_fidelity_score = 0
    
    for i, model in enumerate(models):
        ax = plt.subplot(2, 2, i+1)
        
        try:
            results_df = pd.read_csv(file_paths[model])
            
            for bin_num in range(1, 6):
                bin_data = results_df[results_df['bin'] == bin_num]
                line, = plt.plot(bin_data['iteration'], bin_data['fidelity_score'], 
                         marker='o', markersize=3)
                
                if i == 0:  # Only add to legend for the first subplot
                    legend_lines.append(line)
                    legend_labels.append(f'Bin {bin_num}')
                
                max_fidelity_score = max(max_fidelity_score, bin_data['fidelity_score'].max())
            
            # Add title inside the plot
            ax.text(0.5, 0.98, model, fontsize=font_size+2, ha='center', va='top', transform=ax.transAxes)
            
            if i in [2, 3]:  # Only add x-label for bottom plots
                plt.xlabel('Iteration', fontsize=font_size)
            if i in [0, 2]:  # Only add y-label for left plots
                plt.ylabel('Fidelity Score', fontsize=font_size)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tick_params(axis='both', which='major', labelsize=font_size-2)
            
            # Remove x-axis labels for top plots
            if i in [0, 1]:
                ax.set_xticklabels([])
        
        except Exception as e:
            print(f"Error processing data for {model}: {e}")
            plt.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=plt.gca().transAxes)

    # Set y-limit a bit higher than the maximum fidelity score
    y_limit = max_fidelity_score * 1.1  # 10% higher than the max score
    for ax in plt.gcf().axes:
        ax.set_ylim(0, y_limit)

    # Add a single legend to the first subplot
    first_ax = plt.gcf().axes[0]
    first_ax.legend(legend_lines, legend_labels, fontsize=font_size-2, loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{output_folder}/sentiment_fidelity_scores.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_folder}/sentiment_fidelity_scores.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Sentiment fidelity scores plot saved to {output_folder}/sentiment_fidelity_scores.png")

###########################################################################################################################
######################################## MAIN FUNCTION ####################################################################
def main():
    while True:
        parent_folder = input("Enter the folder path to analyse: ")
        if os.path.exists(parent_folder):
            break
        else:
            print("The folder does not exist. Please try again.")

    output_folder = os.path.join('output_graphs', os.path.basename(parent_folder))

    model_name = get_model_name()
    font_size = int(input("Enter the base font size for plots (default is 12): ") or "12")

    while True:
        print("\nOptions:")
        print("0. EXIT")
        print("00. CHANGE FOLDER PATH")
        print("\nNegative/Neutral/Positive Sentiment Analysis:")
        print("1. Produce Scatter Plots")
        print("2. Produce Box Plots")
        print("\nSentiment Bins Analysis:")
        print("3. Produce Scatter Plots")
        print("4. Produce Box Plots")
        print("5. Produce Result Analysis")
        print("\nSingle Bin Analysis:")
        print("6. Produce Scatter Plot")
        print("7. Produce Box Plot")
        print("8. Produce Violin Plot (Box plots are better)")
        print("9. Produce Result Analysis")
        print("\nProduct Score Distribution:")
        print("10. Perform Single Product Analysis")
        print("11. Perform All Product Analysis")
        print("\nSentiment Fidelity Comparison")
        print("12. Compare Fidelity Scores")
        print("13. Compare Positive/Negative Shifts")
        print("14. Compare Shift Magnitudes")
        print("15. Compare Shift Magnitudes with Bin Selection")
        
        choice = input("Choose an option: ")

        if choice == '0':
            break
        elif choice == '00':
            parent_folder = input("Enter the folder path to analyse: ")
            if os.path.exists(parent_folder):
                output_folder = os.path.join('output_graphs', os.path.basename(parent_folder))
            else:
                print("The folder does not exist. Please try again.")

            model_name = get_model_name()
            font_size = int(input("Enter the base font size for plots (default is 12): ") or "12")

        elif choice == '1':
            try:
                csv_files = load_csv_files_negative_positive(parent_folder)
                trend_dfs = calculate_avg_trend_negative_positive(csv_files)
                plot_scatter_trends_negative_positive(trend_dfs, output_folder)
                print(f"Scatter plot graphs saved in {output_folder}")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("You may be using the incorrect output data format. Please try again.")
        elif choice == '2':
            try: 
                csv_files = load_csv_files_negative_positive(parent_folder)
                trend_dfs = calculate_trend_negative_positive(csv_files)
                num_bins = int(input("Enter the number of time bins for the box plot: "))
                plot_boxplot_trends_negative_positive(trend_dfs, output_folder, num_bins)
                print(f"Box plot graphs saved in {output_folder}")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("You may be using the incorrect output data format. Please try again.")
        elif choice == '3':
            try:
                plot_scatter_trends_all_bins(parent_folder, output_folder, model_name, font_size)
            except Exception as e:
                print(f"An error occurred: {e}")
            print("You may be using the incorrect output data format. Please try again.")
        elif choice == '4':
            try:
                num_bins = int(input("Enter the number of time bins for the box plots: "))
                plot_boxplot_trends_all_bins(parent_folder, output_folder, num_bins, model_name, font_size)
            except Exception as e:
                print(f"An error occurred: {e}")
                print("You may be using the incorrect output data format. Please try again.")
        elif choice == '6':
            try:
                bin_folder = input("Enter the bin folder to analyse (e.g., bin_1): ")
                bin_path = os.path.join(parent_folder, bin_folder)
                if not os.path.exists(bin_path):
                    print("The specified bin folder does not exist.")
                    continue
                csv_files = load_csv_files_single_bin(bin_path)
                trend_df = calculate_trend_single_bin(csv_files)
                bin_number = bin_folder.split('_')[1]
                plot_scatter_trend_single_bin(trend_df, output_folder, bin_number, model_name, font_size)
                print(f"Scatter plot for {bin_folder} saved in {output_folder}")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("You may be using the incorrect output data format. Please try again.")
        elif choice == '7':
            try:
                bin_folder = input("Enter the bin folder to analyse (e.g., bin_1): ")
                bin_path = os.path.join(parent_folder, bin_folder)
                if not os.path.exists(bin_path):
                    print("The specified bin folder does not exist.")
                    continue
                num_bins = int(input("Enter the number of time bins for the box plot: "))
                csv_files = load_csv_files_single_bin(bin_path)
                
                # Load and normalize all CSV files
                normalized_dfs = []
                for file in csv_files:
                    df = pd.read_csv(file)
                    df['normalized_polarity'] = normalize_series(df['polarity'])
                    normalized_dfs.append(df[['normalized_polarity']])  # Keep only the normalized_polarity column
                
                bin_number = bin_folder.split('_')[1]
                plot_boxplot_trend_single_bin(normalized_dfs, output_folder, bin_number, num_bins, model_name, font_size)
                print(f"Box plot for {bin_folder} saved in {output_folder}")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("You may be using the incorrect output data format. Please try again.")
        elif choice == '8':
            try:
                bin_folder = input("Enter the bin folder to analyse (e.g., bin_1): ")
                bin_path = os.path.join(parent_folder, bin_folder)
                if not os.path.exists(bin_path):
                    print("The specified bin folder does not exist.")
                    continue
                csv_files = load_csv_files_single_bin(bin_path)
                
                # Load and normalize all CSV files
                normalized_dfs = []
                for file in csv_files:
                    df = pd.read_csv(file)
                    df['normalized_polarity'] = normalize_series(df['polarity'])
                    normalized_dfs.append(df[['normalized_polarity']])
                
                bin_number = bin_folder.split('_')[1]
                plot_violin_trend_single_bin(normalized_dfs, output_folder, bin_number)
                print(f"Violin plot for {bin_folder} saved in {output_folder}")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("You may be using the incorrect output data format. Please try again.")
        elif choice == '9':
            try:
                results_df = perform_result_analysis(parent_folder, output_folder, model_name)
                print("Result analysis completed. CSV file has been saved.")
                
                plot_sentiment_trajectory(results_df, output_folder, model_name, font_size=12)
                print("Sentiment trajectory plot has been generated.")

                fidelity_scores = calculate_sentiment_fidelity(results_df, output_folder, weights=(0.4, 0.4, 0.2))
                print("Sentiment fidelity scores have been calculated.")

                fidelity_scores = fidelity_scores[fidelity_scores["iteration"] != 0]
                plot_fidelity_scores(fidelity_scores, output_folder, model_name, font_size=18)                
                print("Fidelity scores plot has been generated.")
                # Linear extrapolation (assuming this function already exists)
                linear_final, final_iteration = plot_extrapolated_sentiment_trajectory(results_df, output_folder, model_name, extrapolation_steps=200, font_size=12)
                print("Linear extrapolation plot has been generated.")

                # Compare the results
                print("\nComparison of extrapolations:")
                print(f"Linear extrapolation final range: {min(linear_final):.2f} to {max(linear_final):.2f}")
                #print(f"Exponential extrapolation final range: {min(exp_final):.2f} to {max(exp_final):.2f}")
                print(f"Final iteration: {final_iteration}")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("You may be using the incorrect output data format. Please try again.")
        elif choice == '10':
            try:
                product_id = input("Enter the product ID to analyze: ")
                perform_product_analysis(parent_folder, product_id)
                print(f"Product analysis for product {product_id} completed.")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("You may be using the incorrect output data format. Please try again.")
        elif choice == '11':
            try:
                perform_product_analysis(parent_folder)
                print("Product analysis for all products completed.")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("You may be using the incorrect output data format. Please try again.")
        elif choice == '12':
            try:
                folder_paths = []
                folder_names = {}
                while True:
                    folder = input("Enter a folder path (or press Enter to finish): ")
                    if folder == "":
                        break
                    if os.path.exists(folder):
                        folder_paths.append(folder)
                        folder_name = input("Enter a name for the folder: ")
                        folder_names[folder] = folder_name
                    else:
                        print("The folder does not exist. Please try again.")
                
                if folder_paths:
                    comparison_output_folder = os.path.join('output_graphs', 'fidelity_comparison')
                    os.makedirs(comparison_output_folder, exist_ok=True)
                    compare_fidelity_scores(folder_paths, comparison_output_folder, folder_names)
                    print("Fidelity scores comparison completed.")
                else:
                    print("No valid folders were provided.")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("There might be an issue with the folder paths or data format.")
        elif choice == '13':
            try:
                folder_paths = []
                folder_names = {}
                while True:
                    folder = input("Enter a folder path (or press Enter to finish): ")
                    if folder == "":
                        break
                    if os.path.exists(folder):
                        folder_paths.append(folder)
                        folder_name = input("Enter a name for the folder: ")
                        folder_names[folder] = folder_name
                    else:
                        print("The folder does not exist. Please try again.")
                
                if folder_paths:
                    comparison_output_folder = os.path.join('output_graphs', 'fidelity_comparison')
                    os.makedirs(comparison_output_folder, exist_ok=True)
                    compare_positive_negative_shifts(folder_paths, folder_names, comparison_output_folder)
                    print("Positive/Negative shifts comparison completed.")
                else:
                    print("No valid folders were provided.")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("There might be an issue with the folder paths or data format.")
        elif choice == '14':
            try:
                folder_paths = []
                folder_names = {}
                while True:
                    folder = input("Enter a folder path (or press Enter to finish): ")
                    if folder == "":
                        break
                    if os.path.exists(folder):
                        folder_paths.append(folder)
                        folder_name = input("Enter a name for the folder: ")
                        folder_names[folder] = folder_name
                    else:
                        print("The folder does not exist. Please try again.")
                
                if folder_paths:
                    comparison_output_folder = os.path.join('output_graphs', 'fidelity_comparison')
                    os.makedirs(comparison_output_folder, exist_ok=True)
                    calculate_shift_magnitude(folder_paths, folder_names, comparison_output_folder)
                    print("Shift magnitude comparison completed.")
                else:
                    print("No valid folders were provided.")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("There might be an issue with the folder paths or data format.")
        elif choice == '15':
            try:
                folder_paths = []
                folder_names = {}
                while True:
                    folder = input("Enter a folder path (or press Enter to finish): ")
                    if folder == "":
                        break
                    if os.path.exists(folder):
                        folder_paths.append(folder)
                        folder_name = input("Enter a name for the folder: ")
                        folder_names[folder] = folder_name
                    else:
                        print("The folder does not exist. Please try again.")
                
                if folder_paths:
                    comparison_output_folder = os.path.join('output_graphs', 'fidelity_comparison')
                    os.makedirs(comparison_output_folder, exist_ok=True)
                    calculate_shift_magnitude_with_bin_selection(folder_paths, folder_names, comparison_output_folder)
                    print("Shift magnitude comparison with bin selection completed.")
                else:
                    print("No valid folders were provided.")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("There might be an issue with the folder paths or data format.")
        elif choice == '16':
            try:
                plot_average_sentiment_trajectory(output_folder, font_size)
                print("Average sentiment trajectory plot has been generated.")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("There might be an issue with the file paths or data format.")

        elif choice == '17':
            try:
                plot_linear_extrapolation(output_folder, extrapolation_steps=200, font_size=font_size)
                print("Linear extrapolation plot has been generated.")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Make sure the analysis_results.csv file exists and contains the required data.")

        elif choice == '18':
            try:
                plot_sentiment_fidelity_scores(output_folder, font_size)
                print("Sentiment fidelity scores plot has been generated.")
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Make sure the sentiment_fidelity_scores.csv file exists and contains the required data.")

if __name__ == "__main__":
    main()