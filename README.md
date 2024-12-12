# Sentiment Fidelity Analysis Framework

This framework analyzes how well Large Language Models (LLMs) preserve sentiment when rephrasing text. It uses Amazon fine food reviews as a baseline dataset and measures sentiment preservation across multiple rephrasings.

Please read our paper: 
https://doi.org/10.1145/3672608.3707717

## Overview

The framework:
1. Processes Amazon reviews through multiple sentiment analysis models to establish ground truth
2. Uses an LLM to generate multiple rephrasings of each review
3. Analyzes sentiment preservation using metrics like RMSE, absolute change, and standard deviation
4. Generates visualizations and reports on sentiment fidelity

## Prerequisites

- Python 3.8+
- OpenAI API key (if using GPT-4) or equivalent for other LLMs
- Required Python packages (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - transformers
  - torch
  - vaderSentiment
  - tqdm
  - matplotlib
  - scipy
  - openai

## Setup

1. Clone the repository
2. Install dependencies
3. Set up your API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
4. Download the Amazon Fine Food Reviews dataset from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

## Usage

### 1. Data Processing

First, process the raw Amazon reviews (please refer to the readme in the input_data) folder. 

This will:
- Sample reviews across all star ratings
- Perform sentiment analysis using multiple models
- Normalize and bin sentiment scores
- Create a processed dataset

### 2. Running Analysis

First, run the main script to set up the analysis:
```bash
python main.py
```

This will:
- Create a timestamped output directory
- Prompt for LLM model information
- Generate initial reports

Then run the analysis:
```bash
python analysis.py
```

This will:
- Calculate sentiment fidelity metrics (RMSE, absolute change, standard deviation)
- Generate visualizations of sentiment preservation
- Produce detailed analysis reports
- Output raw data for further analysis

### 3. Customization

To use a different LLM:
1. Modify the `rephraser.py` file
2. Update the model configuration in `generate_completion()` function

```Python
python
def generate_completion(prompt, conversation_history):
# Modify this function to use your preferred LLM
```


## Output

The framework generates:
- Sentiment analysis across multiple rephrasings
- Visualizations of sentiment drift
- Fidelity metrics reports
- Raw data in CSV format for further analysis

## Key Files

- `get_dataset_sentiments.py`: Initial dataset processing and sentiment analysis
- `convert_dataset.py`: Data normalization and binning
- `main.py`: Core analysis pipeline
- `rephraser.py`: Text rephrasing using LLM
- `natural_language_processing.py`: Sentiment analysis tools
- `analysis.py`: Fidelity metrics calculation

## Metrics

The framework calculates sentiment fidelity using three main metrics:
1. RMSE (Root Mean Square Error): Measures overall sentiment prediction error
2. Absolute Change: Measures magnitude of sentiment shifts from original
3. Standard Deviation: Measures consistency of sentiment changes

## Customizing Analysis

You can modify:
- Number of rephrasing iterations
- Sampling methodology
- Sentiment analysis models
- Fidelity metrics and weights
- Visualization parameters