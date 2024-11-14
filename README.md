# Twitter Sentiment Analysis

**DS Club Participants** used a machine learning algorithm to train a model that classifies tweets into four categories:
1. Positive
2. Negative
3. Neutral
4. Irrelevant

This project involves training a sentiment analysis model on Twitter data, followed by deploying it as a user-friendly web application for real-time sentiment prediction.

## Project Overview

This repository contains the scripts and datasets used for training and deploying the sentiment analysis model. The model was trained on a labeled dataset of tweets, using features engineered to capture the sentiment of each tweet. After training, the model was integrated into a web application using Streamlit, making it easy for users to input text and receive instant sentiment predictions.

## Repository Contents

- **`headers_only.csv`**: A CSV file with headers only, presumably used as a template or for dataset structure reference.
- **`train_set.ipynb`**: A Jupyter notebook used for training the sentiment analysis model.
- **`twitter_sentiment.ipynb`**: A Jupyter notebook for experimenting with or testing the sentiment analysis model.
- **`twitter_streamlit.py`**: A Streamlit application script to deploy the model as a web app, allowing users to input tweet text and see the predicted sentiment.
- **`twitter_training.csv`**: The main dataset used for training the model, containing tweets and their associated sentiment labels.
- **`twitter_validation.csv`**: A validation dataset to evaluate the performance of the model.

## Getting Started

### Prerequisites

To run this project, you will need Python installed with the following packages:
- Pandas
- Scikit-learn
- Streamlit
- Any other dependencies specified in your training or Streamlit scripts

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/sentiment_analysis_twitter.git
2. **Install dependencies:**:
   ```bash
   pip install -r requirements.txt
3. **Running the Training Script**:
To train the model, open and execute the train_set.ipynb notebook. This notebook will walk you through the data preprocessing, model training, and evaluation steps.
4. **Launching the Web App**:
   ```bash
   streamlit run twitter_streamlit.py
App will launch automatically in your browser.
   
### Usage:
The web app provides an easy-to-use interface for analyzing the sentiment of tweets. Simply enter a tweet, and the app will classify it as Positive, Negative, Neutral, or Irrelevant.

### Contributors
DS Club 1.0 Participants from Makerspace Petropavl
This project was developed by DS Club Participants as part of a club initiative to explore machine learning applications in sentiment analysis.

### License
This project is licensed under the MIT License.
