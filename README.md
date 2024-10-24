# Sentiment-Analysis-Python

## Overview
This project analyzes customer reviews from Amazon to determine the overall sentiment (positive, negative, neutral). By leveraging Natural Language Processing (NLP) techniques and Python, we aim to classify customer reviews into different sentiment categories. This analysis can help businesses better understand customer feedback and make data-driven decisions for improving products and services.

## Table of Contents

### Introduction
Project Workflow
Technologies Used
Data Preprocessing
Sentiment Analysis
Key Insights
Conclusion
Contributing

## Introduction
Amazon customer reviews offer valuable insights into user opinions and experiences. This project performs sentiment analysis using a dataset of customer reviews. The main goal is to classify the reviews into three categories:

Positive
Negative
Neutral
The outcome of this analysis can help Amazon and sellers gauge customer satisfaction and adjust their business strategies accordingly. Here is the Link of the Dataset: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

## Project Workflow
1. Data Collection
Amazon review data was collected from an open dataset, including columns such as Review Text, Rating, and Sentiment.
2. Data Preprocessing
Data Cleaning: Removal of null values, duplicate entries, and irrelevant columns.
Text Preprocessing: Tokenization, removal of stopwords, stemming, and lemmatization were performed to clean the text for analysis.
3. Exploratory Data Analysis (EDA)
Analyzed the distribution of reviews based on rating.
Performed visualizations to understand patterns and trends in customer reviews.
4. Sentiment Classification
Implemented NLP techniques such as Term Frequency-Inverse Document Frequency (TF-IDF) and vectorization to convert text data into machine-readable format.
Applied supervised machine learning models, including:
Logistic Regression
Random Forest Classifier
Naive Bayes
Evaluated model performance using metrics such as accuracy, precision, recall, and F1-score.
5. Model Deployment
The final sentiment classifier was deployed for real-time sentiment prediction on new reviews.

## Technologies Used
1. Python: Core programming language used for data manipulation and model building.
2. Pandas: For data manipulation and analysis.
3. NumPy: To handle large datasets efficiently.
4. Scikit-learn: For machine learning models and evaluation metrics.
5. NLTK & SpaCy: Libraries used for Natural Language Processing (NLP).
6. Matplotlib & Seaborn: For data visualization.
7. Jupyter Notebook: For developing and documenting the code.

## Data Preprocessing
1. Text Cleaning: Removed punctuation, special characters, and non-alphanumeric text.
2. Tokenization: Split the reviews into individual words for analysis.
3. Stopwords Removal: Common words like "the", "and", "is" were removed using NLTK.
4. Stemming & Lemmatization: Reduced words to their root forms.

## Sentiment Analysis
After preprocessing the review data, various machine learning models were trained to classify the sentiment. The best-performing model was selected based on accuracy and other performance metrics. Below are the steps involved:
1,. TF-IDF Vectorization: Transformed text data into numerical format.
2. Model Training: Logistic Regression and Naive Bayes were implemented to classify sentiment.
3. Performance Evaluation: Accuracy of 85% was achieved using Logistic Regression with a 0.82 F1-score.

## Key Insights
1. Positive Reviews: Most customers were satisfied with their purchases, particularly in the electronics and apparel categories.
2. Negative Reviews: Common complaints related to late delivery and product quality.
3. Neutral Reviews: Reviews that didn’t express strong opinions either way.

## Conclusion
This project demonstrates the effective use of NLP and machine learning to extract meaningful insights from Amazon customer reviews. By identifying customer sentiment, businesses can take action to improve customer satisfaction and retention.

## Contributing
Contributions are always welcome! If you have suggestions for improving the project or wish to report any issues, feel free to create an issue or submit a pull request.
