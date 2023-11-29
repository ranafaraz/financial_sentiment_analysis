# Financial Sentiment Analysis using Machine Learning Blending Approach
Repository for Financial Sentiment Analysis Using Blending Machine Learning Approach

## 1. Setting Up the Environment and Data Loading

Google Colab and Drive Integration: The script starts by mounting Google Drive to access data stored there. This is useful when working with Colab notebooks.
Importing Libraries: A range of libraries are imported for data manipulation (pandas, numpy), text processing (re, nltk), machine learning (sklearn, imblearn), and visualization (matplotlib, seaborn).
Loading Data: The dataset is loaded from a CSV file using pandas.
## 2. Preprocessing and Text Analysis

Custom Function for Master Dictionary: A function load_masterdictionary is created to load a sentiment dictionary, which can be useful for sentiment-specific text analysis.
Text Preprocessing: Various text preprocessing steps are applied, including lowercasing, HTML tag removal, tokenization, lemmatization, and custom stopwords handling.
SentencePiece for Tokenization: The script uses SentencePiece for advanced tokenization, which can be particularly effective for languages with no clear word boundaries.
Number to Word Conversion: A function is used to convert numerical values to words, which can be important in financial texts where numbers play a significant role.
## 3. Feature Extraction and Dataset Balancing

TF-IDF Vectorization: The script applies TF-IDF vectorization to transform the text data into numerical format.
Word2Vec Embedding: Gensim’s Word2Vec model is trained to create word embeddings, offering a dense representation of words based on their context.
Balancing with SMOTE: The dataset is balanced using SMOTE (Synthetic Minority Over-sampling Technique), addressing class imbalance issues.
Data Splitting: The dataset is split into training and testing sets.

## 4. Model Training and Blending
Classifier Initialization: Various classifiers like Logistic Regression, SVM, Naive Bayes, and others are initialized.
Blended Classifier Training: A VotingClassifier from sklearn is used for blending different models. It's trained on the combined feature set (TF-IDF and Word2Vec).
Evaluation Metrics: The script evaluates the blended model using metrics like accuracy, precision, recall, F1-score, and confusion matrix.

## 5. Results Handling and Logging
Resource Utilization: The script records CPU and memory usage during the model training, which is valuable for understanding computational costs.
Result Storage: The results are stored in a DataFrame and saved to a CSV file, facilitating easy analysis and comparison of different models.

## 6. Word2Vec Model Training and Evaluation
Training Word2Vec: The script trains a Word2Vec model on the preprocessed sentences and evaluates the embeddings by calculating the loss and examining the vocabulary.
Combining Features: TF-IDF and Word2Vec features are combined to form a comprehensive feature set for training classifiers.
