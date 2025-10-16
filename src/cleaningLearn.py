# import pandas as pd
# import re
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sentence_transformers import SentenceTransformer
# import torch

# # to use gpu for mac (mps -> metal performance shaders)
# device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")

# # use this if your pc has nvidia gpu
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the training and test datasets from CSV files
# train = pd.read_csv("dataset/train.csv")
# test = pd.read_csv("dataset/test.csv")

# # Function to clean text by converting to lowercase, stripping spaces, and removing punctuation
# def clean_text(text):
#     text = str(text).lower().strip()  # convert to lowercase and remove leading/trailing spaces
#     text = re.sub(r'[^\w\s]', '', text)  # remove all punctuation
#     text = re.sub(r'\s+', ' ', text)  # replace multiple spaces with a single space
#     return text

# # Apply text cleaning to the 'catalog_content' column
# train['clean_catalog_content'] = train['catalog_content'].apply(clean_text)
# test['clean_catalog_content'] = test['catalog_content'].apply(clean_text)

# # Clean image links - here we are removing extra spaces only
# train['image_link'] = train['image_link'].apply(lambda x: str(x).strip())
# test['image_link'] = test['image_link'].apply(lambda x: str(x).strip())

# # Function to extract the first number from text as item pack quantity, default to 1 if not found
# # def extract_ipq(text):
# #     match = re.search(r'(\d+)', text)  # search for the first number
# #     return int(match.group(1)) if match else 1
# def extract_ipq(text):
#     numbers = re.findall(r'\d+', text) # search for the greatest number
#     return max(map(int, numbers)) if numbers else 1

# # Apply IPQ extraction to the 'catalog_content' column
# train['ipq'] = train['catalog_content'].apply(extract_ipq)
# test['ipq'] = test['catalog_content'].apply(extract_ipq)

# # List of numeric columns to clean
# # numeric_cols = ['price']

# # Clean numeric columns by removing commas, dollar signs, converting to float, and filling missing values
# # for col in numeric_cols:
#     # train[col] = train[col].replace('[\$,]', '', regex=True).astype(float)
#     # test[col] = test[col].replace('[\$,]', '', regex=True).astype(float)
#     # train[col] = train[col].replace(r'[\$,]', '', regex=True).astype(float)
#     # test[col] = test[col].replace(r'[\$,]', '', regex=True).astype(float)
# numeric_cols = ['price']

# for col in numeric_cols:
#     if col in train.columns:
#         train[col] = train[col].replace(r'[\$,]', '', regex=True).astype(float)
#         train[col].fillna(train[col].median(), inplace=True)
#     if col in test.columns:
#         test[col] = test[col].replace(r'[\$,]', '', regex=True).astype(float)
#         test[col].fillna(test[col].median(), inplace=True)

    
#     train[col].fillna(train[col].median(), inplace=True)
#     test[col].fillna(test[col].median(), inplace=True) 

# # Convert cleaned text into TF-IDF vectors (term frequency)
# tfidf = TfidfVectorizer(max_features=5000)
# X_train_tfidf = tfidf.fit_transform(train['clean_catalog_content'])  # fit on training data
# X_test_tfidf = tfidf.transform(test['clean_catalog_content'])  # transform test data

# # Generate BERT embeddings using SentenceTransformer (optional)
# bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
# X_train_bert = bert_model.encode(train['clean_catalog_content'].tolist(), batch_size=64, show_progress_bar=True)
# X_test_bert = bert_model.encode(test['clean_catalog_content'].tolist(), batch_size=64, show_progress_bar=True)

# # Combine TF-IDF vectors with extracted numeric feature (IPQ) for ML
# X_train_final = np.hstack([X_train_tfidf.toarray(), train['ipq'].values.reshape(-1, 1)])
# X_test_final = np.hstack([X_test_tfidf.toarray(), test['ipq'].values.reshape(-1, 1)])

# # Save the cleaned and preprocessed datasets to new CSV files
# train.to_csv("dataset/train_cleaned.csv", index=False)
# test.to_csv("dataset/test_cleaned.csv", index=False)


import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch

# to use gpu for mac (mps -> metal performance shaders)
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")

# use this if your pc has nvidia gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training and test datasets from CSV files
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

# Function to clean text by converting to lowercase, stripping spaces, and removing punctuation
def clean_text(text):
    text = str(text).lower().strip()  # convert to lowercase and remove leading/trailing spaces
    text = re.sub(r'[^\w\s]', '', text)  # remove all punctuation
    text = re.sub(r'\s+', ' ', text)  # replace multiple spaces with a single space
    return text

# Apply text cleaning to the 'catalog_content' column
train['clean_catalog_content'] = train['catalog_content'].apply(clean_text)
test['clean_catalog_content'] = test['catalog_content'].apply(clean_text)

# Clean image links - here we are removing extra spaces only
train['image_link'] = train['image_link'].apply(lambda x: str(x).strip())
test['image_link'] = test['image_link'].apply(lambda x: str(x).strip())

# Function to extract the first number from text as item pack quantity, default to 1 if not found
def extract_ipq(text):
    numbers = re.findall(r'\d+', text)  # search for all numbers
    return max(map(int, numbers)) if numbers else 1

# Apply IPQ extraction to the 'catalog_content' column
train['ipq'] = train['catalog_content'].apply(extract_ipq)
test['ipq'] = test['catalog_content'].apply(extract_ipq)

# List of numeric columns to clean
numeric_cols = ['price']

# Clean numeric columns safely
for col in numeric_cols:
    if col in train.columns:
        train[col] = train[col].replace(r'[\$,]', '', regex=True).astype(float)
        train[col] = train[col].fillna(train[col].median())
    if col in test.columns:
        test[col] = test[col].replace(r'[\$,]', '', regex=True).astype(float)
        test[col] = test[col].fillna(train[col].median())  # use train median if test exists

# Convert cleaned text into TF-IDF vectors (term frequency)
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(train['clean_catalog_content'])  # fit on training data
X_test_tfidf = tfidf.transform(test['clean_catalog_content'])  # transform test data

# Generate BERT embeddings using SentenceTransformer
bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
X_train_bert = bert_model.encode(train['clean_catalog_content'].tolist(), batch_size=64, show_progress_bar=True)
X_test_bert = bert_model.encode(test['clean_catalog_content'].tolist(), batch_size=64, show_progress_bar=True)

# Combine TF-IDF vectors with extracted numeric feature (IPQ) for ML
X_train_final = np.hstack([X_train_tfidf.toarray(), train['ipq'].values.reshape(-1, 1)])
X_test_final = np.hstack([X_test_tfidf.toarray(), test['ipq'].values.reshape(-1, 1)])

# Save the cleaned and preprocessed datasets to new CSV files
train.to_csv("dataset/train_cleaned.csv", index=False)
test.to_csv("dataset/test_cleaned.csv", index=False)