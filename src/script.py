import pandas as pd
from preprocess import preprocess
import sys
sys.path.append('src')

# Load training data
df_train = pd.read_csv("dataset/train.csv")

# Preprocess
df_train = preprocess(df_train, image_folder="images")

# Inspect
print(df_train.head())
print(df_train[['ipq','char_len','word_len','num_digits','num_upper','image_missing']].describe())