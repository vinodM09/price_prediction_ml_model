import pandas as pd

train = pd.read_csv("dataset/train.csv") # this read the csv and converts it to DataFrame (2D labeled data structure)
test = pd.read_csv("dataset/test.csv")

print(train.head()) # head() print first 5 rows only
print(train['price'].describe()) #  take the price column and describe it (count, mean etc.)