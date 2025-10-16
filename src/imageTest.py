# import cv2
# import numpy as np
# import urllib.request


# url = "https://m.media-amazon.com/images/I/61Ea735uaML.jpg"


# resp = urllib.request.urlopen(url)
# image_data = np.asarray(bytearray(resp.read()), dtype="uint8")

# img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)


# cv2.imshow("Amazon Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("Pixel value at (0,0):", img[0, 0])   # [Blue, Green, Red]

# # Required Libraries
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

# # Step 1: Sample Data
# data = [
#     {
#         "sample_id": 33127,
#         "catalog_content": "Item Name: La Victoria Green Taco Sauce Mild, 12 Ounce (Pack of 6)\nValue: 72.0\nUnit: Fl Oz",
#         "image_link": "https://m.media-amazon.com/images/I/51mo8htwTHL.jpg",
#         "price": 4.89
#     },
#     {
#         "sample_id": 198967,
#         "catalog_content": """Item Name: Salerno Cookies, The Original Butter Cookies, 8 Ounce (Pack of 4)
# Bullet Point 1: Original Butter Cookies: Classic butter cookies made with real butter
# Bullet Point 2: Variety Pack: Includes 4 boxes with 32 cookies total
# Bullet Point 3: Occasion Perfect: Delicious cookies for birthdays, weddings, anniversaries
# Bullet Point 4: Shareable Treats: Fun to give and enjoy with friends and family
# Bullet Point 5: Salerno Brand: Trusted brand of delicious butter cookies since 1925""",
#         "image_link": "https://m.media-amazon.com/images/I/51mo8htwTHL.jpg",
#         "price": 4.89
#     }
# ]


# df = pd.DataFrame(data)

# # Step 2: Extract numerical and categorical features from catalog_content
# def extract_value_unit(text):
#     value, unit = None, None
#     for line in text.split("\n"):
#         if "Value:" in line:
#             value = float(line.split("Value:")[1].strip())
#         if "Unit:" in line:
#             unit = line.split("Unit:")[1].strip()
#     return pd.Series([value, unit])

# df[['Value', 'Unit']] = df['catalog_content'].apply(extract_value_unit)

# # Step 3: Prepare features
# X = df[['catalog_content', 'Unit', 'Value']]
# y = df['price']

# # Step 4: Preprocessing pipeline
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('text', TfidfVectorizer(), 'catalog_content'),
#         ('unit', OneHotEncoder(), ['Unit']),
#         ('value', 'passthrough', ['Value'])
#     ]
# )

# # Step 5: Regression pipeline
# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
# ])

# # Step 6: Train the model
# model.fit(X, y)

# # Step 7: Predict
# predictions = model.predict(X)
# df['predicted_price'] = predictions


# print(df[['sample_id', 'price', 'predicted_price']])


# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

# # Step 1: Load train.csv
# df = pd.read_csv("dataset/train_cleaned.csv")

# # Step 2: Take only first 1000 entries
# df = df.head(1000)

# # Step 3: Extract numerical and categorical features from catalog_content
# def extract_value_unit(text):
#     value, unit = None, None
#     for line in str(text).split("\n"):
#         if "Value:" in line:
#             try:
#                 value = float(line.split("Value:")[1].strip())
#             except:
#                 value = None
#         if "Unit:" in line:
#             unit = line.split("Unit:")[1].strip()
#     return pd.Series([value, unit])

# df[['Value', 'Unit']] = df['catalog_content'].apply(extract_value_unit)

# # Step 4: Prepare features and target
# X = df[['catalog_content', 'Unit', 'Value']]
# y = df['price']

# # Step 5: Preprocessing pipeline
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('text', TfidfVectorizer(), 'catalog_content'),
#         ('unit', OneHotEncoder(handle_unknown='ignore'), ['Unit']),
#         ('value', 'passthrough', ['Value'])
#     ]
# )

# # Step 6: Regression pipeline
# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
# ])

# # Step 7: Train the model
# model.fit(X, y)

# # Step 8: Predict
# df['predicted_price'] = model.predict(X)

# # Step 9: Save output to CSV
# df[['sample_id', 'price', 'predicted_price']].to_csv("first_predicted.csv", index=False)

# print("Predictions saved to first_predicted.csv")

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load first 1000 rows
df = pd.read_csv("dataset/train_cleaned.csv").head(1000)

# Extract Value and Unit
def extract_value_unit(text):
    value, unit = None, None
    for line in str(text).split("\n"):
        if "Value:" in line:
            try:
                value = float(line.split("Value:")[1].strip())
            except:
                value = None
        if "Unit:" in line:
            unit = line.split("Unit:")[1].strip()
    return pd.Series([value, unit])

df[['Value', 'Unit']] = df['catalog_content'].apply(extract_value_unit)

X = df[['catalog_content', 'Unit', 'Value']]
y = df['price']

# Preprocessing pipeline with imputers
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'catalog_content'),
        ('unit', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ['Unit']),
        ('value', SimpleImputer(strategy='mean'), ['Value'])
    ]
)

# Regression pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
])

# Train
model.fit(X, y)

# Predict
df['predicted_price'] = model.predict(X)

# Save output
df[['sample_id', 'price', 'predicted_price']].to_csv("first_predicted.csv", index=False)