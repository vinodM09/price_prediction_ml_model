import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
from scipy.sparse import hstack

# Step 1: Read CSV file
df = pd.read_csv("dataset/train_cleaned.csv")  # replace with your file path

df = df.head(500)

# Step 2: Image Feature Extraction
# model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model_resnet = ResNet50(
    weights='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
    include_top=False,
    pooling='avg'
)

def get_image_feature(img_url):
    try:
        response = requests.get(img_url)
        img = image.load_img(BytesIO(response.content), target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model_resnet.predict(x, verbose=0)
        return features.flatten()
    except:
        return np.zeros(2048)  # fallback if image fails

print("Extracting image features...")
image_features = np.array([get_image_feature(url) for url in df['image_link']])
print("Image features done.")

# Step 3: Prepare text + numerical + categorical features
X_text_num = df[['clean_catalog_content', 'ipq']]
y = df['price']

# Step 4: Preprocessing pipeline for text + categorical + numerical
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'clean_catalog_content'),
        # ('unit', OneHotEncoder(), ['Unit']),
        ('num', 'passthrough', ['ipq'])
    ]
)

# Fit transform on text+num features
X_processed = preprocessor.fit_transform(X_text_num)

# Step 5: Combine image features with processed features
X_combined = hstack([X_processed, image_features])  # sparse + dense

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.5, random_state=42)

# Step 7: Train RandomForest Regressor
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Step 8: Predict prices
y_pred = model.predict(X_test)

# Step 9: Show results
results = pd.DataFrame({
    'actual_price': y_test.values,
    'predicted_price': y_pred
})
print(results)