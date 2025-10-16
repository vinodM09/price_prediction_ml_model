# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Paths
DATA_PATH = '/content/drive/MyDrive/practice_dataset'
CSV_PATH = DATA_PATH + '/train_cleaned.csv'
TRAIN_IMG_PATH = DATA_PATH + '/dataset'
FEATURES_PATH = '/content/drive/MyDrive/features'

# Load CSV
import pandas as pd
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} entries")
print(df.head())

# Text Features (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)
X_text = tfidf.fit_transform(df['clean_catalog_content'])
print("Text features shape:", X_text.shape)

# Image Features (ResNet50)
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm
import glob
import pickle

model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def get_image_feature(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model_resnet.predict(x, verbose=0)
        return features.flatten()
    except:
        return np.zeros(2048)

# Extract features in batches and save incrementally
BATCH_SIZE = 64
SAVE_EVERY = 1000  # save features every 1000 images
image_features = []

for i in tqdm(range(len(df))):
    img_path = os.path.join(TRAIN_IMG_PATH, f"{i}.jpg")  # filenames: 0.jpg, 1.jpg, ...
    features = get_image_feature(img_path)
    image_features.append(features)

    # Save every SAVE_EVERY images
    if (i + 1) % SAVE_EVERY == 0 or i == len(df) - 1:
        batch_features = np.vstack(image_features)
        
        # Save as .npy
        np.save(os.path.join(FEATURES_PATH, f'image_features_batch_{i//SAVE_EVERY}.npy'), batch_features)
        
        # Save as .pkl (optional)
        with open(os.path.join(FEATURES_PATH, f'image_features_batch_{i//SAVE_EVERY}.pkl'), 'wb') as f:
            pickle.dump(batch_features, f)
        
        image_features = []  # reset for next batch

# Load all saved batches and combine
all_features = []
for f in sorted(glob.glob(os.path.join(FEATURES_PATH, 'image_features_batch_*.npy'))):
    batch = np.load(f)
    all_features.append(batch)

image_features_combined = np.vstack(all_features)
print("Combined image features shape:", image_features_combined.shape)

# --- Scale Image Features ---
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
image_features_scaled = scaler.fit_transform(image_features_combined)

# --- Add ipq feature ---
ipq_values = df[['ipq']].values

# --- Combine Text + Image + ipq Features ---
from scipy.sparse import hstack, csr_matrix
X_combined = hstack([X_text, image_features_scaled, csr_matrix(ipq_values)])
print("Combined feature shape:", X_combined.shape)

# --- Prepare Target ---
y = df['price']

# --- Train/Test Split ---
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# --- Train LGBM Regressor ---
from lightgbm import LGBMRegressor, log_evaluation

model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    device='gpu',  # use GPU in Colab
    verbose=-1
)

# Add a progress bar callback
model.fit(
    X_train, 
    y_train,
    eval_set=[(X_train, y_train)],
    eval_metric='l1',  # MAE
    callbacks=[log_evaluation(period=10)]  # prints progress every 10 iterations
)

# --- Predict & Evaluate ---
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = model.predict(X_test)

for idx in range(min(25, len(y_test))):
    print(f"Sample {idx+1}: Actual Price = {y_test.iloc[idx]}, Predicted Price = {y_pred[idx]:.2f}")

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

import numpy as np

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

score = smape(y_true, y_pred)
print("SMAPE:", score)

# --- Save Model / TF-IDF / Scaler ---
import joblib
joblib.dump(model, '/content/drive/MyDrive/models/lgbm_model.pkl')
joblib.dump(tfidf, '/content/drive/MyDrive/models/tfidf.pkl')
joblib.dump(scaler, '/content/drive/MyDrive/models/image_scaler.pkl')