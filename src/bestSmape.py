### best smape 51.9
# # Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# # Paths
# DATA_PATH = '/content/drive/MyDrive/dataset'
# CSV_PATH = DATA_PATH + '/train_cleaned.csv'
# FEATURES_PATH = '/content/drive/MyDrive/LGBMFeatures'
# MODEL_PATH = '/content/drive/MyDrive/models'

# # Load CSV
# import pandas as pd
# df = pd.read_csv(CSV_PATH)
# print(f"Loaded {len(df)} entries")
# print(df.head())

# # Text Features (TF-IDF)
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(max_features=3000)
# X_text = tfidf.fit_transform(df['clean_catalog_content'])
# print("Text features shape:", X_text.shape)

# # Load saved image feature batches (.npy files)
# import glob
# import numpy as np
# all_features = []
# for f in sorted(glob.glob(FEATURES_PATH + '/image_features_batch_*.npy')):
#     batch = np.load(f)
#     all_features.append(batch)

# image_features_combined = np.vstack(all_features)
# print("Loaded image features shape:", image_features_combined.shape)

# # Scale image features
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler(with_mean=False)
# image_features_scaled = scaler.fit_transform(image_features_combined)

# # Add ipq feature
# ipq_values = df[['ipq']].values

# # Combine everything
# from scipy.sparse import hstack, csr_matrix
# X_combined = hstack([X_text, image_features_scaled, csr_matrix(ipq_values)])
# print("Combined feature shape:", X_combined.shape)

# # Target with log transform
# import numpy as np
# y = np.log1p(df['price'])

# # Train/Test Split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_combined, y, test_size=0.2, random_state=42
# )

# # Train LightGBM
# from lightgbm import LGBMRegressor, log_evaluation

# model = LGBMRegressor(
#     n_estimators=2000,
#     learning_rate=0.03,
#     num_leaves=128,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     device='gpu',
#     verbose=-1
# )

# model.fit(
#     X_train,
#     y_train,
#     eval_set=[(X_test, y_test)],
#     eval_metric='l1',
#     callbacks=[log_evaluation(period=50)],
#     early_stopping_rounds=100
# )

# # Predict & Evaluate
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# y_pred_log = model.predict(X_test)
# y_pred = np.expm1(y_pred_log)
# y_test_original = np.expm1(y_test)

# for idx in range(min(25, len(y_test_original))):
#     print(f"Sample {idx+1}: Actual Price = {y_test_original.iloc[idx]}, Predicted Price = {y_pred[idx]:.2f}")

# mae = mean_absolute_error(y_test_original, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
# r2 = r2_score(y_test_original, y_pred)

# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R2 Score:", r2)

# def smape(y_true, y_pred):
#     numerator = np.abs(y_pred - y_true)
#     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
#     return np.mean(numerator / denominator) * 100

# score = smape(y_test_original, y_pred)
# print("SMAPE:", score)

# # Save updated model & objects
# import joblib
# joblib.dump(model, MODEL_PATH + '/lgbm_model_improved.pkl')
# joblib.dump(tfidf, MODEL_PATH + '/tfidf_improved.pkl')
# joblib.dump(scaler, MODEL_PATH + '/image_scaler_improved.pkl')
# print("Model, TF-IDF, and Scaler saved successfully!")




# Fine-Tuning Existing LGBM Model: TF-IDF + Image Features + IPQ

from google.colab import drive
drive.mount('/content/drive')

DATA_PATH = '/content/drive/MyDrive/dataset'
CSV_PATH = DATA_PATH + '/train_cleaned.csv'
FEATURES_PATH = '/content/drive/MyDrive/LGBMFeatures'
MODEL_PATH = '/content/drive/MyDrive/models'

import pandas as pd
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} entries")
print(df.head())

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load TF-IDF vectorizer
tfidf = joblib.load(MODEL_PATH + '/tfidf_improved_52,70.pkl')
X_text = tfidf.transform(df['clean_catalog_content'])
print("Text features shape:", X_text.shape)

# Load saved image feature batches
import glob
import numpy as np
all_features = []
for f in sorted(glob.glob(FEATURES_PATH + '/image_features_batch_*.npy')):
    batch = np.load(f)
    all_features.append(batch)

image_features_combined = np.vstack(all_features)
print("Loaded image features shape:", image_features_combined.shape)

# Scale image features
from sklearn.preprocessing import StandardScaler
scaler = joblib.load(MODEL_PATH + '/image_scaler_improved_52.70.pkl')
image_features_scaled = scaler.transform(image_features_combined)

# Add IPQ feature
from scipy.sparse import hstack, csr_matrix
ipq_values = df[['ipq']].values

# Combine features
X_combined = hstack([X_text, image_features_scaled, csr_matrix(ipq_values)])
print("Combined feature shape:", X_combined.shape)

# Target
y = np.log1p(df['price'])

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

# Load existing LightGBM model
from lightgbm import LGBMRegressor, log_evaluation
model = joblib.load(MODEL_PATH + '/lgbm_model_improved_52.70.pkl')

# Update parameters for fine-tuning
model.set_params(
    n_estimators=model.n_estimators + 1000,
    learning_rate=0.02,
    num_leaves=150
)

# Continue training
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='l1',
    callbacks=[log_evaluation(period=50)],
    early_stopping_rounds=100,
    init_model=model
)

# Predict and evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

for idx in range(min(25, len(y_test_original))):
    print(f"Sample {idx+1}: Actual = {y_test_original.iloc[idx]}, Predicted = {y_pred[idx]:.2f}")

mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

score = smape(y_test_original, y_pred)
print("SMAPE:", score)

# Save updated objects
joblib.dump(model, MODEL_PATH + '/lgbm_model_finetuned_52.70.pkl')
joblib.dump(tfidf, MODEL_PATH + '/tfidf_improved_52,70.pkl')
joblib.dump(scaler, MODEL_PATH + '/image_scaler_improved_52.70.pkl')

print("Fine-tuned model, TF-IDF, and Scaler saved successfully!")