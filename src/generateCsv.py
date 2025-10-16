# # Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# DATA_PATH = '/content/drive/MyDrive/dataset'
# CSV_PATH = DATA_PATH + '/train_cleaned.csv'
# TRAIN_IMG_PATH = DATA_PATH + '/train'
# FEATURES_PATH = '/content/drive/MyDrive/LGBMFeatures'

# # --- Load test CSV ---
# TEST_CSV_PATH = DATA_PATH + '/test_cleaned.csv'
# df_test = pd.read_csv(TEST_CSV_PATH)
# print(f"Loaded {len(df_test)} test entries")

# # --- Load saved model, TF-IDF, scaler ---
# import joblib
# model = joblib.load('/content/drive/MyDrive/models/lgbm_model.pkl')
# tfidf = joblib.load('/content/drive/MyDrive/models/tfidf.pkl')
# scaler = joblib.load('/content/drive/MyDrive/models/image_scaler.pkl')

# # --- Extract text features ---
# X_test_text = tfidf.transform(df_test['clean_catalog_content'])

# # --- Extract image features ---
# image_features_test = []
# for i in tqdm(range(len(df_test))):
#     img_path = os.path.join(TRAIN_IMG_PATH, f"{i}.jpg")  # adjust path & naming if needed
#     features = get_image_feature(img_path)
#     image_features_test.append(features)

# image_features_test = np.vstack(image_features_test)
# image_features_test_scaled = scaler.transform(image_features_test)

# # --- Add ipq feature ---
# ipq_test_values = df_test[['ipq']].values

# # --- Combine features ---
# from scipy.sparse import hstack, csr_matrix
# X_test_combined = hstack([X_test_text, image_features_test_scaled, csr_matrix(ipq_test_values)])

# # --- Predict ---
# y_test_pred = model.predict(X_test_combined)

# # --- Prepare CSV ---
# submission = pd.DataFrame({
#     'sample_id': df_test['sample_id'],
#     'price': y_test_pred
# })

# # --- Save CSV ---
# submission.to_csv('/content/drive/MyDrive/test_out.csv', index=False)
# print("Saved test_out.csv")


# # Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

DATA_PATH = '/content/drive/MyDrive/dataset'
CSV_PATH = DATA_PATH + '/train_cleaned.csv'
TRAIN_IMG_PATH = DATA_PATH + '/train'
FEATURES_PATH = '/content/drive/MyDrive/LGBMFeatures'

# --- Load test CSV ---
TEST_CSV_PATH = DATA_PATH + '/test_cleaned.csv'
df_test_full = pd.read_csv(TEST_CSV_PATH)
print(f"Loaded {len(df_test_full)} test entries")

# --- Take first 25k entries ---
df_test = df_test_full.head(25000)
print(f"Using first {len(df_test)} entries for prediction")
df_test.head()

# --- Load saved model, TF-IDF, scaler ---
import joblib
model = joblib.load('/content/drive/MyDrive/models/lgbm_model.pkl')
tfidf = joblib.load('/content/drive/MyDrive/models/tfidf.pkl')
scaler = joblib.load('/content/drive/MyDrive/models/image_scaler.pkl')

# --- Extract text features ---
X_test_text = tfidf.transform(df_test['clean_catalog_content'])

# --- Extract image features ---
image_features_test = []
for i in tqdm(range(len(df_test))):
    img_path = os.path.join(TRAIN_IMG_PATH, f"{i}.jpg")  # adjust path & naming if needed
    features = get_image_feature(img_path)
    image_features_test.append(features)

image_features_test = np.vstack(image_features_test)
image_features_test_scaled = scaler.transform(image_features_test)

# --- Add ipq feature ---
ipq_test_values = df_test[['ipq']].values

# --- Combine features ---
from scipy.sparse import hstack, csr_matrix
X_test_combined = hstack([X_test_text, image_features_test_scaled, csr_matrix(ipq_test_values)])

# --- Predict ---
y_test_pred = model.predict(X_test_combined)

# --- Prepare CSV ---
submission = pd.DataFrame({
    'sample_id': df_test['sample_id'],
    'price': y_test_pred
})

# --- Save CSV ---
submission.to_csv('/content/drive/MyDrive/test_out_25k.csv', index=False)
print("Saved test_out_25k.csv")