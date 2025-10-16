# from google.colab import drive
# drive.mount('/content/drive')

# DATA_PATH = '/content/drive/MyDrive/practice_dataset'  # change this to your folder
# CSV_PATH = DATA_PATH + '/train_cleaned.csv'
# TRAIN_IMG_PATH = DATA_PATH + '/dataset(1000)'

# import pandas as pd
# df = pd.read_csv(CSV_PATH)
# print(f"Loaded {len(df)} entries")
# print(df.head())

# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(max_features=5000)
# X_text = tfidf.fit_transform(df['catalog_content'])  # change column name if needed
# print("Text features shape:", X_text.shape)

# import numpy as np
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing import image
# from tqdm import tqdm  # nice progress bar
# import os
# import pandas as pd


# model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# def get_image_feature(img_path):
#     try:
#         img = image.load_img(img_path, target_size=(224,224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         features = model_resnet.predict(x, verbose=0)
#         return features.flatten()
#     except:
#         return np.zeros(2048)  # fallback if image fails

# # Process images in batches to save memory
# BATCH_SIZE = 500
# image_features = []

# for i in tqdm(range(0, len(df), BATCH_SIZE)):
#     batch = df.iloc[i:i+BATCH_SIZE]
#     for img_file_id in batch['sample_id']:  # change column if needed
#         img_path = os.path.join(TRAIN_IMG_PATH, f"{img_file_id}.jpg") # Assuming image file names are sample_id.jpg
#         feat = get_image_feature(img_path)
#         image_features.append(feat)

# image_features = np.array(image_features)
# print("Image features shape:", image_features.shape)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler(with_mean=False)
# image_features_scaled = scaler.fit_transform(image_features)


# from scipy.sparse import hstack
# X_combined = hstack([X_text, image_features_scaled])
# print("Combined feature shape:", X_combined.shape)

# y = df['price']  # change target column if needed

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)


# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)

# from sklearn.metrics import mean_absolute_error, mean_squared_error
# y_pred = model.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# print("MAE:", mae)
# print("RMSE:", rmse)

# np.save('/content/drive/MyDrive/features/image_features.npy', image_features_scaled)
# import joblib
# joblib.dump(model, '/content/drive/MyDrive/models/random_forest_model.pkl')


# # Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# # GPU Check
# # import tensorflow as tf
# # if tf.test.gpu_device_name():
# #     print("GPU detected:", tf.test.gpu_device_name())
# # else:
# #     print("No GPU found; enable GPU Runtime in Colab")

# # Paths
# # DATA_PATH = '/content/drive/MyDrive/dataset/'
# # CSV_PATH = DATA_PATH + 'train.csv'
# # TRAIN_IMG_PATH = DATA_PATH + 'train_images/'
# DATA_PATH = '/content/drive/MyDrive/practice_dataset'  # change this to your folder
# CSV_PATH = DATA_PATH + '/train_cleaned.csv'
# TRAIN_IMG_PATH = DATA_PATH + '/dataset'

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

# # Image Features (ResNet50)
# import numpy as np
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing import image
# from tqdm import tqdm
# import os

# model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# def get_image_feature(img_path):
#     try:
#         img = image.load_img(img_path, target_size=(224, 224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         features = model_resnet.predict(x, verbose=0)
#         return features.flatten()
#     except Exception as e:
#         print(f"Failed image: {img_path} - {e}")
#         return np.zeros(2048)

# def get_batch_features(image_paths):
#     batch_imgs = []
#     for img_path in image_paths:
#         try:
#             img = image.load_img(img_path, target_size=(224,224))
#             x = image.img_to_array(img)
#             batch_imgs.append(x)
#         except:
#             batch_imgs.append(np.zeros((224,224,3)))

#     batch_imgs = np.array(batch_imgs)
#     batch_imgs = preprocess_input(batch_imgs)
#     features = model_resnet.predict(batch_imgs, verbose=0)
#     return features

# BATCH_SIZE = 64
# image_features = []

# for i in tqdm(range(0, len(df), BATCH_SIZE)):
#     batch = df.iloc[i:i + BATCH_SIZE]
#     paths = [os.path.join(TRAIN_IMG_PATH, f) for f in batch['image_link']]
#     features = get_batch_features(paths)
#     image_features.append(features)

# image_features = np.vstack(image_features)
# print("Image features shape:", image_features.shape)

# # Scale Image Features
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler(with_mean=False)
# image_features_scaled = scaler.fit_transform(image_features)

# # Add ipq feature
# ipq_values = df[['ipq']].values

# # Combine Text + Image + ipq Features
# from scipy.sparse import hstack, csr_matrix
# X_combined = hstack([X_text, image_features_scaled, csr_matrix(ipq_values)])
# print("Combined feature shape:", X_combined.shape)

# # Prepare Target
# y = df['price']

# # Train/Test Split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# # Train Random Forest
# # from sklearn.ensemble import RandomForestRegressor
# # model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
# # model.fit(X_train, y_train)
# # Using LGBM Regressor instead
# from lightgbm import LGBMRegressor
# from lightgbm import log_evaluation

# model = LGBMRegressor(
#     n_estimators=500,
#     learning_rate=0.05,
#     num_leaves=64,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     device='gpu',  # use GPU
#     verbose=-1  # suppress default LightGBM output
# )

# # Add a progress bar callback
# model.fit(
#     X_train, 
#     y_train,
#     eval_set=[(X_train, y_train)],
#     eval_metric='l1',  # mean absolute error
#     callbacks=[log_evaluation(period=10)]  # prints progress every 10 iterations
# )

# # Predict & Evaluate
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# y_pred = model.predict(X_test)

# # Show predictions for first 25 rows in test split
# for idx in range(min(25, len(y_test))):
#     print(f"Sample {idx+1}: Actual Price = {y_test.iloc[idx]}, Predicted Price = {y_pred[idx]:.2f}")

# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R2 Score:", r2)

# # Save Features / Model
# np.save('/content/drive/MyDrive/features/image_features.npy', image_features_scaled)
# import joblib
# joblib.dump(model, '/content/drive/MyDrive/models/random_forest_model.pkl')
# joblib.dump(tfidf, '/content/drive/MyDrive/models/tfidf.pkl')
# joblib.dump(scaler, '/content/drive/MyDrive/models/image_scaler.pkl')

# first train
# # Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# # Paths
# DATA_PATH = '/content/drive/MyDrive/practice_dataset'
# CSV_PATH = DATA_PATH + '/train_cleaned.csv'
# TRAIN_IMG_PATH = DATA_PATH + '/dataset'
# FEATURES_PATH = '/content/drive/MyDrive/features'

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

# # Image Features (ResNet50)
# import os
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tqdm import tqdm
# import glob
# import pickle

# model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# def get_image_feature(img_path):
#     try:
#         img = image.load_img(img_path, target_size=(224, 224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         features = model_resnet.predict(x, verbose=0)
#         return features.flatten()
#     except:
#         return np.zeros(2048)

# # Extract features in batches and save incrementally
# BATCH_SIZE = 64
# SAVE_EVERY = 1000  # save features every 1000 images
# image_features = []

# for i in tqdm(range(len(df))):
#     img_path = os.path.join(TRAIN_IMG_PATH, f"{i}.jpg")  # filenames: 0.jpg, 1.jpg, ...
#     features = get_image_feature(img_path)
#     image_features.append(features)

#     # Save every SAVE_EVERY images
#     if (i + 1) % SAVE_EVERY == 0 or i == len(df) - 1:
#         batch_features = np.vstack(image_features)
        
#         # Save as .npy
#         np.save(os.path.join(FEATURES_PATH, f'image_features_batch_{i//SAVE_EVERY}.npy'), batch_features)
        
#         # Save as .pkl (optional)
#         with open(os.path.join(FEATURES_PATH, f'image_features_batch_{i//SAVE_EVERY}.pkl'), 'wb') as f:
#             pickle.dump(batch_features, f)
        
#         image_features = []  # reset for next batch

# # Load all saved batches and combine
# all_features = []
# for f in sorted(glob.glob(os.path.join(FEATURES_PATH, 'image_features_batch_*.npy'))):
#     batch = np.load(f)
#     all_features.append(batch)

# image_features_combined = np.vstack(all_features)
# print("Combined image features shape:", image_features_combined.shape)

# # --- Scale Image Features ---
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler(with_mean=False)
# image_features_scaled = scaler.fit_transform(image_features_combined)

# # --- Add ipq feature ---
# ipq_values = df[['ipq']].values

# # --- Combine Text + Image + ipq Features ---
# from scipy.sparse import hstack, csr_matrix
# X_combined = hstack([X_text, image_features_scaled, csr_matrix(ipq_values)])
# print("Combined feature shape:", X_combined.shape)

# # --- Prepare Target ---
# y = df['price']

# # --- Train/Test Split ---
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# # --- Train LGBM Regressor ---
# from lightgbm import LGBMRegressor, log_evaluation

# model = LGBMRegressor(
#     n_estimators=500,
#     learning_rate=0.05,
#     num_leaves=64,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     device='gpu',  # use GPU in Colab
#     verbose=-1
# )

# # Add a progress bar callback
# model.fit(
#     X_train, 
#     y_train,
#     eval_set=[(X_train, y_train)],
#     eval_metric='l1',  # MAE
#     callbacks=[log_evaluation(period=10)]  # prints progress every 10 iterations
# )

# # --- Predict & Evaluate ---
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# y_pred = model.predict(X_test)

# for idx in range(min(25, len(y_test))):
#     print(f"Sample {idx+1}: Actual Price = {y_test.iloc[idx]}, Predicted Price = {y_pred[idx]:.2f}")

# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R2 Score:", r2)

# # --- Save Model / TF-IDF / Scaler ---
# import joblib
# joblib.dump(model, '/content/drive/MyDrive/models/lgbm_model.pkl')
# joblib.dump(tfidf, '/content/drive/MyDrive/models/tfidf.pkl')
# joblib.dump(scaler, '/content/drive/MyDrive/models/image_scaler.pkl')



# train again
# # ---------------- Block 1: Mount Google Drive and Paths ----------------
# from google.colab import drive
# drive.mount('/content/drive')

# # Paths
# DATA_PATH = '/content/drive/MyDrive/dataset'
# CSV_PATH = DATA_PATH + '/train_cleaned.csv'
# TRAIN_IMG_PATH = DATA_PATH + '/dataset'
# FEATURES_PATH = '/content/drive/MyDrive/LGBMFeatures'
# MODEL_PATH = '/content/drive/MyDrive/models'

# # ---------------- Block 2: Load CSV ----------------
# import pandas as pd
# df = pd.read_csv(CSV_PATH)
# print(f"Loaded {len(df)} entries")
# df.head()

# # ---------------- Block 3: Extract Text Features ----------------
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(max_features=3000)
# X_text = tfidf.fit_transform(df['clean_catalog_content'])
# print("Text features shape:", X_text.shape)

# # ---------------- Block 4: Extract Image Features in Batches ----------------
# import os
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tqdm import tqdm
# import glob
# import pickle

# model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# def get_image_feature(img_path):
#     try:
#         img = image.load_img(img_path, target_size=(224, 224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         features = model_resnet.predict(x, verbose=0)
#         return features.flatten()
#     except:
#         return np.zeros(2048)

# # Batch extraction
# BATCH_SIZE = 64
# SAVE_EVERY = 1000
# image_features = []

# for i in tqdm(range(len(df))):
#     img_path = os.path.join(TRAIN_IMG_PATH, f"{i}.jpg")
#     features = get_image_feature(img_path)
#     image_features.append(features)

#     if (i + 1) % SAVE_EVERY == 0 or i == len(df) - 1:
#         batch_features = np.vstack(image_features)
#         # Save batch
#         np.save(os.path.join(FEATURES_PATH, f'image_features_batch_{i//SAVE_EVERY}.npy'), batch_features)
#         with open(os.path.join(FEATURES_PATH, f'image_features_batch_{i//SAVE_EVERY}.pkl'), 'wb') as f:
#             pickle.dump(batch_features, f)
#         image_features = []

# # ---------------- Block 5: Load All Image Feature Batches ----------------
# all_features = []
# for f in sorted(glob.glob(os.path.join(FEATURES_PATH, 'image_features_batch_*.npy'))):
#     batch = np.load(f)
#     all_features.append(batch)

# image_features_combined = np.vstack(all_features)
# print("Combined image features shape:", image_features_combined.shape)

# # ---------------- Block 6: Scale Image Features ----------------
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler(with_mean=False)
# image_features_scaled = scaler.fit_transform(image_features_combined)

# # ---------------- Block 7: Combine Text + Image + ipq Features ----------------
# from scipy.sparse import hstack, csr_matrix
# ipq_values = df[['ipq']].values
# X_combined = hstack([X_text, image_features_scaled, csr_matrix(ipq_values)])
# print("Combined feature shape:", X_combined.shape)

# # ---------------- Block 8: Prepare Target and Train/Test Split ----------------
# y = df['price']
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_combined, y, test_size=0.2, random_state=42
# )

# # ---------------- Block 9: Train LightGBM on Full Dataset ----------------
# from lightgbm import LGBMRegressor, log_evaluation

# model = LGBMRegressor(
#     n_estimators=1000,
#     learning_rate=0.05,
#     num_leaves=64,
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
#     callbacks=[log_evaluation(period=10)],
#     early_stopping_rounds=50
# )

# # ---------------- Block 10: Predict & Evaluate ----------------
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np

# y_pred = model.predict(X_test)
# for idx in range(min(25, len(y_test))):
#     print(f"Sample {idx+1}: Actual Price = {y_test.iloc[idx]}, Predicted Price = {y_pred[idx]:.2f}")

# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R2 Score:", r2)

# # ---------------- Block 11: Save Model / TF-IDF / Scaler ----------------
# import joblib
# joblib.dump(model, os.path.join(MODEL_PATH, 'lgbm_model_full75k.pkl'))
# joblib.dump(tfidf, os.path.join(MODEL_PATH, 'tfidf_full75k.pkl'))
# joblib.dump(scaler, os.path.join(MODEL_PATH, 'image_scaler_full75k.pkl'))


# best smape
# # Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# # Paths
# DATA_PATH = '/content/drive/MyDrive/dataset'
# CSV_PATH = DATA_PATH + '/train_cleaned.csv'
# FEATURES_PATH = '/content/drive/MyDrive/RGBMFeatures'
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


# ===============================
# Full LGBM Pipeline: TF-IDF + BERT + Image Features + IPQ
# ===============================

# # Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# # -------------------------------
# # Paths
# # -------------------------------
# DATA_PATH = '/content/drive/MyDrive/dataset'
# CSV_PATH = DATA_PATH + '/train_cleaned.csv'
# FEATURES_PATH = '/content/drive/MyDrive/RGBMFeatures'  # precomputed image features
# MODEL_PATH = '/content/drive/MyDrive/models'

# # -------------------------------
# # Load CSV
# # -------------------------------
# import pandas as pd
# df = pd.read_csv(CSV_PATH)
# print(f"Loaded {len(df)} entries")
# print(df.head())

# # -------------------------------
# # Text Features: TF-IDF
# # -------------------------------
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(max_features=3000)
# X_text_tfidf = tfidf.fit_transform(df['clean_catalog_content'])
# print("TF-IDF features shape:", X_text_tfidf.shape)

# # -------------------------------
# # Text Features: BERT Embeddings
# # -------------------------------
# !pip install -q sentence-transformers
# from sentence_transformers import SentenceTransformer
# import numpy as np
# bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # compact & fast for Colab free GPU

# X_text_bert = bert_model.encode(df['clean_catalog_content'].tolist(), batch_size=64, show_progress_bar=True)
# X_text_bert = np.array(X_text_bert)
# print("BERT features shape:", X_text_bert.shape)

# # -------------------------------
# # Combine TF-IDF + BERT (Text)
# # -------------------------------
# from scipy.sparse import hstack, csr_matrix
# X_text_combined = hstack([X_text_tfidf, csr_matrix(X_text_bert)])
# print("Combined text features shape:", X_text_combined.shape)

# # -------------------------------
# # Load saved image feature batches
# # -------------------------------
# import glob
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

# # -------------------------------
# # Add IPQ feature
# # -------------------------------
# ipq_values = df[['ipq']].values

# # -------------------------------
# # Combine all features: Text + Image + IPQ
# # -------------------------------
# X_combined = hstack([X_text_combined, image_features_scaled, csr_matrix(ipq_values)])
# print("Total combined feature shape:", X_combined.shape)

# # -------------------------------
# # Target (log-transform)
# # -------------------------------
# y = np.log1p(df['price'])

# # -------------------------------
# # Train/Test Split
# # -------------------------------
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_combined, y, test_size=0.2, random_state=42
# )

# # -------------------------------
# # Train LGBM
# # -------------------------------
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

# # -------------------------------
# # Predict & Evaluate
# # -------------------------------
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# y_pred_log = model.predict(X_test)
# y_pred = np.expm1(y_pred_log)
# y_test_original = np.expm1(y_test)

# for idx in range(min(25, len(y_test_original))):
#     print(f"Sample {idx+1}: Actual = {y_test_original.iloc[idx]}, Predicted = {y_pred[idx]:.2f}")

# mae = mean_absolute_error(y_test_original, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
# r2 = r2_score(y_test_original, y_pred)

# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R2 Score:", r2)

# # SMAPE metric
# def smape(y_true, y_pred):
#     numerator = np.abs(y_pred - y_true)
#     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
#     return np.mean(numerator / denominator) * 100

# score = smape(y_test_original, y_pred)
# print("SMAPE:", score)

# # -------------------------------
# # Save Model & Objects
# # -------------------------------
# import joblib
# joblib.dump(model, MODEL_PATH + '/lgbm_model_bert_tfidf_improved.pkl')
# joblib.dump(tfidf, MODEL_PATH + '/tfidf_improved.pkl')
# joblib.dump(scaler, MODEL_PATH + '/image_scaler_improved.pkl')
# joblib.dump(bert_model, MODEL_PATH + '/bert_model.pkl')

# print("Saved LGBM model, TF-IDF, BERT, and Scaler successfully!")



# # ===============================
# # Full LGBM Pipeline: TF-IDF + BERT + Image Features + IPQ
# # ===============================

# # Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# # -------------------------------
# # Paths
# # -------------------------------
# DATA_PATH = '/content/drive/MyDrive/dataset'
# CSV_PATH = DATA_PATH + '/train_cleaned.csv'
# FEATURES_PATH = '/content/drive/MyDrive/RGBMFeatures'  # precomputed image features
# MODEL_PATH = '/content/drive/MyDrive/models'

# # -------------------------------
# # Load CSV
# # -------------------------------
# import pandas as pd
# df = pd.read_csv(CSV_PATH)
# print(f"Loaded {len(df)} entries")
# print(df.head())

# # -------------------------------
# # Text Features: TF-IDF
# # -------------------------------
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(max_features=3000)
# X_text_tfidf = tfidf.fit_transform(df['clean_catalog_content'])
# print("TF-IDF features shape:", X_text_tfidf.shape)

# # -------------------------------
# # Text Features: BERT Embeddings
# # -------------------------------
# !pip install -q sentence-transformers
# from sentence_transformers import SentenceTransformer
# import numpy as np
# bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # compact & fast for Colab free GPU

# X_text_bert = bert_model.encode(df['clean_catalog_content'].tolist(), batch_size=64, show_progress_bar=True)
# X_text_bert = np.array(X_text_bert)
# print("BERT features shape:", X_text_bert.shape)

# # -------------------------------
# # Combine TF-IDF + BERT (Text)
# # -------------------------------
# from scipy.sparse import hstack, csr_matrix
# X_text_combined = hstack([X_text_tfidf, csr_matrix(X_text_bert)])
# print("Combined text features shape:", X_text_combined.shape)

# # -------------------------------
# # Load saved image feature batches
# # -------------------------------
# import glob
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

# # -------------------------------
# # Add IPQ feature
# # -------------------------------
# ipq_values = df[['ipq']].values

# # -------------------------------
# # Combine all features: Text + Image + IPQ
# # -------------------------------
# X_combined = hstack([X_text_combined, image_features_scaled, csr_matrix(ipq_values)])
# print("Total combined feature shape:", X_combined.shape)

# # -------------------------------
# # Target (log-transform)
# # -------------------------------
# y = np.log1p(df['price'])

# # -------------------------------
# # Train/Test Split
# # -------------------------------
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_combined, y, test_size=0.2, random_state=42
# )

# # -------------------------------
# # Train LGBM
# # -------------------------------
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

# # -------------------------------
# # Predict & Evaluate
# # -------------------------------
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# y_pred_log = model.predict(X_test)
# y_pred = np.expm1(y_pred_log)
# y_test_original = np.expm1(y_test)

# for idx in range(min(25, len(y_test_original))):
#     print(f"Sample {idx+1}: Actual = {y_test_original.iloc[idx]}, Predicted = {y_pred[idx]:.2f}")

# mae = mean_absolute_error(y_test_original, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
# r2 = r2_score(y_test_original, y_pred)

# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R2 Score:", r2)

# # SMAPE metric
# def smape(y_true, y_pred):
#     numerator = np.abs(y_pred - y_true)
#     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
#     return np.mean(numerator / denominator) * 100

# score = smape(y_test_original, y_pred)
# print("SMAPE:", score)

# # -------------------------------
# # Save Model & Objects
# # -------------------------------
# import joblib
# joblib.dump(model, MODEL_PATH + '/lgbm_model_bert_tfidf_improved.pkl')
# joblib.dump(tfidf, MODEL_PATH + '/tfidf_improved.pkl')
# joblib.dump(scaler, MODEL_PATH + '/image_scaler_improved.pkl')
# joblib.dump(bert_model, MODEL_PATH + '/bert_model.pkl')

# print("Saved LGBM model, TF-IDF, BERT, and Scaler successfully!")


# Full LGBM Pipeline: BERT Only + Image Features + IPQ
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Paths
DATA_PATH = '/content/drive/MyDrive/dataset'
CSV_PATH = DATA_PATH + '/train_cleaned.csv'
FEATURES_PATH = '/content/drive/MyDrive/RGBMFeatures'
MODEL_PATH = '/content/drive/MyDrive/models'

# Load CSV
import pandas as pd
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} entries")
print(df.head())

# Text Features: BERT Embeddings (ONLY)
from sentence_transformers import SentenceTransformer
import numpy as np

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

X_text_bert = bert_model.encode(
    df['clean_catalog_content'].tolist(),
    batch_size=64,
    show_progress_bar=True
)
X_text_bert = np.array(X_text_bert)
print("BERT features shape:", X_text_bert.shape)

from scipy.sparse import csr_matrix
X_text_combined = csr_matrix(X_text_bert)

# Load saved image feature batches
import glob
all_features = []
for f in sorted(glob.glob(FEATURES_PATH + '/image_features_batch_*.npy')):
    batch = np.load(f)
    all_features.append(batch)

image_features_combined = np.vstack(all_features)
print("Loaded image features shape:", image_features_combined.shape)

# Scale image features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
image_features_scaled = scaler.fit_transform(image_features_combined)

# Add IPQ feature
ipq_values = df[['ipq']].values

# Combine all features: BERT + Image + IPQ
from scipy.sparse import hstack, csr_matrix
X_combined = hstack([
    X_text_combined,
    image_features_scaled,
    csr_matrix(ipq_values)
])
print("Total combined feature shape:", X_combined.shape)

# Target (log-transform)
y = np.log1p(df['price'])

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

# Train LGBM
from lightgbm import LGBMRegressor, log_evaluation

model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    device='gpu',
    verbose=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='l1',
    callbacks=[log_evaluation(period=50)],
    early_stopping_rounds=100
)

# Predict & Evaluate
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

# SMAPE metric
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

score = smape(y_test_original, y_pred)
print("SMAPE:", score)

# Save Model & Objects
import joblib
joblib.dump(model, MODEL_PATH + '/lgbm_model_bert_only.pkl')
joblib.dump(scaler, MODEL_PATH + '/image_scaler_improved.pkl')
joblib.dump(bert_model, MODEL_PATH + '/bert_model.pkl')

print("Saved model, BERT, and scaler successfully!")