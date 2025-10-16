import pandas as pd
import re
import os

# -----------------------------
# 1. Text cleaning
# -----------------------------
def clean_text(text):
    """
    Lowercase, remove extra spaces, and ensure string format.
    """
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# -----------------------------
# 2. Extract Item Pack Quantity (IPQ)
# -----------------------------
def extract_ipq(text):
    """
    Extract first numeric value from text as IPQ.
    Default to 1 if none found.
    """
    try:
        match = re.search(r'(\d+)', str(text))
        return int(match.group(1)) if match else 1
    except:
        return 1

# -----------------------------
# 3. Basic text statistics
# -----------------------------
def text_stats(text):
    """
    Returns dictionary of basic text features:
    - char count
    - word count
    - number of digits
    - number of uppercase letters
    """
    char_len = len(text)
    word_len = len(text.split())
    num_digits = sum(c.isdigit() for c in text)
    num_upper = sum(c.isupper() for c in text)
    return {'char_len': char_len, 'word_len': word_len, 'num_digits': num_digits, 'num_upper': num_upper}

# -----------------------------
# 4. Flag missing images
# -----------------------------
def image_missing_flag(sample_id, image_folder="images"):
    """
    Returns 1 if image missing, 0 if present
    """
    image_path = os.path.join(image_folder, f"{sample_id}.jpg")
    return 0 if os.path.exists(image_path) else 1

# -----------------------------
# 5. Preprocess entire dataframe
# -----------------------------
def preprocess(df, image_folder="images"):
    """
    Main preprocessing function for the dataset.
    """
    # Clean catalog text
    df['catalog_clean'] = df['catalog_content'].fillna('').apply(clean_text)

    # Extract IPQ
    df['ipq'] = df['catalog_content'].apply(extract_ipq)

    # Extract text statistics
    stats = df['catalog_clean'].apply(text_stats)
    df['char_len'] = stats.apply(lambda x: x['char_len'])
    df['word_len'] = stats.apply(lambda x: x['word_len'])
    df['num_digits'] = stats.apply(lambda x: x['num_digits'])
    df['num_upper'] = stats.apply(lambda x: x['num_upper'])

    # Flag missing images
    df['image_missing'] = df['sample_id'].apply(lambda x: image_missing_flag(x, image_folder))

    return df