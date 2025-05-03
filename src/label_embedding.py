import json
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from tqdm import tqdm
import os

# ----------------- CONFIG -----------------
MODEL_NAME = 'xlm-roberta-base'
TOP_K = 5
BATCH_SIZE = 32
EVAL_LANG = 'en'  # Language to evaluate on
# ------------------------------------------

# Change to your project directory
if not os.getcwd().endswith('NLP-Legal-document-classification'):
    os.chdir('NLP-Legal-document-classification')

# Load label descriptions and restrict to English
keys_to_keep = ['text', 'labels']

with open("data/labels/eurovoc_descriptors.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

english_only = {k: v.get("en") for k, v in labels.items() if v.get("en") is not None}
labels = pd.DataFrame(list(english_only.items()), columns=["label_id", "label_description"])

with open("data/labels/eurovoc_concepts.json", "r", encoding="utf-8") as f:
    levels = json.load(f)
level_data = [
    {"label_id": label_id, "level": level}
    for level, label_ids in levels.items()
    for label_id in label_ids
]
df_levels = pd.DataFrame(level_data)
df_labels = pd.merge(labels, df_levels, on='label_id', how='inner')
df_labels = df_labels[df_labels['level'] != 'original']
df_labels = df_labels[df_labels['level'] == 'level_1']
label_ids = df_labels['label_id'].tolist()
label_texts_en = df_labels['label_description'].tolist()  

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModel.from_pretrained(MODEL_NAME)

# Embed English label descriptions
print("[INFO] Encoding English label descriptions...")
label_inputs = tokenizer(label_texts_en, padding=True, truncation=True, return_tensors='tf')
label_outputs = model(**label_inputs)
label_embeddings = tf.reduce_mean(label_outputs.last_hidden_state, axis=1)
label_embeddings = tf.math.l2_normalize(label_embeddings, axis=1)

# Load dataset and extract Level 1 labels
df = pd.read_parquet('https://minio.lab.sspcloud.fr/nguibe/NLP/multi_eurlex_reduced.parquet', engine='pyarrow')
df['level_1_labels'] = df['eurovoc_concepts'].apply(lambda d: d['level_1'] if 'level_1' in d else [])

# Filter test set in target language 
test_df = df[df['split'] == 'test']
test_df = test_df[test_df['text'].apply(lambda x: isinstance(x, dict) and EVAL_LANG in x)].copy()
test_df['text'] = test_df['text'].apply(lambda x: x[EVAL_LANG])
texts = test_df['text'].tolist()
true_labels = test_df['level_1_labels'].tolist()

# Prepare label binarizer for evaluation
mlb = MultiLabelBinarizer(classes=label_ids)
mlb.fit(true_labels)

# Define prediction function
def predict_labels_batch(texts, label_embeddings, top_k=5):
    prompts = [f"This legal document discusses the following topics: {txt}" for txt in texts]
    encodings = tokenizer(prompts, return_tensors='tf', padding=True, truncation=True, max_length=512)
    outputs = model(**encodings)
    doc_embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    doc_embeddings = tf.math.l2_normalize(doc_embeddings, axis=1)
    sims = tf.matmul(doc_embeddings, label_embeddings, transpose_b=True)
    top_k_scores, top_k_indices = tf.math.top_k(sims, k=top_k)
    return top_k_indices.numpy()

# Predict in batches
print(f"[INFO] Predicting labels in language: {EVAL_LANG}")
all_preds = []
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i+BATCH_SIZE]
    top_indices = predict_labels_batch(batch_texts, label_embeddings, top_k=TOP_K)
    batch_preds = [[label_ids[idx] for idx in indices] for indices in top_indices]
    all_preds.extend(batch_preds)

# Evaluation
y_pred = mlb.transform(all_preds)
y_true = mlb.transform(true_labels)

micro_f1 = f1_score(y_true, y_pred, average='micro')
macro_f1 = f1_score(y_true, y_pred, average='macro')
lrap = label_ranking_average_precision_score(y_true, y_pred)

# Results
print("\n[RESULTS]")
print(f"Language: {EVAL_LANG}")
print(f"Top-{TOP_K} Micro F1: {micro_f1:.4f}")
print(f"Top-{TOP_K} Macro F1: {macro_f1:.4f}")
print(f"Top-{TOP_K} LRAP:     {lrap:.4f}")
