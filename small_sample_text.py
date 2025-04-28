import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd 
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from sklearn.metrics import f1_score, label_ranking_average_precision_score
import os
import time
import psutil
import numpy as np


# Change to your project directory
#os.chdir('NLP-Legal-document-classification')

# Load the dataset
start_time = time.time()
print(f"[INFO] Starting to load the dataset...")
df = pd.read_parquet('data/dataset/multi_eurlex_reduced.parquet', engine='pyarrow')
print(f"[INFO] Dataset loaded in {time.time() - start_time:.2f} seconds")

################## FUNCTIONS ################
# Define R-Precision computation
def r_precision(y_true, y_pred, top_k=10):
    """
    R-Precision: Precision at top-k (where k is the number of relevant labels).
    """
    precision_list = []
    for i in range(len(y_true)):
        true_labels = y_true[i]
        predicted_scores = y_pred[i]
        
        # Get the indices of top-k predicted labels based on predicted scores
        top_k_indices = predicted_scores.argsort()[-top_k:][::-1]
        
        # Calculate the number of relevant labels in the top-k predictions
        relevant_in_top_k = sum([1 for idx in top_k_indices if true_labels[idx] == 1])
        precision = relevant_in_top_k / top_k
        precision_list.append(precision)
    
    return np.mean(precision_list)

# Add additional metrics (Micro, Macro F1)
def evaluate_model(model, test_dataset, batch_size=8):
    start_time = time.time()
    
    # Evaluate the model
    y_true = []
    y_pred = []
    
    for batch in test_dataset.batch(batch_size):
        #input_ids = batch['input_ids']
        #attention_mask = batch['attention_mask']
        #(input_ids, attention_mask), labels = batch
        #labels = batch['labels']
        inputs, labels = batch
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Get model predictions
        logits = model(input_ids, attention_mask=attention_mask)[0]  # Directly access the first element
        predictions = tf.sigmoid(logits).numpy()  # Apply sigmoid to get probabilities
        #logits = model(input_ids, attention_mask=attention_mask)[0]
        #predictions = tf.sigmoid(logits.logits).numpy()
        
        y_true.extend(labels.numpy())
        y_pred.extend(predictions)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate R-Precision
    r_precision_score = r_precision(y_true, y_pred)
    
    # Calculate Micro and Macro F1 Scores
    micro_f1 = f1_score(y_true, (y_pred > 0.5), average='micro')
    macro_f1 = f1_score(y_true, (y_pred > 0.5), average='macro')
    
    # Calculate Label Ranking Average Precision (LRAP)
    lrap_score = label_ranking_average_precision_score(y_true, y_pred)
    
    # Log the results
    print(f"R-Precision: {r_precision_score:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"LRAP: {lrap_score:.4f}")
    
    # Calculate evaluation time
    evaluation_time = time.time() - start_time
    print(f"Evaluation time: {evaluation_time:.2f} seconds")
    
    return r_precision_score, micro_f1, macro_f1, lrap_score, evaluation_time

# Function to track training time and memory usage
def track_training_time_and_memory(model, train_dataset, batch_size=8, epochs=5):
    # Track training time
    start_time = time.time()
    
    # Use psutil to track memory usage
    process = psutil.Process(os.getpid())
    
    # Get initial memory usage
    initial_memory = process.memory_info().rss / 1024 ** 2  # in MB
    
    # Train the model
    model.fit(train_dataset.batch(batch_size), epochs=epochs)
    
    # Get final memory usage
    final_memory = process.memory_info().rss / 1024 ** 2  # in MB
    
    # Track training time
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Memory used during training: {final_memory - initial_memory:.2f} MB")
    
    return training_time, initial_memory, final_memory



############################## DATA PREPARATION ########################################
# ----------- Train and test datasets with one level of labels ----------------
# Keep only level 1 labels
start_time = time.time()
df['level_1_labels'] = df['eurovoc_concepts'].apply(lambda d: d['level_1'] if 'level_1' in d else [])
print(f"[INFO] Level 1 labels extracted in {time.time() - start_time:.2f} seconds")
print(df.head())

# Split dataset into train and test sets
train_df = df[df['split'] == 'train']
train_df.loc[:, 'text'] = train_df["text"].apply(lambda x: isinstance(x, dict) and x.get("en"))
print(f"[INFO] English-only training set processed in {time.time() - start_time:.2f} seconds")
print(train_df.head())

# Prepare test dataset
test_df = df[df['split'] == 'test']

# Log test dataset filtering
test_langs = ["fr", "de", "pl", 'fi'] 
test_dfs = []
for lang in test_langs:
    # Filter rows by language
    df_lang = test_df[test_df["text"].apply(lambda x: isinstance(x, dict) and lang in x)]
    df_lang.loc[:, "text"] = df_lang["text"].apply(lambda x: x[lang])  # Extract language text
    df_lang["lang"] = lang  # Add language column
    test_dfs.append(df_lang)
print(f"[INFO] Filtered test set for {len(test_langs)} languages in {time.time() - start_time:.2f} seconds")

# Combine the list of DataFrames into one (exploded test set)
final_test_df = pd.concat(test_dfs, ignore_index=True)
print(f"[INFO] Combined test set in {time.time() - start_time:.2f} seconds")
print(final_test_df.head())



# Extract a small subset of data (e.g., 5 examples for quick testing)
small_train_df = train_df.sample(5, random_state=42)  # Randomly select 5 samples from training set
small_test_df = final_test_df.sample(5, random_state=42)  # Randomly select 5 samples from test set

# You could also print out the data to see the sample
print(small_train_df[['text', 'level_1_labels']])
print(small_test_df[['text', 'level_1_labels']])

# Now, proceed with the same preprocessing steps for this small sample:
# Encode labels
start_time = time.time()
mlb = MultiLabelBinarizer()
mlb.fit(df["level_1_labels"])

small_train_df["label_vector"] = [row.tolist() for row in mlb.transform(small_train_df["level_1_labels"])]
small_test_df["label_vector"] = [row.tolist() for row in mlb.transform(small_test_df["level_1_labels"])]
print(f"[INFO] Label encoding completed in {time.time() - start_time:.2f} seconds")

# Tokenization
small_train_dataset = Dataset.from_pandas(small_train_df[["text", "label_vector"]])
small_test_dataset = Dataset.from_pandas(small_test_df[["text", "label_vector"]])

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
print(f"[INFO] Tokenizer loaded in {time.time() - start_time:.2f} seconds")

def tokenize_and_format_tf(batch):
    encodings = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)
    encodings['labels'] = batch['label_vector']
    return encodings

# Tokenize the small sample datasets
small_train_dataset = small_train_dataset.map(tokenize_and_format_tf, batched=True)
small_test_dataset = small_test_dataset.map(tokenize_and_format_tf, batched=True)

# Convert to TensorFlow datasets
def dataset_to_tf(dataset):
    def gen():
        for example in dataset:
            yield {
                "input_ids": example['input_ids'],
                "attention_mask": example['attention_mask']
            }, example['labels']

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=( 
            {
                "input_ids": tf.TensorSpec(shape=(512,), dtype=tf.int64),
                "attention_mask": tf.TensorSpec(shape=(512,), dtype=tf.int64)
            },
            tf.TensorSpec(shape=(len(mlb.classes_),), dtype=tf.float32)
        )
    )


small_train_tf_dataset = dataset_to_tf(small_train_dataset)
small_test_tf_dataset = dataset_to_tf(small_test_dataset)

# Initialize the model again for small testing
num_labels = len(mlb.classes_)
model = TFAutoModelForSequenceClassification.from_pretrained(
    'xlm-roberta-base', num_labels=num_labels, problem_type='multi_label_classification'
)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.AUC(multi_label=True))

# Training the model on the small dataset (limit to 1 epoch for quick testing)
training_time, initial_memory, final_memory = track_training_time_and_memory(model, small_train_tf_dataset, epochs=1)

# Evaluation on the small test dataset
#results = evaluate_model(model, small_test_tf_dataset)
#print("Test results on small sample:", results)

for lang in test_langs:
    lang_specific_df = small_test_df[small_test_df['lang'] == lang]
    if lang_specific_df.empty:
        print(f"[INFO] No samples available for {lang}. Skipping evaluation.")
        continue  # Skip to next language
    
    # Proceed with tokenization and evaluation as usual
    lang_specific_dataset = Dataset.from_pandas(lang_specific_df[["text", "label_vector", "lang"]])
    lang_specific_dataset = lang_specific_dataset.map(tokenize_and_format_tf, batched=True)
    lang_specific_tf_dataset = dataset_to_tf(lang_specific_dataset)

    results = evaluate_model(model, lang_specific_tf_dataset)
    print(f"[INFO] Evaluation for {lang} completed")
    print(f"Language: {lang}")
    print("Evaluation results:", results)


