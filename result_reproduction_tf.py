# imports
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd 
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import f1_score
import os
# imports
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd 
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import f1_score
import os
import time

os.chdir('NLP-Legal-document-classification')

# import data base
df = pd.read_parquet('data/dataset/multi_eurlex_reduced.parquet', engine='pyarrow')

##############################
# keep only level 3 labels
df['level_3_labels'] = df['eurovoc_concepts'].apply(lambda d: d['level_3'] if 'level_3' in d else [])
print(df.head())
train_df = df[df['split']=='train']
# English-only training set
train_df.loc[:,'text'] = train_df["text"].apply(lambda x: isinstance(x, dict) and x.get("en"))
print(train_df.head())

# test 
test_df = df[df['split']=='test']

# Test set in multiple languages
test_langs = ["fr", "de", "pl",'fi'] 
test_dfs = []

for lang in test_langs:
    # Filter rows where the language exists in the text dictionary
    df_lang = test_df[test_df["text"].apply(lambda x: isinstance(x, dict) and lang in x)]
    
    # Now extract the respective language text, and add the 'lang' column
    df_lang.loc[:,"text"] = df_lang["text"].apply(lambda x: x[lang])  # Extract the language text
    df_lang["lang"] = lang  # Add a new column for language
    
    # Append to test_dfs
    test_dfs.append(df_lang)

# Combine the list of DataFrames into one (exploded test set)
final_test_df = pd.concat(test_dfs, ignore_index=True)
print(final_test_df.head())

# ----------- Label encoding ----------------
# Fit on all labels
mlb = MultiLabelBinarizer()
mlb.fit(df["level_3_labels"])

# Now safely transform
train_df["label_vector"] = [row.tolist() for row in mlb.transform(train_df["level_3_labels"])]
final_test_df["label_vector"] = [row.tolist() for row in mlb.transform(final_test_df["level_3_labels"])]

print(len(mlb.classes_))

row_index = 5  # Change to the row you want to inspect
label_vector = train_df["label_vector"].iloc[row_index]

active_labels = [
    (i, mlb.classes_[i]) for i, val in enumerate(label_vector) if val == 1
]

print(f"Active labels for row {row_index}:")
for idx, label in active_labels:
    print(f"Index: {idx}, Label: {label}")

# check same index for same label in test dataset
label_id = "1810"
label_index = list(mlb.classes_).index(label_id)

# Get row indices in the test set where label 1810 is present
matching_indices = [
    i for i, row in enumerate(final_test_df["label_vector"]) if row[label_index] == 1
]

row_index = matching_indices[0]  
label_vector = final_test_df["label_vector"].iloc[row_index]

active_labels = [
    (i, mlb.classes_[i]) for i, val in enumerate(label_vector) if val == 1
]

print(f"Active labels for row {row_index}:")
for idx, label in active_labels:
    print(f"Index: {idx}, Label: {label}")


train_dataset = Dataset.from_pandas(train_df[["text", "label_vector"]])
test_datasets = {
    lang: Dataset.from_pandas(df[["text", "label_vector"]]) 
    for lang, df in final_test_df.groupby("lang")
}

# ----------- Tokenization ----------------
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

def tokenize_and_format_tf(batch):
    encodings = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)
    encodings['labels'] = batch['label_vector']
    return encodings

# Tokenize the datasets (train and test)
train_dataset = train_dataset.map(tokenize_and_format_tf, batched=True)
for lang in test_datasets:
    test_datasets[lang] = test_datasets[lang].map(tokenize_and_format_tf, batched=True)

# Inspect output
print("Original Text:", train_dataset['text'][0])
print("Token IDs:", train_dataset["input_ids"][0])
print("Tokens:", tokenizer.convert_ids_to_tokens(train_dataset["input_ids"][0]))


# ----------- Convert datasets to TensorFlow Dataset ----------------
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

# Convert both train and test datasets to TensorFlow Dataset
train_tf_dataset = dataset_to_tf(train_dataset)
test_tf_datasets = {lang: dataset_to_tf(test_datasets[lang]) for lang in test_datasets}


############# Model ###################
# ----------- Initialisation ----------------
# Model for multi-label classification
# Get the number of labels
num_labels = len(mlb.classes_)
model = TFAutoModelForSequenceClassification.from_pretrained(
    'xlm-roberta-base', num_labels=num_labels, problem_type='multi_label_classification'
)

# Compile the model with appropriate loss and optimizer
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.AUC(multi_label=True))

# ----------- Training ----------------
model.fit(train_tf_dataset.batch(8), epochs=5)

# ----------- Evaluation ----------------
for lang, tf_dataset in test_tf_datasets.items():
    results = model.evaluate(tf_dataset.batch(8))
    print(f"Language: {lang}")
    print("Evaluation results:", results)

os.chdir('NLP-Legal-document-classification')

# import data base
df = pd.read_parquet('data/dataset/multi_eurlex_reduced.parquet', engine='pyarrow')

##############################
# keep only level 3 labels
df['level_3_labels'] = df['eurovoc_concepts'].apply(lambda d: d['level_3'] if 'level_3' in d else [])
print(df.head())
train_df = df[df['split']=='train']
# English-only training set
train_df.loc[:,'text'] = train_df["text"].apply(lambda x: isinstance(x, dict) and x.get("en"))
print(train_df.head())

# test 
test_df = df[df['split']=='test']

# Test set in multiple languages
test_langs = ["fr", "de", "pl",'fi'] 
test_dfs = []

for lang in test_langs:
    # Filter rows where the language exists in the text dictionary
    df_lang = test_df[test_df["text"].apply(lambda x: isinstance(x, dict) and lang in x)]
    
    # Now extract the respective language text, and add the 'lang' column
    df_lang.loc[:,"text"] = df_lang["text"].apply(lambda x: x[lang])  # Extract the language text
    df_lang["lang"] = lang  # Add a new column for language
    
    # Append to test_dfs
    test_dfs.append(df_lang)

# Combine the list of DataFrames into one (exploded test set)
final_test_df = pd.concat(test_dfs, ignore_index=True)
print(final_test_df.head())

# ----------- Label encoding ----------------
# Fit on all labels
mlb = MultiLabelBinarizer()
mlb.fit(df["level_3_labels"])

# Now safely transform
train_df["label_vector"] = [row.tolist() for row in mlb.transform(train_df["level_3_labels"])]
final_test_df["label_vector"] = [row.tolist() for row in mlb.transform(final_test_df["level_3_labels"])]

print(len(mlb.classes_))

row_index = 5  # Change to the row you want to inspect
label_vector = train_df["label_vector"].iloc[row_index]

active_labels = [
    (i, mlb.classes_[i]) for i, val in enumerate(label_vector) if val == 1
]

print(f"Active labels for row {row_index}:")
for idx, label in active_labels:
    print(f"Index: {idx}, Label: {label}")

# check same index for same label in test dataset
label_id = "1810"
label_index = list(mlb.classes_).index(label_id)

# Get row indices in the test set where label 1810 is present
matching_indices = [
    i for i, row in enumerate(final_test_df["label_vector"]) if row[label_index] == 1
]

row_index = matching_indices[0]  
label_vector = final_test_df["label_vector"].iloc[row_index]

active_labels = [
    (i, mlb.classes_[i]) for i, val in enumerate(label_vector) if val == 1
]

print(f"Active labels for row {row_index}:")
for idx, label in active_labels:
    print(f"Index: {idx}, Label: {label}")


train_dataset = Dataset.from_pandas(train_df[["text", "label_vector"]])
test_datasets = {
    lang: Dataset.from_pandas(df[["text", "label_vector"]]) 
    for lang, df in final_test_df.groupby("lang")
}

# ----------- Tokenization ----------------
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

def tokenize_and_format_tf(batch):
    encodings = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)
    encodings['labels'] = batch['label_vector']
    return encodings

# Tokenize the datasets (train and test)
train_dataset = train_dataset.map(tokenize_and_format_tf, batched=True)
for lang in test_datasets:
    test_datasets[lang] = test_datasets[lang].map(tokenize_and_format_tf, batched=True)

# Inspect output
print("Original Text:", train_dataset['text'][0])
print("Token IDs:", train_dataset["input_ids"][0])
print("Tokens:", tokenizer.convert_ids_to_tokens(train_dataset["input_ids"][0]))


# ----------- Convert datasets to TensorFlow Dataset ----------------
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

# Convert both train and test datasets to TensorFlow Dataset
train_tf_dataset = dataset_to_tf(train_dataset)
test_tf_datasets = {lang: dataset_to_tf(test_datasets[lang]) for lang in test_datasets}


############# Model ###################
# ----------- Initialisation ----------------
# Model for multi-label classification
# Get the number of labels
num_labels = len(mlb.classes_)
model = TFAutoModelForSequenceClassification.from_pretrained(
    'xlm-roberta-base', num_labels=num_labels, problem_type='multi_label_classification'
)

# Compile the model with appropriate loss and optimizer
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.AUC(multi_label=True))

# ----------- Training ----------------
model.fit(train_tf_dataset.batch(8), epochs=5)

# ----------- Evaluation ----------------
for lang, tf_dataset in test_tf_datasets.items():
    results = model.evaluate(tf_dataset.batch(8))
    print(f"Language: {lang}")
    print("Evaluation results:", results)