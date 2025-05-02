import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd 
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import f1_score
import os

os.chdir('NLP-Legal-document-classification')

def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()
    return {
        "micro_f1": f1_score(labels, preds, average="micro"),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


# import data base
df = pd.read_parquet('data/dataset/multi_eurlex_reduced.parquet', engine='pyarrow')

############## Data preparation #############
# keep only level 3 labels
df['level_3_labels'] = df['eurovoc_concepts'].apply(lambda d: d['level_3'] if 'level_3' in d else [])

# train
train_df = df[df['split']=='train']
train_df['text'] = train_df["text"].apply(lambda x: isinstance(x, dict) and x.get("en"))

# test 
test_df = df[df['split']=='test']
test_langs = ["fr", "de", "pl","fi"] 
test_dfs = []

for lang in test_langs:
    # Filter rows where the language exists in the text dictionary
    df_lang = test_df[test_df["text"].apply(lambda x: isinstance(x, dict) and lang in x)]
    
    # Now extract the respective language text, and add the 'lang' column
    df_lang["text"] = df_lang["text"].apply(lambda x: x[lang])  # Extract the language text
    df_lang["lang"] = lang  # Add a new column for language
    
    # Append to test_dfs
    test_dfs.append(df_lang)

# Combine the list of DataFrames into one (exploded test set)
final_test_df = pd.concat(test_dfs, ignore_index=True)

# Label encoding
mlb = MultiLabelBinarizer()
label_matrix = mlb.fit_transform(train_df["level_3_labels"])
train_df["label_vector"] = [row.tolist() for row in label_matrix]
# Apply same transformation to test sets
final_test_df["label_vector"] = [row.tolist() for row in mlb.transform(final_test_df["level_3_labels"])]
print('After label encoding', train_df["label_vector"].iloc[0])

# dataset
train_dataset = Dataset.from_pandas(train_df[["text", "label_vector"]])
test_datasets = {
    lang: Dataset.from_pandas(df[["text", "label_vector"]]) 
    for lang, df in final_test_df.groupby("lang")
}

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

def tokenize_and_format_tf(batch):
    encodings = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)
    encodings['labels'] = batch['label_vector']
    return encodings

# Tokenize the datasets (train and test)
train_dataset = train_dataset.map(tokenize_and_format_tf, batched=True)
for lang in test_datasets:
    test_datasets[lang] = test_datasets[lang].map(tokenize_and_format_tf, batched=True)

# Convert datasets to TensorFlow Dataset
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

# Model for multi-label classification
# Get the number of labels
num_labels = len(mlb.classes_)
model = TFAutoModelForSequenceClassification.from_pretrained(
    'xlm-roberta-base', num_labels=num_labels, problem_type='multi_label_classification'
)

# Compile the model with appropriate loss and optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_tf_dataset.batch(8), epochs=5)

# Evaluate the model for each language
for lang, tf_dataset in test_tf_datasets.items():
    results = model.evaluate(tf_dataset.batch(8))
    print(f"Language: {lang}")
    print("Evaluation results:", results)
