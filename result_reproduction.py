################# Legal document classification in zero-shot cross lingual transfer setting ####################
#  Part II: Results reproduction
"""
Date: May 2025

Project of course: Natural Language Processing - ENSAE 3A S2

Author: Noémie Guibé
"""

# imports
from datasets import Dataset
import pandas as pd
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

# truncation with tokenization
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize(batch):
    # Make sure batch["text"] is a list of strings, not a list of dictionaries
    if isinstance(batch["text"], list):
        # If already a list of strings, continue
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
    else:
        # If it's not, extract the correct string from each entry (e.g., handling dicts)
        texts = [str(item) for item in batch["text"]]  # Convert each item to string (adjust if it's a dictionary)
        return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
# Apply tokenization to each language-specific test dataset
for lang in test_datasets:
    test_datasets[lang] = test_datasets[lang].map(tokenize, batched=True)
print(train_dataset)
print(train_dataset[0]["label_vector"])
print(type(train_dataset[0]["label_vector"]))

# last modif
def prepare_dataset(example):
    example["labels"] = example["label_vector"]
    return example
train_dataset = train_dataset.map(prepare_dataset)
for lang in test_datasets:
    test_datasets[lang] = test_datasets[lang].map(prepare_dataset)
train_dataset.set_format(
    type="torch", 
    columns=["input_ids", "attention_mask"], 
    dtype=torch.int64  # Use int64 for input_ids and attention_mask as they are indices
)
# Set the format for labels as float32 (for BCEWithLogitsLoss)
train_dataset.set_format(
    type="torch", 
    columns=["labels"], 
    dtype=torch.float32  # Use float32 for labels for binary/multilabel classification
)

for lang in test_datasets:
    test_datasets[lang].set_format(type="torch", columns=["input_ids", "attention_mask"],dtype=torch.int64)
    test_datasets[lang].set_format(type="torch", columns=["labels"],dtype=torch.int64)


##################### Model ###################
# Get the number of labels
num_labels = len(mlb.classes_)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    problem_type="multi_label_classification",
    num_labels=num_labels,
    id2label={i: label for i, label in enumerate(mlb.classes_)},
    label2id={label: i for i, label in enumerate(mlb.classes_)}
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./xlm-roberta-eurovoc",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    logging_dir="./logs",                    # Log directory
    report_to="tensorboard"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_datasets["fr"],  # Or "de", "es" — you can loop through them too
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

################### Results ###############
for lang, dataset in test_datasets.items():
    results = trainer.evaluate(dataset)
    print(f"Language: {lang}")
    print(results)