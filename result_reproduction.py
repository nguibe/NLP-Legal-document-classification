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
langs_to_keep = ['en', 'de', 'fr', 'pl', 'fi'] 
df_reduced = df.copy()

# Calculate the length of the document for each language
def compute_lengths(text_dict):
    lengths = {lang: len(text_dict[lang]) for lang in langs_to_keep if text_dict.get(lang) is not None}
    return lengths
# Apply the function to the 'text' column and store the result in a new column 'doc_lengths'
df_reduced['doc_lengths'] = df_reduced['text'].apply(compute_lengths)
df_reduced['max_doc_length'] = df_reduced['doc_lengths'].apply(lambda d: max(d.values(), default=0))
df_reduced = df_reduced[df_reduced['max_doc_length']<500000]


############## Data preparation #############
# keep only level 3 labels
df_reduced['level_3_labels'] = df_reduced['eurovoc_concepts'].apply(lambda d: d['level_3'] if 'level_3' in d else [])

# train
train_df = df_reduced[df_reduced['split']=='train']
train_df['text'] = train_df["text"].apply(lambda x: isinstance(x, dict) and x.get("en"))

# test 
test_df = df_reduced[df_reduced['split']=='test']
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
train_df["label_vector"] = list(label_matrix)
final_test_df["label_vector"] = list(mlb.transform(final_test_df["level_3_labels"]))

# dataset
train_dataset = Dataset.from_pandas(train_df[["text", "label_vector"]])
test_datasets = {
    lang: Dataset.from_pandas(df[["text", "label_vector"]]) 
    for lang, df in final_test_df.groupby("lang")
}

# truncation with tokenization
tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/")
# Apply tokenization to the training dataset
def tokenize(batch):
    # batch["text"] is already List[str]
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)

# Apply tokenization to each language-specific test dataset
for lang in test_datasets:
    test_datasets[lang] = test_datasets[lang].map(tokenize, batched=True)

# last modif
def prepare_dataset(example):
    example["labels"] = example["label_vector"]
    return example
train_dataset = train_dataset.map(prepare_dataset)
for lang in test_datasets:
    test_datasets[lang] = test_datasets[lang].map(prepare_dataset)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
for lang in test_datasets:
    test_datasets[lang].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


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