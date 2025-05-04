"""
===========================================================
Baseline model: Training on english and evaluating in other languages
===========================================================

This script defines a pipeline for training and evaluating a multi-label classification model 
using the XLM-Roberta transformer model.

===========================================================
"""

import pandas as pd
import tensorflow as tf
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from src.utils import evaluate_model, track_training_time_and_memory


def run_training_pipeline(data, train_sample_size=1000, test_sample_size=5000, batch_size=8, epochs=2):
    """
    Runs the complete training and evaluation pipeline for a multi-label classification task using the XLM-Roberta model.

    This function processes the input data, tokenizes it, sets up the model, and trains it in english using multi-label classification 
    with a transformer-based model (XLM-Roberta). It also evaluates the model on test datasets across multiple languages 
    and reports evaluation metrics.

    Parameters:
        data (pandas.DataFrame): The dataset containing the training and test data with text and associated labels.
        train_sample_size (int, optional): Number of samples to use for training. Default is 1000.
        test_sample_size (int, optional): Number of samples to use for testing. Default is 5000.
        batch_size (int, optional): Batch size for training. Default is 8.
        epochs (int, optional): Number of epochs for training. Default is 2.

    Returns:
        dict: A dictionary containing evaluation metrics (R-Precision, Micro F1, Macro F1, LRAP, Eval Time) 
              for each language in the test set.
    """
    df = data.copy()

    # Preprocess training and test sets
    df['level_1_labels'] = df['eurovoc_concepts'].apply(lambda d: d.get('level_1', []))
    train_df = df[df['split'] == 'train']
    train_df.loc[:, 'text'] = train_df['text'].apply(lambda x: x.get("en") if isinstance(x, dict) else "")
    test_df = df[df['split'] == 'test']

    test_langs = ["en", "fr", "de", "pl", "fi"]
    test_dfs = []
    for lang in test_langs:
        df_lang = test_df[test_df['text'].apply(lambda x: isinstance(x, dict) and lang in x)].copy()
        df_lang.loc[:, 'text'] = df_lang['text'].apply(lambda x: x[lang])
        df_lang['lang'] = lang
        test_dfs.append(df_lang)

    final_test_df = pd.concat(test_dfs, ignore_index=True)

    train_df = train_df.sample(train_sample_size, random_state=42)
    final_test_df = final_test_df.sample(test_sample_size, random_state=42)

    mlb = MultiLabelBinarizer()
    mlb.fit(df["level_1_labels"])
    train_df["label_vector"] = list(mlb.transform(train_df["level_1_labels"]))
    final_test_df["label_vector"] = list(mlb.transform(final_test_df["level_1_labels"]))

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    def tokenize(batch): 
        encodings = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)
        encodings['labels'] = batch['label_vector']
        return encodings

    train_dataset = Dataset.from_pandas(train_df[["text", "label_vector"]]).map(tokenize, batched=True)
    test_datasets = {
        lang: Dataset.from_pandas(df[["text", "label_vector"]]).map(tokenize, batched=True)
        for lang, df in final_test_df.groupby("lang")
    }

    def dataset_to_tf(dataset):
        def gen():
            for ex in dataset:
                yield {"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"]}, ex["labels"]
        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                {
                    "input_ids": tf.TensorSpec(shape=(512,), dtype=tf.int64),
                    "attention_mask": tf.TensorSpec(shape=(512,), dtype=tf.int64),
                },
                tf.TensorSpec(shape=(len(mlb.classes_),), dtype=tf.float32)
            )
        )

    train_tf = dataset_to_tf(train_dataset)
    test_tf = {lang: dataset_to_tf(ds) for lang, ds in test_datasets.items()}

    num_labels = len(mlb.classes_)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        'xlm-roberta-base',
        num_labels=num_labels,
        problem_type='multi_label_classification'
    )
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.AUC(multi_label=True))
    
    # Train and evaluate with reusable functions
    training_time, initial_memory, final_memory = track_training_time_and_memory(
        model, train_tf, batch_size=batch_size, epochs=epochs
    )

    # Evaluate on all test languages using existing evaluate_model()
    results = {}
    for lang, lang_dataset in test_tf.items():
        print(f"[INFO] Evaluating on language: {lang}")
        r_prec, micro_f1, macro_f1, lrap, eval_time = evaluate_model(
        model, lang_dataset, batch_size=batch_size
        )
        results[lang] = {
        "R-Precision": r_prec,
        "Micro F1": micro_f1,
        "Macro F1": macro_f1,
        "LRAP": lrap,
        "Eval Time (s)": eval_time
    }

    return results
