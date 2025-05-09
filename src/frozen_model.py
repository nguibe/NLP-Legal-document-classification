"""
===========================================================
XLM-Roberta with layer freezing model
===========================================================

This script defines a multi-label classification pipeline using XLM-Roberta for training and evaluation. 
It includes functionality to freeze a specified number of transformer layers during training to optimize 
performance and memory usage, corresponding to the first adaptation strategy reproduced. 

===========================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from src.utils import (
    evaluate_model,
    freeze_transformer_layers,
    track_training_time_and_memory,
)


def run_training_pipeline_with_freezing(
    df,
    train_sample_size=5000,
    test_sample_size=1000,
    batch_size=8,
    epochs=2,
    n_frozen_layer=6
):
    """
    Runs the complete training and evaluation pipeline for a multi-label classification task using the XLM-Roberta model,
    with an option to freeze the first N layers of the transformer model.

    Parameters:
        df (pandas.DataFrame): The dataset containing the training and test data with text and associated labels. 
                                It must contain the columns 'split', 'text', and 'eurovoc_concepts'.
        train_sample_size (int, optional): Number of samples to use for training. Default is 5000.
        test_sample_size (int, optional): Number of samples to use for testing. Default is 1000.
        batch_size (int, optional): Batch size for training. Default is 8.
        epochs (int, optional): Number of epochs for training. Default is 2.
        n_frozen_layer (int, optional): Number of transformer layers to freeze during training. Default is 6.

    Returns:
        dict: A dictionary containing the following keys:
            - 'results': A dictionary of evaluation metrics (R-Precision, Micro F1, Macro F1, LRAP, Eval Time) for each language.
            - 'training_time': Total time taken for training the model (in seconds).
            - 'initial_memory': Memory usage (in MB) before training starts.
            - 'final_memory': Memory usage (in MB) after training ends.
            - 'model_params': Number of parameters in the model.
    """
    test_langs = ["en", "fr", "de", "pl", "fi"]

    # Preprocessing
    df['level_1_labels'] = df['eurovoc_concepts'].apply(lambda d: d.get('level_1', []))

    train_df = df[df['split'] == 'train'].copy()
    train_df['text'] = train_df['text'].apply(lambda x: x.get("en") if isinstance(x, dict) else "")

    test_df = df[df['split'] == 'test']
    test_dfs = []
    for lang in test_langs:
        df_lang = test_df[test_df["text"].apply(lambda x: isinstance(x, dict) and lang in x)].copy()
        df_lang['text'] = df_lang['text'].apply(lambda x: x[lang])
        df_lang['lang'] = lang
        test_dfs.append(df_lang)
    final_test_df = pd.concat(test_dfs, ignore_index=True)

    train_df = train_df.sample(train_sample_size, random_state=42)
    final_test_df = final_test_df.sample(test_sample_size, random_state=42)

    # Label binarization
    mlb = MultiLabelBinarizer()
    mlb.fit(df["level_1_labels"])
    train_df["label_vector"] = list(map(lambda x: x.astype(np.float32), mlb.transform(train_df["level_1_labels"])))
    final_test_df["label_vector"] = list(map(lambda x: x.astype(np.float32), mlb.transform(final_test_df["level_1_labels"])))

    # Tokenization
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
                yield {
                    "input_ids": ex["input_ids"],
                    "attention_mask": ex["attention_mask"]
                }, ex["labels"]
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

    train_tf_dataset = dataset_to_tf(train_dataset)
    test_tf_datasets = {lang: dataset_to_tf(ds) for lang, ds in test_datasets.items()}

    # Model setup
    num_labels = len(mlb.classes_)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        'xlm-roberta-base',
        num_labels=num_labels,
        problem_type='multi_label_classification'
    )

    if n_frozen_layer > 0:
        freeze_transformer_layers(model, N=n_frozen_layer)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.AUC(multi_label=True)]
    )

    # Training with memory/time tracking (from utils)
    training_time, initial_memory, final_memory = track_training_time_and_memory(
        model, train_tf_dataset, batch_size=batch_size, epochs=epochs
    )

    # Evaluation (from utils)
    results = {}
    for lang, lang_dataset in test_tf_datasets.items():
        print(f"\n[INFO] Evaluating for {lang}")
        metrics = evaluate_model(model, lang_dataset, batch_size=batch_size)
        results[lang] = {
            "R-Precision": metrics[0],
            "Micro F1": metrics[1],
            "Macro F1": metrics[2],
            "LRAP": metrics[3],
            "Eval Time": metrics[4]
        }

    return {
        "results": results,
        "training_time": training_time,
        "initial_memory": initial_memory,
        "final_memory": final_memory,
        "model_params": model.count_params(),
    }
