
import os
import time
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import Dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from src.utils import evaluate_model, track_training_time_and_memory, freeze_transformer_layers

def run_training_pipeline_with_freezing(
    df,
    train_sample_size=5000,
    test_sample_size=1000,
    batch_size=8,
    epochs=2,
    n_frozen_layer=12
):


    test_langs = ["en", "fr", "de", "pl", "fi"]

    df['level_1_labels'] = df['eurovoc_concepts'].apply(lambda d: d.get('level_1', []))

    train_df = df[df['split'] == 'train'].copy()
    train_df.loc[:, 'text'] = train_df['text'].apply(lambda x: x.get("en") if isinstance(x, dict) else "")

    test_df = df[df['split'] == 'test']
    test_dfs = []
    for lang in test_langs:
        df_lang = test_df[test_df["text"].apply(lambda x: isinstance(x, dict) and lang in x)].copy()
        df_lang.loc[:, 'text'] = df_lang['text'].apply(lambda x: x[lang])
        df_lang["lang"] = lang
        test_dfs.append(df_lang)
    final_test_df = pd.concat(test_dfs, ignore_index=True)

    train_df = train_df.sample(train_sample_size, random_state=42)
    final_test_df = final_test_df.sample(test_sample_size, random_state=42)

    mlb = MultiLabelBinarizer()
    mlb.fit(df["level_1_labels"])
    train_df["label_vector"] = list(map(lambda x: x.astype(np.float32), mlb.transform(train_df["level_1_labels"])))
    final_test_df["label_vector"] = list(map(lambda x: x.astype(np.float32), mlb.transform(final_test_df["level_1_labels"])))

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

    train_tf_dataset = dataset_to_tf(train_dataset).batch(batch_size)
    test_tf_datasets = {lang: dataset_to_tf(ds).batch(batch_size) for lang, ds in test_datasets.items()}

    num_labels = len(mlb.classes_)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        'xlm-roberta-base',
        num_labels=num_labels,
        problem_type='multi_label_classification'
    )

    freeze_transformer_layers(model, N=n_frozen_layer)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=tf.keras.metrics.AUC(multi_label=True))

    # Training time and memory tracking
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 ** 2
    model.fit(train_tf_dataset, epochs=epochs)
    final_memory = process.memory_info().rss / 1024 ** 2
    training_time = time.time() - start_time

    def r_precision(y_true, y_pred, top_k=10):
        return np.mean([
            np.sum(y_true[i][np.argsort(y_pred[i])[-top_k:]]) / top_k
            for i in range(len(y_true))
        ])

    def evaluate_model(model, dataset):
        y_true, y_pred = [], []
        for inputs, labels in dataset:
            logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
            preds = tf.sigmoid(logits).numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        r_prec = r_precision(y_true, y_pred)
        micro_f1 = f1_score(y_true, y_pred > 0.5, average='micro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred > 0.5, average='macro', zero_division=0)
        lrap = label_ranking_average_precision_score(y_true, y_pred)
        return r_prec, micro_f1, macro_f1, lrap

    results = {}
    for lang, lang_dataset in test_tf_datasets.items():
        r_prec, micro_f1, macro_f1, lrap = evaluate_model(model, lang_dataset)
        results[lang] = {
            "R-Precision": r_prec,
            "Micro F1": micro_f1,
            "Macro F1": macro_f1,
            "LRAP": lrap
        }

    return {
        "results": results,
        "training_time": training_time,
        "initial_memory": initial_memory,
        "final_memory": final_memory,
        "model_params": model.count_params(),
    }
