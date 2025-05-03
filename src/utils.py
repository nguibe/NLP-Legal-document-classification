# Functions used in the different models
import numpy as np
import time
from sklearn.metrics import f1_score, label_ranking_average_precision_score
import os
import psutil
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, AutoConfig

LABEL_DESCRIPTIONS = {
    '100142': 'politics',
    '100143': 'international relations',
    '100144': 'EUROPEAN UNION',
    '100145': 'law',
    '100146': 'economics',
    '100147': 'trade',
    '100148': 'finance',
    '100149': 'social questions',
    '100150': 'education and communications',
    '100151': 'science',
    '100152': 'business and competition',
    '100153': 'employment and working conditions',
    '100154': 'transport',
    '100155': 'environment',
    '100156': 'agriculture, forestry and fisheries',
    '100157': 'agri-foodstuffs',
    '100158': 'production, technology and research',
    '100159': 'energy',
    '100160': 'industry',
    '100161': 'geography',
    '100162': 'international organisations'
}


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


def evaluate_model(model, test_dataset, batch_size=32):
    start_time = time.time()
    
    y_true = []
    y_pred = []
    
    for batch in test_dataset.batch(batch_size):
        inputs, labels = batch
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Check if model is AdapterXLMRModel or another model
        if isinstance(model, AdapterXLMRModel):
            logits = model({"input_ids": input_ids, "attention_mask": attention_mask}, training=False)
        else:
            # For other models that might return a tuple
            outputs = model({"input_ids": input_ids, "attention_mask": attention_mask}, training=False)
            logits = outputs[0]  # Assuming logits are the first element of the returned tuple

        predictions = tf.sigmoid(logits).numpy()
        
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
    
    r_precision_score = r_precision(y_true, y_pred)
    micro_f1 = f1_score(y_true, (y_pred > 0.5), average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, (y_pred > 0.5), average='macro', zero_division=0)
    lrap_score = label_ranking_average_precision_score(y_true, y_pred)
    
    evaluation_time = time.time() - start_time
    print(f"R-Precision: {r_precision_score:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"LRAP: {lrap_score:.4f}")
    print(f"Evaluation time: {evaluation_time:.2f} seconds")
    
    return r_precision_score, micro_f1, macro_f1, lrap_score, evaluation_time


def freeze_transformer_layers(model, N):
    """
    Freezes the first N encoder layers of the XLM-Roberta transformer.
    
    Parameters:
        model (tf.keras.Model): The TensorFlow HuggingFace model.
        N (int): Number of transformer layers to freeze.
    """
    try:
        encoder = model.roberta.encoder.layer
    except AttributeError:
        raise ValueError("Expected model to have `roberta.encoder.layer` structure.")

    for i in range(N):
        encoder[i].trainable = False
    print(f"[INFO] Successfully froze first {N} transformer layers.")


def track_training_time_and_memory(model, train_dataset, batch_size=32, epochs=2):
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


class AdapterLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, bottleneck_size=64, **kwargs):
        super().__init__(**kwargs)
        self.down_proj = tf.keras.layers.Dense(bottleneck_size, activation='relu')
        self.up_proj = tf.keras.layers.Dense(hidden_size)

    def call(self, inputs):
        x = self.down_proj(inputs)
        x = self.up_proj(x)
        return x + inputs  # residual connection


class AdapterXLMRModel(tf.keras.Model):
    def __init__(self, num_labels, bottleneck_size=64, freeze_base=False, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.bottleneck_size = bottleneck_size

        config = AutoConfig.from_pretrained("xlm-roberta-base", output_hidden_states=True)
        self.base_model = TFAutoModel.from_pretrained("xlm-roberta-base", config=config)

        if freeze_base:
            self.base_model.trainable = False

        # Adapter layers: one per transformer layer
        self.adapters = [
            AdapterLayer(hidden_size=config.hidden_size, bottleneck_size=bottleneck_size)
            for _ in range(config.num_hidden_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(num_labels)

    def call(self, input_ids=None, attention_mask=None, training=False, **kwargs):
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training
        )

        hidden_states = base_outputs.hidden_states
        x = hidden_states[-1]

        for adapter in self.adapters:
            x = adapter(x)

        cls_output = x[:, 0, :]
        cls_output = self.dropout(cls_output, training=training)
        logits = self.classifier(cls_output)

        return logits


def load_label_embeddings(model_name='xlm-roberta-base', level='level_1', labels_path="data/labels"):
    import json
    import pandas as pd

    with open(f"{labels_path}/eurovoc_descriptors.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    english_only = {k: v.get("en") for k, v in labels.items() if v.get("en")}
    labels_df = pd.DataFrame(english_only.items(), columns=["label_id", "label_description"])

    with open(f"{labels_path}/eurovoc_concepts.json", "r", encoding="utf-8") as f:
        levels = json.load(f)

    level_data = [
        {"label_id": label_id, "level": lvl}
        for lvl, ids in levels.items()
        for label_id in ids
    ]
    df_levels = pd.DataFrame(level_data)

    df_labels = labels_df.merge(df_levels, on="label_id")
    df_labels = df_labels[df_labels["level"] == level]
    label_texts = df_labels["label_description"].tolist()
    label_ids = df_labels["label_id"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name)

    inputs = tokenizer(label_texts, padding=True, truncation=True, return_tensors="tf")
    outputs = model(**inputs)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    embeddings = tf.math.l2_normalize(embeddings, axis=1)

    return label_ids, embeddings, tokenizer, model


def predict_labels_batch(texts, tokenizer, model, label_embeddings, top_k=5):
    prompts = [f"This legal document discusses the following topics: {txt}" for txt in texts]
    encodings = tokenizer(prompts, return_tensors='tf', padding=True, truncation=True, max_length=512)
    outputs = model(**encodings)
    doc_embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    doc_embeddings = tf.math.l2_normalize(doc_embeddings, axis=1)
    sims = tf.matmul(doc_embeddings, label_embeddings, transpose_b=True)
    top_k_scores, top_k_indices = tf.math.top_k(sims, k=top_k)
    return top_k_indices.numpy()


def generate_prompt(text, prompt_type="generic"):
    if prompt_type == "guided":
        return (
            f"This legal document is about: {text}. "
            f"Relevant categories include: {', '.join(LABEL_DESCRIPTIONS.values())}. Which ones apply?"
        )
    return f"This legal document discusses the following topics: {text}. What legal categories apply?"
