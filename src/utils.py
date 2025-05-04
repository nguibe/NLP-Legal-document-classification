# Functions used in the different models
import json
import os
import time

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from transformers import AutoConfig, AutoTokenizer, TFAutoModel


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
    Compute R-Precision at top-k for multi-label classification.

    R-Precision measures how many of the top-k predicted labels are actually relevant,
    where k is a fixed number (default: 10).

    Parameters:
        y_true (np.ndarray): Binary ground truth labels (shape: [n_samples, n_labels]).
        y_pred (np.ndarray): Predicted scores for each label (shape: [n_samples, n_labels]).
        top_k (int): Number of top predictions to consider for precision (default: 10).

    Returns:
        float: Mean R-Precision score across all samples.
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
    """
    Evaluate a multi-label classification model on a test dataset for 
    R-Precision, Micro F1, Macro F1, and LRAP metrics.

    Parameters:
        model (tf.keras.Model or AdapterXLMRModel): The trained model to evaluate.
        test_dataset (tf.data.Dataset): The test dataset, containing inputs and labels.
        batch_size (int): The batch size to use for evaluation (default: 32).

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - R-Precision score (float)
            - Micro F1 score (float)
            - Macro F1 score (float)
            - LRAP score (float)
            - Evaluation time in seconds (float)
    """
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
    Freezes the first N encoder layers of the XLM-Roberta transformer model to prevent 
    their weights from being updated during training, which can speed up training 
    and reduce overfitting in some scenarios.

    Parameters:
        model (tf.keras.Model): The TensorFlow model (typically a HuggingFace model) to modify.
        N (int): The number of transformer encoder layers to freeze (starting from the first layer).

    Raises:
        ValueError: If the model does not have the expected `roberta.encoder.layer` structure.
    """
    try:
        encoder = model.roberta.encoder.layer
    except AttributeError:
        raise ValueError("Expected model to have `roberta.encoder.layer` structure.")

    for i in range(N):
        encoder[i].trainable = False
    print(f"[INFO] Successfully froze first {N} transformer layers.")


def track_training_time_and_memory(model, train_dataset, batch_size=32, epochs=2):
     """
    Tracks training time and memory usage during model training.

    Parameters:
        model (tf.keras.Model): The TensorFlow model to train.
        train_dataset (tf.data.Dataset): The training dataset to use for training.
        batch_size (int): The batch size used during training (default: 32).
        epochs (int): The number of epochs to train the model (default: 2).

    Returns:
        tuple: A tuple containing:
            - training_time (float): The time taken to train the model (in seconds).
            - initial_memory (float): The memory usage before training (in MB).
            - final_memory (float): The memory usage after training (in MB).
    """
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
     """
    A custom Keras layer implementing an adapter for transformer-based models.

    This adapter layer consists of a bottleneck architecture where the input is first projected 
    down to a smaller size (bottleneck size), then projected back up to the original hidden size. 
    A residual connection is added to the output to allow for better gradient flow.

    Parameters:
        hidden_size (int): The size of the input and output of the adapter layer.
        bottleneck_size (int): The size of the bottleneck layer, used to reduce the dimensionality 
                               before projecting back to the hidden size (default: 64).
        **kwargs: Additional arguments passed to the parent class.

    Returns:
        tf.Tensor: The output tensor with the same shape as the input, after applying the down-projection, 
                   up-projection, and residual connection.
    """
    def __init__(self, hidden_size, bottleneck_size=64, **kwargs):
        super().__init__(**kwargs)
        self.down_proj = tf.keras.layers.Dense(bottleneck_size, activation='relu')
        self.up_proj = tf.keras.layers.Dense(hidden_size)

    def call(self, inputs):
        x = self.down_proj(inputs)
        x = self.up_proj(x)
        return x + inputs  # residual connection


class AdapterXLMRModel(tf.keras.Model):
    """
    A custom Keras model implementing an adapter-based approach on top of XLM-RoBERTa.

    This model uses XLM-RoBERTa as a base transformer model and adds adapter layers on top of each 
    transformer layer in the base model. The adapters are used to introduce lightweight modifications 
    to the model while retaining the original pretrained weights. Optionally, the base transformer model 
    can be frozen to prevent further training on the base layers.

    Parameters:
        num_labels (int): The number of output labels for classification.
        bottleneck_size (int): The size of the bottleneck layer in the adapter (default: 64).
        freeze_base (bool): Whether to freeze the base transformer model layers during training 
                             (default: False).
        **kwargs: Additional arguments passed to the parent class.

    Attributes:
        base_model (TFAutoModel): The XLM-RoBERTa model used as the base.
        adapters (list): A list of AdapterLayer objects, one for each transformer layer in the base model.
        dropout (tf.keras.layers.Dropout): A dropout layer to prevent overfitting.
        classifier (tf.keras.layers.Dense): A dense layer for classification output.

    Methods:
        call(input_ids, attention_mask, training=False, **kwargs):
            Performs the forward pass of the model, applying the base model, adapter layers, 
            and classification layer to the input data.

    Returns:
        logits (tf.Tensor): The predicted logits for each input, used for classification.
    """
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
    """
    Loads label embeddings for a given level of Eurovoc concepts using a transformer model.

    Parameters:
        model_name (str): The name of the pre-trained transformer model to use (default: 'xlm-roberta-base').
        level (str): The Eurovoc level to filter labels by (default: 'level_1').
        labels_path (str): The path to the folder containing the Eurovoc labels and concepts JSON files 
                           (default: "data/labels").

    Returns:
        label_ids (list): A list of label IDs for the selected level.
        embeddings (tf.Tensor): A tensor containing the embeddings of the label descriptions, 
                                normalized using L2 normalization.
        tokenizer (transformers.AutoTokenizer): The tokenizer used for generating embeddings.
        model (TFAutoModel): The pre-trained transformer model used for generating embeddings.
    """

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
     """
    Predicts the top-k relevant labels for a batch of input texts using a pre-trained model and label embeddings.
    Parameters:
        texts (list of str): A list of input texts (e.g., legal documents) for which labels are to be predicted.
        tokenizer (transformers.AutoTokenizer): The tokenizer used to preprocess the input texts.
        model (TFAutoModel): The pre-trained model used to compute document embeddings.
        label_embeddings (tf.Tensor): Pre-computed embeddings for the labels (e.g., Eurovoc concepts).
        top_k (int): The number of top relevant labels to return for each document (default: 5).

    Returns:
        numpy.ndarray: An array of shape (batch_size, top_k) containing the indices of the top-k most relevant labels 
                       for each input text in the batch.
    """
    prompts = [f"This legal document discusses the following topics: {txt}" for txt in texts]
    encodings = tokenizer(prompts, return_tensors='tf', padding=True, truncation=True, max_length=512)
    outputs = model(**encodings)
    doc_embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    doc_embeddings = tf.math.l2_normalize(doc_embeddings, axis=1)
    sims = tf.matmul(doc_embeddings, label_embeddings, transpose_b=True)
    top_k_scores, top_k_indices = tf.math.top_k(sims, k=top_k)
    return top_k_indices.numpy()


def generate_prompt(text, prompt_type="generic"):
    """
    Generates a prompt for text classification based on the type of prompt.
    
    Parameters:
        text (str): The input text (e.g., a legal document) for which the prompt is to be generated.
        prompt_type (str): The type of prompt to generate. It can either be "generic" or "guided" (default: "generic").

    Returns:
        str: A formatted prompt string based on the specified `prompt_type`.
            - "generic": A general prompt asking for legal categories.
            - "guided": A more specific prompt with a list of relevant categories.
    """
    if prompt_type == "guided":
        return (
            f"This legal document is about: {text}. "
            f"Relevant categories include: {', '.join(LABEL_DESCRIPTIONS.values())}. Which ones apply?"
        )
    return f"This legal document discusses the following topics: {text}. What legal categories apply?"
