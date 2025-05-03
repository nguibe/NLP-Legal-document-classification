# Functions used in the different models
import numpy as np
import time
from sklearn.metrics import f1_score, label_ranking_average_precision_score
import os
import psutil
import tensorflow as tf
from transformers import TFAutoModel, AutoConfig


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
    micro_f1 = f1_score(y_true, (y_pred > 0.5), average='micro',zero_division=0)
    macro_f1 = f1_score(y_true, (y_pred > 0.5), average='macro',zero_division=0)
    
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

        # Load pretrained XLM-RoBERTa model
        config = AutoConfig.from_pretrained("xlm-roberta-base", output_hidden_states=True)
        self.base_model = TFAutoModel.from_pretrained("xlm-roberta-base", config=config)

        if freeze_base:
            self.base_model.trainable = False

        # Adapter layers: one for each transformer layer output
        self.adapters = [
            AdapterLayer(hidden_size=config.hidden_size, bottleneck_size=bottleneck_size)
            for _ in range(config.num_hidden_layers)
        ]

        # Final classification head
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(num_labels)

    def call(self, inputs, training=False):
        base_outputs = self.base_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            training=training
        )

        # Use hidden_states to access all transformer layer outputs
        hidden_states = base_outputs.hidden_states  # tuple of length 13 (embeddings + 12 layers)
        x = hidden_states[-1]

        # Apply adapters sequentially
        for i, adapter in enumerate(self.adapters):
            x = adapter(x)

        cls_output = x[:, 0, :]  # CLS token
        cls_output = self.dropout(cls_output, training=training)
        logits = self.classifier(cls_output)

        return logits