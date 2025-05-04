"""
===========================================================
Label embedding classification with XLM-Roberta
===========================================================

This function performs multi-label classification on a given dataset of legal documents using precomputed 
label embeddings,- our second "original" strategy. It uses the XLM-Roberta model to predict the top-k relevant labels for each document. 

===========================================================
"""

import os

from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from src.utils import load_label_embeddings, predict_labels_batch


def run_label_embedding_classification(df, top_k=5, batch_size=32, eval_lang='en'):
    """
    Runs label embedding-based multi-label classification on a test dataset and evaluates the performance.
    
    Args:
        df (pd.DataFrame): DataFrame containing the test dataset with 'text' and 'eurovoc_concepts' columns.
        top_k (int, optional): The number of top labels to predict for each document (default is 5).
        batch_size (int, optional): The batch size for predictions (default is 32).
        eval_lang (str, optional): The language in which the documents are evaluated (default is 'en').
    
    Returns:
        dict: A dictionary with the classification performance metrics:
              - "micro_f1" : The micro-average F1 score.
              - "macro_f1" : The macro-average F1 score.
              - "lrap"     : The Label Ranking Average Precision score.
    """
    # Ensure in project root directory
    if not os.getcwd().endswith('NLP-Legal-document-classification'):
        os.chdir('NLP-Legal-document-classification')
    # Load label embeddings
    label_ids, label_embeddings, tokenizer, model = load_label_embeddings()

    # Prepare test data
    df['level_1_labels'] = df['eurovoc_concepts'].apply(lambda d: d.get('level_1', []))
    test_df = df[(df['split'] == 'test') & df['text'].apply(lambda x: isinstance(x, dict) and eval_lang in x)].copy()
    test_df['text'] = test_df['text'].apply(lambda x: x[eval_lang])
    texts = test_df['text'].tolist()
    true_labels = test_df['level_1_labels'].tolist()

    # Label binarization
    mlb = MultiLabelBinarizer(classes=label_ids)
    mlb.fit(true_labels)

    # Batch prediction
    all_preds = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        indices = predict_labels_batch(batch_texts, tokenizer, model, label_embeddings, top_k=top_k)
        preds = [[label_ids[idx] for idx in row] for row in indices]
        all_preds.extend(preds)

    # Evaluation
    y_true = mlb.transform(true_labels)
    y_pred = mlb.transform(all_preds)

    micro = f1_score(y_true, y_pred, average='micro')
    macro = f1_score(y_true, y_pred, average='macro')
    lrap = label_ranking_average_precision_score(y_true, y_pred)

    print("\n[RESULTS]")
    print(f"Language: {eval_lang}")
    print(f"Top-{top_k} Micro F1: {micro:.4f}")
    print(f"Top-{top_k} Macro F1: {macro:.4f}")
    print(f"Top-{top_k} LRAP:     {lrap:.4f}")

    return {
        "micro_f1": micro,
        "macro_f1": macro,
        "lrap": lrap
    }