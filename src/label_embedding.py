from src.utils import load_label_embeddings, predict_labels_batch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, label_ranking_average_precision_score
import numpy as np
from tqdm import tqdm
import os


def run_label_embedding_classification(df, top_k=5, batch_size=32, eval_lang='en'):
    # Change to your project directory
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