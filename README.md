# Multilingual Legal Document Classification – Reproducing and Extending MultiEURLEX

This project is part of the "Natural Language Processing" course. Based on a chosen specific NLP problem the goal was to explore and experiment from state of the art paper and models


## Context & Motivation

This project investigates **multilingual zero-/one-shot classification** of legal documents based on the [MultiEURLEX dataset](https://huggingface.co/datasets/multi_eurlex), following the framework and findings of:

> **Chalkidis et al. (2021)** – *"MultiEURLEX: A Multilingual and Multi-label Legal Document Classification Dataset for Zero-shot Cross-lingual Transfer"*  
> [arxiv link](https://arxiv.org/abs/2109.00904)

Their work highlights the limitations of standard multilingual fine-tuning for legal classification across languages, especially in low-resource or typologically distant languages. The authors show that adaptation strategies can help recover or boost multilingual generalization.

## Dataset

- **Source**: EU laws translated into 23 official languages, labeled using a hierarchical taxonomy (EUROVOC).
- **Structure**:
  - `train`: 55,000 laws (1958–2010)
  - `validation`: 5,000 (2010–2012)
  - `test`: 5,000 (2012–2015)
- **Label levels**: 1–8 (this project uses only **Level 1**, with 21 top-level categories)
- **Splitting strategy**: Temporal (to mitigate concept drift)
- **Languages selected for evaluation**:  
  - 🇫🇷 French (Romance)  
  - 🇩🇪 German (Germanic)  
  - 🇵🇱 Polish (Slavic)  
  - 🇫🇮 Finnish (Uralic)

### Reproducibility & Data Access

#### 1. Data Analysis (Part 1)

The full dataset contains rich metadata (e.g., publication date) that is **not accessible** via the usual Hugging Face `load_dataset` interface.  
To reproduce the data analysis as closely as possible, we recommend:

- Downloading the full dataset archive (`multi_eurlex.tar.gz`, ~2.7 GB) from Hugging Face,  
- Extracting it locally into a dedicated folder, and  
- Updating script paths to point to this local copy.

> ⚠️ While remote extraction is technically possible, we **do not recommend it** due to the file size and risk of incomplete decompression or corrupted downloads.  
> A template script is nevertheless provided in the notebook.

#### 2. Model Training & Evaluation (Parts 2 & 3)

For all modeling experiments, we use a **preprocessed and reduced version** of the dataset, containing only the essential fields (`law_id`, `text`, `level_1_labels`, `split`).

This simplified dataset is generated during the data analysis phase and stored as a `.parquet` file in the `data/` folder.  
It is:

- **Synchronized via S3**,  
- **Excluded from version control** (via `.gitignore`),  
- **Not necessary to regenerate**, for those only looking to reproduce results.

This makes experiments fully reproducible without requiring the full original archive.

## Model Architecture

- **Base Model**: `XLM-Roberta-base`  
- **Adaptation Strategy**: Adapters inserted after FF layers and / or frozen layers (N ∈ {3, 6, 9, 12})
- **Implementation**: Custom training functions located in `src/`, imported into notebooks

## Project Structure

├── data/ # with eurovocs labels and label descriptions in labels folder

(├── model/ ) # temporary folder to store model weight (in gitingore de to size)

├── src/

│ ├── adapter_model.py #  script corresponding to the second adaptation strategy in Notebook 2 (adapters)

│ ├── baseline_model.py # model fully retrained in english

│ ├── frozen_model.py #   called in Notebook 2 as first adaptation strategy

│ ├── label_embedding.py #   called in Notebook 3 as second adaptation strategy

│ ├── prompt_model.py #   called in Notebook 3 as first adaptation strategy

│ ├── utils.py # various utility functions called by the models for instance for evaluation or training with specific metrics and logs


├── 1_data_analysis.ipynb

├── 2_results_reproduction.ipynb

└── 3_beyound_results_reproduction.ipynb

├──  pdf report

├── .gitignore

└── README.md

This repository contains three main stages:

### 1. 📁 `notebooks/01_data_analysis.ipynb`
Exploratory analysis of:
- Label hierarchy and class imbalance  
- Language coverage and distribution  
- Document lengths and temporal evolution  
- Concept drift and label noise  

### 2. 📁 `notebooks/02_results_reproduction.ipynb`
Reproduces two key observations from the paper:
- **Retraining on English** causes performance degradation on other languages (catastrophic forgetting)
- **Adapter-based fine-tuning** partially restores multilingual performance with reduced compute and memory cost

### 3. 📁 `notebooks/03_exploration_and_improvement.ipynb`
Original experimental ideas:
- (Manual analysis of token-label co-occurrence across languages)
- (Pattern mining (e.g., use of legal Latin roots, NER, citation structure)) 
- Optional low-cost strategies (label embeddings, prompts)

## Limitations

Due to **limited compute (CPU-only)** and **disk constraints**, the following simplifications were made:
- Subset of data: ~10k training examples instead of 55k
- Only **Level 1** labels (21 categories)
- Maximum of 2 training epochs
- Evaluation restricted to 4 languages
- Only **1 adaptation strategy (adapters)** explored in-depth

Despite these constraints, key trends were confirmed, and several practical insights were gained.

## Future Work

- Explore more adaptation techniques (e.g., BitFit, LoRA)
- Analyze weight drift to quantify catastrophic forgetting
- Test low-cost multilingual prompting strategies

---

Author: Noémie Guibé  
Date: May 2025  
