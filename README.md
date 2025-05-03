# Multilingual Legal Document Classification – Reproducing and Extending MultiEURLEX

This project is part of the "Natural Language Processing" course. Based on a chosen specific NLP problem the goal was to explore and experiment from state of the art paper and models


## Context & Motivation

This project investigates **multilingual zero-/one-shot classification** of legal documents based on the [MultiEURLEX dataset](https://huggingface.co/datasets/multi_eurlex), following the framework and findings of:

> **Chalkidis et al. (2021)** – *"MultiEURLEX: A Multilingual and Multi-label Legal Document Classification Dataset for Zero-shot Cross-lingual Transfer"*  
> [arxiv link]((https://arxiv.org/abs/2109.00904)

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

## Model Architecture

- **Base Model**: `XLM-Roberta-base`  
- **Adaptation Strategy**: Adapters inserted after FF layers, with frozen layers (varied N ∈ {3, 6, 9, 12})
- **Implementation**: Custom training functions located in `src/`, imported into notebooks

## Project Structure

├── notebooks/

│ ├── 01_data_analysis.ipynb

│ ├── 02_results_reproduction.ipynb

│ └── 03_exploration_and_improvement.ipynb

├── src/

│ ├── model.py # model definition and adapter integration

│ ├── training.py # training loop, metrics, evaluation

│ ├── utils.py 

├── data/ 

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
- 🔻 **Retraining on English** causes performance degradation on other languages (catastrophic forgetting)
- 🔼 **Adapter-based fine-tuning** partially restores multilingual performance with reduced compute and memory cost

### 3. 📁 `notebooks/03_exploration_and_improvement.ipynb`
Original experimental ideas:
- Manual analysis of token-label co-occurrence across languages  
- Pattern mining (e.g., use of legal Latin roots, NER, citation structure)  
- Attempt to infer categories without full model retraining  
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
