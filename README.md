# SemEval-2026 Task 13: Detecting Machine-Generated Code with Multiple Programming Languages, Generators, and Application Scenarios


## 🚀 Project Overview

The **Procastic Simulators** project is an implementation for **SemEval-2026 Task 13**, focusing on detecting machine-generated code. This repository contains data loaders, baseline models, improved training notebooks, and evaluation scripts to tackle the challenge across multiple programming languages and LLM generators.

---

## 📁 Repository Structure

```text
procastic-simulaters-SemEval-2026-Task13/
├── baselines/              # Foundational starter notebooks
│   ├── Kaggle_starters/    # Official Kaggle starter implementations
│   ├── train.py            # Fine-tune CodeBERT for tasks A/B/C
│   └── predict.py          # Script for batch inference
├── Improved_Notebooks/     # Advanced experiments and refined models
│   ├── task-A/            # Stylometric and architectural improvements for Task A
│   └── task-B/            # Imbalance handling and fine-tuning for Task B
├── modified_notebooks/     # Optimized versions with specific enhancements
├── logs/                   # Training logs and experimental results
├── scorer.py               # Official Macro F1-score calculator
└── README.md               # Main project documentation
```

## 📓 Notebooks Overview

### Subtask A: Binary Classification (Human vs. AI)

| Notebook | Description |
| :--- | :--- |
| `baselines/Kaggle_starters/Task-A-Baseline-Full-Datset.ipynb` | A robust baseline for Task A using the full dataset. Implements a `CodeClassifierTrainer` capable of switching between standard transformer heads and BiLSTM-augmented heads. |
| `baselines/Kaggle_starters/Task-A-Baseline-limited.ipynb` | An advanced baseline that introduces **15 hand-crafted stylometric features** integrated via gated fusion and attention pooling. |
| `Improved_Notebooks/task-A/backbone_and_architecture_changes.ipynb` | Explores architectural modifications, specifically placing a **BiLSTM layer** between the transformer backbone and the classification head. |
| `Improved_Notebooks/task-A/stylometric_feature_addition.ipynb` | Experimentation with various language-agnostic code features to complement semantic embeddings. |
| `Improved_Notebooks/task-A/final_model.ipynb` | Production-ready implementation consolidating the most effective techniques for Task A. |

### Subtask B: Multi-class Authorship Detection

| Notebook | Description |
| :--- | :--- |
| `baselines/Kaggle_starters/Task-B-Baseline.ipynb` | Colab-optimized baseline for Task B, demonstrating HuggingFace dataset integration. |
| `Improved_Notebooks/task-b/Task-B-Colab.ipynb` | Refined experimental notebook for Task B with optimized data processing for Colab. |
| `modified_notebooks/task-b-phase-1 (1).ipynb` | Enhanced Task B model using **Weighted Cross-Entropy Loss** and **stratified subsampling** to handle class imbalance. |

### Subtask C & Utilities

| Notebook | Description |
| :--- | :--- |
| `baselines/Kaggle_starters/task-C-Baseline.ipynb` | Foundational 4-class classification baseline for Subtask C. |
| `Improved_Notebooks/HuggingFace_DataLoaders.ipynb` | Utility for efficient loading and tokenization of large-scale code datasets. |
| `Improved_Notebooks/Inference_Notebook.ipynb` | Dedicated workspace for running inference and generating competition submissions. |

---

## 🔍 Task Overview

The rise of generative models has made it increasingly difficult to distinguish machine-generated code from human-written code — especially across different programming languages, domains, and generation techniques. 

**SemEval-2026 Task 13** challenges participants to build systems that can **detect machine-generated code** under diverse conditions by evaluating generalization to unseen languages, generator families, and code application scenarios.

The task consists of **three subtasks**:

---

### Subtask A: Binary Machine-Generated Code Detection

**Goal:**  
Given a code snippet, predict whether it is:

- **(i)** Fully **human-written**, or  
- **(ii)** Fully **machine-generated**

**Training Languages:** `C++`, `Python`, `Java`  
**Training Domain:** `Algorithmic` (e.g., Leetcode-style problems)

**Evaluation Settings:**

| Setting                              | Language                | Domain                 |
|--------------------------------------|-------------------------|------------------------|
| (i) Seen Languages & Seen Domains    | C++, Python, Java       | Algorithmic            |
| (ii) Unseen Languages & Seen Domains | Go, PHP, C#, C, JS      | Algorithmic            |
| (iii) Seen Languages & Unseen Domains| C++, Python, Java       | Research, Production   |
| (iv) Unseen Languages & Domains      | Go, PHP, C#, C, JS      | Research, Production   |

**Dataset Size**: 
- Train - 500K samples (238K Human-Written | 262K Machine-Generated)
- Validation - 100K samples

**Target Metric** - Macro F1-score (we will build the leaderboard based on it), but you are free to use whatever works best for your approach during training.

---

###  Subtask B: Multi-Class Authorship Detection

**Goal:**  
Given a code snippet, predict its author:

- **(i)** Human  
- **(ii–xi)** One of 10 LLM families:
  - `DeepSeek-AI`, `Qwen`, `01-ai`, `BigCode`, `Gemma`, `Phi`, `Meta-LLaMA`, `IBM-Granite`, `Mistral`, `OpenAI`

**Evaluation Settings:**

- **Seen authors**: Test-time generators appeared in training  
- **Unseen authors**: Test-time generators are new but from known model families

**Dataset Size**: 
- Train - 500K samples (442K Human |4K DeepSeek-AI | 8K Qwen| 3K 01-ai |2 K BigCode |2K Gemma | 5K Phi | 8K Meta-LLaMA |8K IBM-Granite| 4K  Mistral   |10K OpenAI)
- Validation - 100K samples

**Target Metric** - Macro F1-score (we will build the leaderboard based on it), but you are free to use whatever works best for your approach during training.

---

### Subtask C: Hybrid Code Detection

**Goal:**  
Classify each code snippet as one of:

1. **Human-written**  
2. **Machine-generated**  
3. **Hybrid** — partially written or completed by LLM  
4. **Adversarial** — generated via adversarial prompts or RLHF to mimic humans

**Dataset Size**: 
- Train - 900K samples (485K Human-written | 210K Machine-generated |  85K Hybrid | 118K Adversarial)
- Validation - 200K samples

**Target Metric** - Macro F1-score (we will build the leaderboard based on it), but you are free to use whatever works best for your approach during training.

---

## 📁 Data Format

- All data will be released via:
  - [Kaggle](https://www.kaggle.com/datasets/daniilor/semeval-2026-task13)  
  - [HuggingFace Datasets](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13)
  - In this GitHub repo as `.parquet` file

- For each subtask:
  - Dataset contains `code`,  `label` (which is label id), and additional meta-data such as programming language (`language`), and the `generator`.
  - Label mappings (`label_to_id.json` and `id_to_label.json`) are provided in each task folder  


## Important Dates
- ~~Sample data ready: 15 July 2025~~
- ~~Training data ready: **1 September 2025**~~
- **Evaluation data ready: 1 December 2025** (we already released the training and validation datasets) 
- Evaluation data ready and evaluation start: 10 January 2026 (we will share private test data at this time)
- Evaluation end: 24 January 2026
- Paper submission due: 2nd of March 2026
- Notification to authors: April 2026
- Camera ready due April 2026
- SemEval workshop Summer 2026 (co-located with a major NLP conference)


## Citation
Our task is based on enriched data from our previous works. Please, consider citing them, when using data from this task

Droid: A Resource Suite for AI-Generated Code Detection
```
@inproceedings{orel2025droid,
  title={Droid: A resource suite for ai-generated code detection},
  author={Orel, Daniil and Paul, Indraneil and Gurevych, Iryna and Nakov, Preslav},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={31251--31277},
  year={2025}
}
```

CoDet-M4: Detecting Machine-Generated Code in Multi-Lingual, Multi-Generator and Multi-Domain Settings
```
@inproceedings{orel-etal-2025-codet,
    title = "{C}o{D}et-M4: Detecting Machine-Generated Code in Multi-Lingual, Multi-Generator and Multi-Domain Settings",
    author = "Orel, Daniil  and
      Azizov, Dilshod  and
      Nakov, Preslav",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.550/",
    pages = "10570--10593",
    ISBN = "979-8-89176-256-5",
}
```

