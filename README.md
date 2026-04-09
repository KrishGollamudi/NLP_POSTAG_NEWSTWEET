# Advanced News-Aware POS Tagger

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces/Krishna9939/News-POS-Tagger)

**Live Deployment:** [Access the Web Application via Hugging Face Spaces](https://huggingface.co/spaces/Krishna9939/News-POS-Tagger)

## Project Overview

Standard Part-of-Speech (POS) taggers trained on formal linguistic datasets suffer significant accuracy degradation when processing journalistic tweets due to typographical noise (dropped determiners, compound hashtags, visual syntax). 

This project introduces a custom deep-learning architecture designed to bridge the domain gap between formal linguistics and chaotic social media data. By engineering a parallel embedding layer that forces the neural network to weigh the structural type of a token alongside its semantic meaning, this model achieves State-of-the-Art (95.86%) accuracy on journalistic tweets.

*The full training pipeline, custom PyTorch classes, and architectural ablation studies are available in the attached Jupyter Notebook.*

## Key Architectural Innovations

**1. Lexical Recovery via Semantic Preprocessing**
* **Visual-to-Semantic Mapping:** Dynamically converts visual markers into semantic text blocks.
* **Syntactic Hashtag Splitting:** Fractures Out-Of-Vocabulary (OOV) compound hashtags into constituent lexical units, recovering latent Proper Noun classifications.

**2. Typographical Self-Awareness (Structural Injection)**
* Every token is pre-classified into six discrete structural categories (News Marker, Hashtag, Mention, URL, Visual Syntax, Normal Word).
* This metadata is passed through a parallel `nn.Embedding` layer and merged into the transformer's hidden state via Isotropic Addition and Layer Normalization.

**3. Sub-Word Alignment and Differential Fine-Tuning**
* Implements a deterministic alignment mask (`-100`) to ensure Cross-Entropy Loss is calculated exclusively on primary sub-words.
* Utilizes a differential learning rate strategy to prevent catastrophic forgetting of the pre-trained RoBERTa backbone.

## Performance Metrics

| Model Architecture | POS Accuracy | Contextual Integration Method |
| :--- | :--- | :--- |
| Standard BERT (`bert-base-uncased`) | 94.62% | None |
| Plain BERTweet (`vinai/bertweet-base`) | 95.40% | Domain Pre-training Only |
| **Proposed Advanced Model** | **95.86%** | **Isotropic Addition + LayerNorm** |

## Installation and Local Usage
## Model Weights
Download the pre-trained weights (`advanced_pytorch_model.bin`) from [this Google Drive link](https://drive.google.com/file/d/1VuXXFTVmegOVJiJBNqpA5wCpFbPu6n-M/view?usp=drive_link) and place them in your root directory.
To deploy the Streamlit inference engine locally:

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/News-Aware-POS-Tagger.git](https://github.com/yourusername/News-Aware-POS-Tagger.git)
cd News-Aware-POS-Tagger
