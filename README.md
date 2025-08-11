# Text_Summarizer_project

Text Summarizer Project (PEGASUS on SAMSum)
This project implements a text summarization model using the PEGASUS model from the Hugging Face transformers library. The model is fine-tuned on the SAMSum dataset, which consists of conversational dialogues and their summaries.
This project focuses on building and evaluating a text summarization model specifically for conversational data using the PEGASUS model and the SAMSum dataset.

# Table of Contents

Project Description

Dataset

Model

Setup and Installation

Training

Evaluation

Inference

Results

Future Work

Contributing

License

# Project Description

The primary objective of this project is to develop an effective abstractive text summarization system for informal, conversational text. We leverage the power of the PEGASUS model, a state-of-the-art sequence-to-sequence model pre-trained for summarization tasks, and fine-tune it on the SAMSum dataset, which comprises thousands of messenger-like dialogues and their human-written summaries. This fine-tuning process adapts the model to the specific characteristics of conversational language, such as informal phrasing, interjections, and shorter sentence structures.
The goal of this project is to create a model that can generate concise summaries of conversational text. The PEGASUS model is chosen for its strong performance on abstractive summarization tasks. The SAMSum dataset provides a good source of data for training a model specifically on dialogues.



# Dataset

The dataset used in this project is the SAMSum Dataset.

Source: Hugging Face Datasets library (samsum)
Content: The dataset contains pairs of messenger-like dialogues and their corresponding summaries.
Splits: It is divided into training, validation, and test sets.
Training: 14732 examples
Validation: 818 examples
Test: 819 examples
Characteristics: The dialogues are relatively short and informal, reflecting real-world conversations. The summaries are concise and capture the main points of the dialogue.
Model
The model used is PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive Summarization).

Architecture: PEGASUS is a transformer-based encoder-decoder model.
Pre-training Objective: It was pre-trained using a unique objective where whole sentences are masked from the input text and the model is trained to generate these masked sentences. This pre-training task aligns well with the summarization task.

Checkpoint: We use the google/pegasus-cnn_dailymail checkpoint as the starting point for fine-tuning. Although this checkpoint was pre-trained on news articles, its strong summarization capabilities transfer well to other domains with fine-tuning.

# Setup and Installation

Clone the repository (if applicable)

To run this project, you need to have Python and the necessary libraries installed. You can install the required libraries using pip:

import nltk

nltk.download("punkt")

    !pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr -q
    
    !pip install --upgrade accelerate
    
    !pip uninstall -y transformers accelerate # Uninstall existing versions
    
    !pip install transformers accelerate # Install compatible versions
