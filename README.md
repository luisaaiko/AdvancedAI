# AdvancedAI: Hotel Review Sentiment Analysis

Final project for comparing traditional machine learning, recurrent neural networks, and BERT-based models for hotel review sentiment analysis.

## Project overview

This project follows the abstract **Comparing BERT and Aspect-Aware BERT Models for Hotel Review Sentiment Analysis**. The goal is to predict overall customer satisfaction from hotel review text and evaluate whether aspect-level information improves performance beyond standard sentiment classification.

The current code workflow focuses on reproducible experiments:

1. Preprocess the hotel review data.
2. Explore class balance, missing values, rating patterns, and review text.
3. Train traditional TF-IDF baselines.
4. Train a recurrent neural network baseline.
5. Train a stronger BiLSTM recurrent model.
6. Fine-tune BERT and an aspect-enhanced BERT model.

## Repository structure

```text
.
├── datasets/
│   ├── hotel_reviews.csv                  # Original Kaggle dataset, not committed if too large
│   ├── clean_hotel_reviews.csv            # Produced by preprocessing notebook
│   ├── train_hotel_reviews.csv            # Produced by preprocessing notebook
│   └── test_hotel_reviews.csv             # Produced by preprocessing notebook
├── results/                               # Metrics, predictions, histories, and plots
├── models/                                # Saved model weights/checkpoints
├── 1preprocessing.ipynb                   # Data cleaning and train/test split
├── 2eda.ipynb                             # Exploratory data analysis
├── 3baselineModel.ipynb                   # TF-IDF classical baselines
├── 4rnn_model.ipynb                       # Simple RNN text classifier baseline
├── 5lstm_model.ipynb                      # BiLSTM text classifier
├── 6bert_model.ipynb                      # BERT text-only and aspect-enhanced models
└── README.md
```

## Dataset

The project uses the Kaggle Hotel Reviews Dataset derived from TripAdvisor hotel reviews. Each row represents one hotel review and includes review text, overall satisfaction/rating information, and aspect-level ratings such as service, cleanliness, location, room quality, and sleep quality.

Expected raw file location:

```text
datasets/hotel_reviews.csv
```

The preprocessing notebook creates the cleaned train/test files used by all later notebooks.

## Models included

### 1. Traditional baselines

`3baselineModel.ipynb` trains classical text classifiers using TF-IDF features. These baselines provide a non-neural comparison point.

### 2. RNN baseline

`4rnn_model.ipynb` trains a simple recurrent neural network using an embedding layer followed by a `SimpleRNN` encoder. This replaces the previous CNN notebook so the comparison is aligned with sequence modeling for text.

Saved outputs:

```text
models/rnn_hotel_sentiment.weights.h5
results/rnn_training_history.csv
results/rnn_results.csv
results/rnn_test_predictions.csv
```

### 3. BiLSTM model

`5lstm_model.ipynb` trains a bidirectional LSTM model. This is a stronger recurrent architecture than the simple RNN because LSTM gates help preserve longer-range dependencies in review text.

Saved outputs:

```text
models/bilstm_hotel_sentiment.weights.h5
results/lstm_training_history.csv
results/lstm_results.csv
results/lstm_test_predictions.csv
```

### 4. BERT and aspect-enhanced BERT

`6bert_model.ipynb` fine-tunes BERT for text-only sentiment classification and compares it with an aspect-enhanced version that incorporates structured hotel aspect ratings.

Recommended pretrained checkpoint:

```text
google-bert/bert-base-uncased
```

## Setup

Recommended environment: Python 3.10+.

Install core dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow torch transformers tqdm
```

For GPU acceleration, install the PyTorch and TensorFlow versions appropriate for your CUDA setup.

## Reproducing results

Run the notebooks in this order:

1. `1preprocessing.ipynb`
2. `2eda.ipynb`
3. `3baselineModel.ipynb`
4. `4rnn_model.ipynb`
5. `5lstm_model.ipynb`
6. `6bert_model.ipynb`

The most important outputs for the final paper are the model metrics saved in `results/`, especially accuracy, precision, recall, F1-score, and ROC-AUC.

## Notes on Windows Unicode issues

Some review text contains private-use Unicode characters such as `\uf0d8`. The RNN and LSTM notebooks clean these characters before vectorization and save weights only instead of full Keras `.keras` models. This avoids Windows `cp1252` encoding errors during training and saving.

## Final report usage

Use the `results/` files to build the final comparison table for the paper:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF baseline | TBD | TBD | TBD | TBD | TBD |
| RNN | TBD | TBD | TBD | TBD | TBD |
| BiLSTM | TBD | TBD | TBD | TBD | TBD |
| BERT text-only | TBD | TBD | TBD | TBD | TBD |
| Aspect-enhanced BERT | TBD | TBD | TBD | TBD | TBD |

The final paper should discuss not only which model scores highest but also where each model fails, especially on mixed reviews where some aspects are positive and others are negative.
