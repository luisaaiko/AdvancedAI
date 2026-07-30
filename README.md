# AdvancedAI: Hotel Review Sentiment Analysis

Final project repository for **Hotel Review Sentiment Analysis: A Comparison of RNN, LSTM, and BERT Architectures**.

## Project overview

This project compares traditional machine learning baselines, recurrent neural network models, and a pretrained BERT classifier for binary sentiment classification of hotel reviews. The task predicts whether each review expresses negative or positive overall customer satisfaction.

The project follows the abstract direction of comparing BERT-based sentiment modeling against simpler alternatives, while also documenting the performance tradeoff between lightweight sequence models and transformer-based language models.

## Team members

- Jessica Wentworth
- Luisa Ribeiro
- Matheus Lucena

## Dataset

The project uses the Kaggle Hotel Reviews Dataset derived from TripAdvisor hotel reviews. The raw dataset contains review text, overall ratings, and aspect-level hotel ratings. The preprocessing notebook creates a binary sentiment label from the overall rating:

- Ratings 1-3: negative sentiment (`0`)
- Ratings 4-5: positive sentiment (`1`)

Expected raw file location:

```text
datasets/hotel_reviews.csv
```

The preprocessing notebook writes the cleaned files used by later notebooks:

```text
datasets/clean_hotel_reviews.csv
datasets/train_hotel_reviews.csv
datasets/test_hotel_reviews.csv
```

## Repository structure

```text
.
├── datasets/
│   ├── hotel_reviews.csv
│   ├── clean_hotel_reviews.csv
│   ├── train_hotel_reviews.csv
│   └── test_hotel_reviews.csv
├── models/
│   └── saved model weights and checkpoints
├── results/
│   ├── baseline_results.csv
│   ├── baseline_roc_curves.png
│   ├── baseline_test_predictions.csv
│   ├── rnn_results.csv
│   ├── rnn_training_history.csv
│   ├── rnn_test_predictions.csv
│   ├── lstm_results.csv
│   ├── lstm_training_history.csv
│   └── lstm_test_predictions.csv
├── 1preprocessing.ipynb
├── 2eda.ipynb
├── 3baselineModel.ipynb
├── 4rnn_model.ipynb
├── 5lstm_model.ipynb
├── 6bert_model.ipynb
└── README.md
```

## Notebook workflow

Run the notebooks in this order:

1. **`1preprocessing.ipynb`**  
   Loads the raw hotel review dataset, removes neutral reviews, creates the binary sentiment target, handles missing values, and creates the stratified train/test split.

2. **`2eda.ipynb`**  
   Explores class balance, rating distributions, review length, text characteristics, and common terms in the corpus.

3. **`3baselineModel.ipynb`**  
   Trains TF-IDF baselines using Logistic Regression, Multinomial Naive Bayes, and Random Forest. It saves baseline metrics, predictions, and ROC curves.

4. **`4rnn_model.ipynb`**  
   Trains a SimpleRNN text classifier using `TextVectorization`, an embedding layer, and a recurrent layer. This is the lightweight neural sequence baseline.

5. **`5lstm_model.ipynb`**  
   Trains a Bidirectional LSTM model using the same train/test split and sequence length as the RNN model. This tests whether gated recurrent memory improves performance on longer hotel reviews.

6. **`6bert_model.ipynb`**  
   Fine-tunes a pretrained BERT classifier for binary sentiment classification. The recommended checkpoint is:

```text
google-bert/bert-base-uncased
```

## Models included

### Traditional TF-IDF baselines

The classical baselines use TF-IDF text features. Logistic Regression provides a strong linear benchmark, Naive Bayes provides a probabilistic text baseline, and Random Forest provides a nonlinear tree-based comparison.

### SimpleRNN

The RNN model uses:

```text
TextVectorization -> Embedding(128) -> SimpleRNN(64) -> Dropout -> Dense -> Sigmoid
```

This model is included as a lightweight recurrent baseline. It processes reviews sequentially but has limited ability to retain information across long reviews.

### BiLSTM

The BiLSTM model uses:

```text
TextVectorization -> Embedding(128) -> Bidirectional LSTM(64) -> Dropout -> Dense -> Sigmoid
```

This model improves on the SimpleRNN by using gated memory and bidirectional context.

### BERT

The BERT model uses pretrained contextual WordPiece representations and a classification head for binary sentiment prediction. It is expected to perform best but requires more compute and memory.

## Current result summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.8648 | 0.8800 | 0.9387 | 0.9084 | 0.9259 |
| TF-IDF + Naive Bayes | 0.8272 | 0.8237 | 0.9644 | 0.8885 | 0.8936 |
| TF-IDF + Random Forest | 0.8220 | 0.8107 | 0.9794 | 0.8871 | 0.8942 |
| RNN Text Classifier | 0.6463 | 0.8174 | 0.6497 | 0.7240 | 0.7050 |
| BiLSTM Text Classifier | 0.8529 | 0.9303 | 0.8581 | 0.8928 | 0.9294 |
| BERT Text Classifier | 0.8942 | 0.9044 | 0.9526 | 0.9278 | 0.9523 |


The strongest traditional baseline is TF-IDF with Logistic Regression. The BiLSTM substantially improves over the SimpleRNN, and BERT achieves the strongest overall ROC-AUC and F1 among the models currently reported.

## Output files

### `models/`

The `models/` folder stores trained model weights/checkpoints. These files allow models to be reloaded later without retraining.

### `results/`

The `results/` folder stores metrics, predictions, training histories, and figures. These are the main files used to build the final report tables, ROC curves, confusion matrices, and error analysis.

Important result files include:

- `baseline_results.csv`: metrics for Logistic Regression, Naive Bayes, and Random Forest.
- `baseline_roc_curves.png`: ROC curve comparison for traditional baselines.
- `rnn_results.csv`: final RNN test metrics.
- `rnn_training_history.csv`: epoch-by-epoch RNN training and validation metrics.
- `rnn_test_predictions.csv`: RNN predictions on the test set.
- `lstm_results.csv`: final BiLSTM test metrics.
- `lstm_training_history.csv`: epoch-by-epoch BiLSTM training and validation metrics.
- `lstm_test_predictions.csv`: BiLSTM predictions on the test set.

## Setup

Recommended Python version: **Python 3.10+**.

Install core dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow torch transformers datasets tqdm
```

For GPU acceleration, install PyTorch and TensorFlow versions that match your CUDA setup.

## Reproducibility notes

- All notebooks use the same processed train/test split created by `1preprocessing.ipynb`.
- The RNN and BiLSTM notebooks use class weights to reduce the effect of class imbalance.
- Some review text contains private-use Unicode characters such as `\uf0d8`. The RNN and LSTM notebooks sanitize these characters before vectorization to avoid Windows `cp1252` encoding errors.
- Keras recurrent models save weights only (`.weights.h5`) instead of full `.keras` models to avoid saving problematic text-vectorizer vocabulary on Windows.

## Deprecated / not part of final comparison

Earlier CNN files or CNN result artifacts may still exist in the local project history. CNNs can be used for text classification with 1D convolutions, but the final project comparison is now focused on:

```text
TF-IDF baselines -> SimpleRNN -> BiLSTM -> BERT
```

For the final paper, use the RNN and BiLSTM files rather than the older CNN outputs.

## Suggested final report usage

Use the saved `results/` files to support the **Experiments**, **Results & Discussion**, and **Error Analysis** sections. The final paper should compare models quantitatively using accuracy, precision, recall, F1, and ROC-AUC, and qualitatively examine misclassified hotel reviews with mixed or ambiguous sentiment.
