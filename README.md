# POS Tagger Project

This project develops a POS tagger and evaluates it using different datasets.

## Data Structure

The data is organized as follows:

- **Training Data**:
  - `EWT`: `../data/UD_English-EWT/en_ewt-ud-train.conllu`
  - `GUM`: `../data/UD_English-GUM/en_gum-ud-train.conllu`

- **Test Data**:
  - `EWT`: `../data/UD_English-EWT/en_ewt-ud-test.conllu`
  - `GUM`: `../data/UD_English-GUM/en_gum-ud-test.conllu`
  - `Reviews`: `../data/informal_movie_reviews` (folder without file extension)

## Usage

The `main` function trains and evaluates the model using the specified datasets:

```python
def main(train_dataset="EWT", test_dataset="reviews"):
