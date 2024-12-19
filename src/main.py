from viterbi import viterbi_algorithm
from preprocess import load_conllu_data, load_reviews_with_ud_pos
from train import calculate_probabilities_from_data
from evaluate import evaluate_accuracy, evaluate_common_errors, evaluate_errors_by_sentence_length, evaluate_errors_for_rare_words
from pathlib import Path

# Define file paths for training and testing data
data_paths = {
    "train": {
        "EWT": "../data/UD_English-EWT/en_ewt-ud-train.conllu",
        "GUM": "../data/UD_English-GUM/en_gum-ud-train.conllu"
    },
    "test": {
        "EWT": "../data/UD_English-EWT/en_ewt-ud-test.conllu",
        "GUM": "../data/UD_English-GUM/en_gum-ud-test.conllu",
        "reviews": "../data/informal_movie_reviews"
    }
}

def load_data(file_path, is_reviews):
    """Load data based on file type."""
    if is_reviews:
        return load_reviews_with_ud_pos(file_path)
    else:
        return load_conllu_data(file_path)

def main(train_dataset="EWT", test_dataset="reviews"):
    """Main function to train and evaluate the POS tagger."""
    # Select file paths
    train_file_path = data_paths["train"].get(train_dataset)
    test_file_path = data_paths["test"].get(test_dataset)

    if not train_file_path or not test_file_path:
        raise ValueError("Invalid dataset selection. Check train_dataset or test_dataset.")

    # Determine if the test dataset is informal movie reviews
    is_reviews = Path(test_file_path).suffix == ""  # Movie reviews folder has no file extension

    # Load data
    train_data = load_conllu_data(train_file_path)
    test_data = load_data(test_file_path, is_reviews)

    # Train model
    initial_probs, transition_probs, emission_probs = calculate_probabilities_from_data(train_data)

    # Evaluate model
    evaluate_accuracy(initial_probs, transition_probs, emission_probs, test_data)
    evaluate_common_errors(initial_probs, transition_probs, emission_probs, test_data)
    evaluate_errors_by_sentence_length(initial_probs, transition_probs, emission_probs, test_data)
    evaluate_errors_for_rare_words(train_data, initial_probs, transition_probs, emission_probs, test_data)

if __name__ == "__main__":
    # Modify arguments here to switch datasets
    # train_dataset can be "EWT" or "GUM"
    # test_dataset can be "EWT", "GUM" or "reviews"
    main(train_dataset="GUM", test_dataset="reviews")
