from viterbi import viterbi_algorithm
from preprocess import load_conllu_data
from train import calculate_probabilities_from_data
from collections import Counter

def evaluate_pos_tagger_accuracy(initial_probs, transition_probs, emission_probs, test_file_path):
    # Load the test data
    test_data = load_conllu_data(test_file_path)
    
    # Initialize variables to compute accuracy
    total_tags = 0
    correct_tags = 0
    
    # Predict POS tags for the test data
    for test_sentence_words, test_sentence_tags in test_data:
        predicted_tags = viterbi_algorithm(test_sentence_words, initial_probs, transition_probs, emission_probs)
        for predicted_tag, actual_tag in zip(predicted_tags, test_sentence_tags):
            total_tags += 1
            if predicted_tag == actual_tag:
                correct_tags += 1
    
    # Calculate accuracy
    accuracy = correct_tags / total_tags if total_tags > 0 else 0
    print("Accutacy of the model: {:f}".format(accuracy))
    return accuracy


def evaluate_pos_tagger_common_errors(initial_probs, transition_probs, emission_probs, test_file_path):
    # Load the test data
    test_data = load_conllu_data(test_file_path)
    
    # Initialize a counter to track errors
    error_counter = Counter()
    
    # Predict POS tags for the test data
    for test_sentence_words, test_sentence_tags in test_data:
        predicted_tags = viterbi_algorithm(test_sentence_words, initial_probs, transition_probs, emission_probs)
        for predicted_tag, actual_tag in zip(predicted_tags, test_sentence_tags):
            if predicted_tag != actual_tag:
                error_counter[(actual_tag, predicted_tag)] += 1
    
    # Identify the 5 most common errors
    most_common_errors = error_counter.most_common(5)
    
    # Print the results in a structured format
    print("Top 5 Most Frequent Errors:")
    for i, ((actual_tag, predicted_tag), count) in enumerate(most_common_errors, 1):
        print(f"{i}. Predicted '{predicted_tag}' instead of '{actual_tag}' {count} times.")
    
    return most_common_errors


def evaluate_pos_tagger_errors_by_sentence_length(initial_probs, transition_probs, emission_probs, test_file_path):
    # Load the test data
    test_data = load_conllu_data(test_file_path)
    
    # Define sentence length intervals
    length_intervals = [(1, 5), (6, 10), (11, 20), (21, float('inf'))]
    interval_stats = {interval: {'total': 0, 'errors': 0} for interval in length_intervals}
    
    # Predict POS tags for the test data and count errors
    for test_sentence_words, test_sentence_tags in test_data:
        predicted_tags = viterbi_algorithm(test_sentence_words, initial_probs, transition_probs, emission_probs)
        errors = sum(1 for predicted_tag, actual_tag in zip(predicted_tags, test_sentence_tags) if predicted_tag != actual_tag)
        total_tags = len(test_sentence_words)
        
        # Determine the interval the sentence length falls into
        sentence_length = len(test_sentence_words)
        for interval in length_intervals:
            if interval[0] <= sentence_length <= interval[1]:
                interval_stats[interval]['total'] += total_tags
                interval_stats[interval]['errors'] += errors
                break
    
    # Calculate and display error rates by interval
    print("Error rates by sentence length:")
    for interval, stats in interval_stats.items():
        total = stats['total']
        errors = stats['errors']
        error_rate = errors / total if total > 0 else 0
        print(f"Sentence length {interval[0]}-{int(interval[1]) if interval[1] != float('inf') else '∞'}: {error_rate:.2%} error rate ({errors}/{total} tags)")
    
    return interval_stats
