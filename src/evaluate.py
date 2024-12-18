from viterbi import viterbi_algorithm
from preprocess import load_conllu_data
from train import calculate_probabilities_from_data
from collections import Counter

# def evaluate_pos_tagger_accuracy(initial_probs, transition_probs, emission_probs, test_file_path):
#     # Load the test data
#     test_data = load_conllu_data(test_file_path)
    
#     # Initialize variables to compute accuracy
#     total_tags = 0
#     correct_tags = 0
    
#     # Predict POS tags for the test data
#     for test_sentence_words, test_sentence_tags in test_data:
#         predicted_tags = viterbi_algorithm(test_sentence_words, initial_probs, transition_probs, emission_probs)
#         for predicted_tag, actual_tag in zip(predicted_tags, test_sentence_tags):
#             total_tags += 1
#             if predicted_tag == actual_tag:
#                 correct_tags += 1
    
#     # Calculate accuracy
#     accuracy = correct_tags / total_tags if total_tags > 0 else 0
#     print("Accutacy of the model: {:f}".format(accuracy))
#     return accuracy


# def evaluate_pos_tagger_common_errors(initial_probs, transition_probs, emission_probs, test_file_path):
#     # Load the test data
#     test_data = load_conllu_data(test_file_path)
    
#     # Initialize a counter to track errors
#     error_counter = Counter()
    
#     # Predict POS tags for the test data
#     for test_sentence_words, test_sentence_tags in test_data:
#         predicted_tags = viterbi_algorithm(test_sentence_words, initial_probs, transition_probs, emission_probs)
#         for predicted_tag, actual_tag in zip(predicted_tags, test_sentence_tags):
#             if predicted_tag != actual_tag:
#                 error_counter[(actual_tag, predicted_tag)] += 1
    
#     # Identify the 5 most common errors
#     most_common_errors = error_counter.most_common(5)
    
#     # Print the results in a structured format
#     print("Top 5 Most Frequent Errors:")
#     for i, ((actual_tag, predicted_tag), count) in enumerate(most_common_errors, 1):
#         print(f"{i}. Predicted '{predicted_tag}' instead of '{actual_tag}' {count} times.")
    
#     return most_common_errors


# def evaluate_pos_tagger_errors_by_sentence_length(initial_probs, transition_probs, emission_probs, test_file_path):
#     # Load the test data
#     test_data = load_conllu_data(test_file_path)
    
#     # Define sentence length intervals
#     length_intervals = [(1, 5), (6, 10), (11, 20), (21, float('inf'))]
#     interval_stats = {interval: {'total': 0, 'errors': 0} for interval in length_intervals}
    
#     # Predict POS tags for the test data and count errors
#     for test_sentence_words, test_sentence_tags in test_data:
#         predicted_tags = viterbi_algorithm(test_sentence_words, initial_probs, transition_probs, emission_probs)
#         errors = sum(1 for predicted_tag, actual_tag in zip(predicted_tags, test_sentence_tags) if predicted_tag != actual_tag)
#         total_tags = len(test_sentence_words)
        
#         # Determine the interval the sentence length falls into
#         sentence_length = len(test_sentence_words)
#         for interval in length_intervals:
#             if interval[0] <= sentence_length <= interval[1]:
#                 interval_stats[interval]['total'] += total_tags
#                 interval_stats[interval]['errors'] += errors
#                 break
    
#     # Calculate and display error rates by interval
#     print("Error rates by sentence length:")
#     for interval, stats in interval_stats.items():
#         total = stats['total']
#         errors = stats['errors']
#         error_rate = errors / total if total > 0 else 0
#         print(f"Sentence length {interval[0]}-{int(interval[1]) if interval[1] != float('inf') else '∞'}: {error_rate:.2%} error rate ({errors}/{total} tags)")
    
#     return interval_stats

# def evaluate_pos_tagger_errors_for_rare_words(train_file_path, test_file_path):
#     # Load the training and test data
#     train_data = load_conllu_data(train_file_path)
#     test_data = load_conllu_data(test_file_path)
    
#     # Count word frequencies in the training data
#     word_counter = Counter(word for sentence_words, _ in train_data for word in sentence_words)
    
#     # Categorize words by frequency
#     rare_word_categories = {
#         "1-2 occurrences": set(word for word, count in word_counter.items() if 1 <= count <= 2),
#         "3-5 occurrences": set(word for word, count in word_counter.items() if 3 <= count <= 5),
#         "6-10 occurrences": set(word for word, count in word_counter.items() if 6 <= count <= 10),
#     }
    
#     # Train the model on the training data
#     initial_probs, transition_probs, emission_probs = calculate_probabilities_from_data(train_data)
    
#     # Initialize error statistics for each category
#     error_stats = {category: {"total": 0, "errors": 0} for category in rare_word_categories}
    
#     # Predict POS tags for the test data
#     for test_sentence_words, test_sentence_tags in test_data:
#         predicted_tags = viterbi_algorithm(test_sentence_words, initial_probs, transition_probs, emission_probs)
        
#         # Evaluate errors for each word in the sentence
#         for word, predicted_tag, actual_tag in zip(test_sentence_words, predicted_tags, test_sentence_tags):
#             for category, word_set in rare_word_categories.items():
#                 if word in word_set:
#                     error_stats[category]["total"] += 1
#                     if predicted_tag != actual_tag:
#                         error_stats[category]["errors"] += 1
#                     break
    
#     # Calculate and display error rates for each category
#     print("Error rates for rare words:")
#     for category, stats in error_stats.items():
#         total = stats["total"]
#         errors = stats["errors"]
#         error_rate = errors / total if total > 0 else 0
#         print(f"{category}: {error_rate:.2%} error rate ({errors}/{total} words)")
    
#     return error_stats

def evaluate_accuracy(initial_probs, transition_probs, emission_probs, test_data):
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
    print(f"Accuracy of the model: {accuracy:.2%}")
    return accuracy

def evaluate_common_errors(initial_probs, transition_probs, emission_probs, test_data):
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

def evaluate_errors_by_sentence_length(initial_probs, transition_probs, emission_probs, test_data):
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

def evaluate_errors_for_rare_words(train_data, initial_probs, transition_probs, emission_probs, test_data):
    # Count word frequencies in the training data
    word_counter = Counter(word for sentence_words, _ in train_data for word in sentence_words)
    
    # Categorize words by frequency
    rare_word_categories = {
        "1-2 occurrences": set(word for word, count in word_counter.items() if 1 <= count <= 2),
        "3-5 occurrences": set(word for word, count in word_counter.items() if 3 <= count <= 5),
        "6-10 occurrences": set(word for word, count in word_counter.items() if 6 <= count <= 10),
    }
    
    # Initialize error statistics for each category
    error_stats = {category: {"total": 0, "errors": 0} for category in rare_word_categories}
    
    # Predict POS tags for the test data
    for test_sentence_words, test_sentence_tags in test_data:
        predicted_tags = viterbi_algorithm(test_sentence_words, initial_probs, transition_probs, emission_probs)
        
        # Evaluate errors for each word in the sentence
        for word, predicted_tag, actual_tag in zip(test_sentence_words, predicted_tags, test_sentence_tags):
            for category, word_set in rare_word_categories.items():
                if word in word_set:
                    error_stats[category]["total"] += 1
                    if predicted_tag != actual_tag:
                        error_stats[category]["errors"] += 1
                    break
    
    # Calculate and display error rates for each category
    print("Error rates for rare words:")
    for category, stats in error_stats.items():
        total = stats["total"]
        errors = stats["errors"]
        error_rate = errors / total if total > 0 else 0
        print(f"{category}: {error_rate:.2%} error rate ({errors}/{total} words)")
    
    return error_stats
