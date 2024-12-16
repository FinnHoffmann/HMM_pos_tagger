from viterbi import viterbi_algorithm
from preprocess import load_conllu_data
from train import calculate_probabilities_from_data

def evaluate_pos_tagger(train_file_path, test_file_path):
    # Step 1: Load the training and test data
    train_data = load_conllu_data(train_file_path)
    test_data = load_conllu_data(test_file_path)
    
    # Step 2: Train the model on the training data
    initial_probs, transition_probs, emission_probs = calculate_probabilities_from_data(train_data)
    
    # Step 3: Initialize variables to compute accuracy
    total_tags = 0
    correct_tags = 0
    
    # Step 4: Predict POS tags for the test data
    for test_sentence_words, test_sentence_tags in test_data:
        # Predict the POS tags for the test sentence
        predicted_tags = viterbi_algorithm(test_sentence_words, initial_probs, transition_probs, emission_probs)
        
        # Compare predicted tags with actual tags
        for predicted_tag, actual_tag in zip(predicted_tags, test_sentence_tags):
            total_tags += 1
            if predicted_tag == actual_tag:
                correct_tags += 1
    
    # Step 5: Calculate accuracy
    accuracy = correct_tags / total_tags if total_tags > 0 else 0
    return accuracy
