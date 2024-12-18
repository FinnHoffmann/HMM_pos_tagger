from viterbi import viterbi_algorithm
from preprocess import load_conllu_data
from train import calculate_probabilities_from_data
from evaluate import evaluate_pos_tagger_accuracy, evaluate_pos_tagger_common_errors, evaluate_pos_tagger_errors_by_sentence_length, evaluate_pos_tagger_errors_for_rare_words

file_path_train_ewt = "../data/UD_English-EWT/en_ewt-ud-train.conllu"
file_path_test_ewt = "../data/UD_English-EWT/en_ewt-ud-test.conllu"

file_path_train_gum = "../data/UD_English-GUM/en_gum-ud-train.conllu"
file_path_test_gum = "../data/UD_English-GUM/en_gum-ud-test.conllu"

data = load_conllu_data(file_path_train_gum)
initial_probs, transition_probs, emission_probs = calculate_probabilities_from_data(data)

evaluate_pos_tagger_accuracy(initial_probs, transition_probs, emission_probs, file_path_test_gum)
evaluate_pos_tagger_common_errors(initial_probs, transition_probs, emission_probs, file_path_test_gum)
evaluate_pos_tagger_errors_by_sentence_length(initial_probs, transition_probs, emission_probs, file_path_test_gum)
# evaluate_pos_tagger_errors_for_rare_words(file_path_train, file_path_test)

