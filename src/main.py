from viterbi import viterbi_algorithm
from preprocess import load_conllu_data
from train import calculate_probabilities_from_data
from evaluate import evaluate_pos_tagger, evaluate_errors

file_path_train = "../data/UD_English-EWT/en_ewt-ud-train.conllu"
file_path_test = "../data/UD_English-EWT/en_ewt-ud-test.conllu"

# sentence = ['I', 'love', 'Python']
# data = load_conllu_data(file_path)
# initial_probs, transition_probs, emission_probs = calculate_probabilities_from_data(data)
# tags = viterbi_algorithm(sentence, initial_probs, transition_probs, emission_probs)
# print("Sentence:", sentence)
# print("POS Tags:", tags)

# print(evaluate_pos_tagger(file_path_train,file_path_test))

evaluate_errors(file_path_train, file_path_test)

