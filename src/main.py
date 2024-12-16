from viterbi import viterbi_algorithm
from preprocess import load_conllu_data
from train import calculate_probabilities_from_data


sentence = ['I', 'love', 'Python']
file_path = "../data/UD_English-EWT/en_ewt-ud-train.conllu"
data = load_conllu_data(file_path)

initial_probs, transition_probs, emission_probs = calculate_probabilities_from_data(data)

tags = viterbi_algorithm(sentence, initial_probs, transition_probs, emission_probs)

print("Sentence:", sentence)
print("POS Tags:", tags)


