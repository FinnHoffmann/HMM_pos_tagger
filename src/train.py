from collections import defaultdict
from preprocess import load_conllu_data

def calculate_probabilities_from_data(data):
    # Data structure to hold counts
    transition_counts = defaultdict(lambda: defaultdict(int))  # transition_counts[tag1][tag2] = count
    emission_counts = defaultdict(lambda: defaultdict(int))    # emission_counts[tag][word] = count
    initial_counts = defaultdict(int)                           # initial_counts[tag] = count
    tag_counts = defaultdict(int)                                # tag_counts[tag] = count
    total_sentences = 0                                          # To track total number of sentences
    
    # Loop through each sentence in the data
    for sentence_words, sentence_pos_tags in data:
        total_sentences += 1
        previous_tag = None
        
        # Loop through each token in the sentence
        for i, (word, pos_tag) in enumerate(zip(sentence_words, sentence_pos_tags)):
            # Count emission
            emission_counts[pos_tag][word] += 1
            tag_counts[pos_tag] += 1
            
            # Count transition
            if i == 0:  # First token in the sentence
                initial_counts[pos_tag] += 1
            else:  # For subsequent tokens
                transition_counts[previous_tag][pos_tag] += 1
            
            # Update previous tag
            previous_tag = pos_tag
    
    # Calculate probabilities from counts
    transition_probs = defaultdict(lambda: defaultdict(float))
    emission_probs = defaultdict(lambda: defaultdict(float))
    initial_probs = defaultdict(float)
    
    # Calculate transition probabilities
    for tag1 in transition_counts:
        total_transitions = sum(transition_counts[tag1].values())
        for tag2 in transition_counts[tag1]:
            transition_probs[tag1][tag2] = transition_counts[tag1][tag2] / total_transitions
    
    # Calculate emission probabilities
    for tag in emission_counts:
        total_emissions = tag_counts[tag]
        for word in emission_counts[tag]:
            emission_probs[tag][word] = emission_counts[tag][word] / total_emissions
    
    # Calculate initial probabilities
    for tag in initial_counts:
        initial_probs[tag] = initial_counts[tag] / total_sentences
    
    return initial_probs, transition_probs, emission_probs



file_path = "../data/UD_English-EWT/en_ewt-ud-train.conllu"

   # Load the data from the CoNLL-U file
data = load_conllu_data(file_path)
initial_probs, transition_probs, emission_probs = calculate_probabilities_from_data(data)

print(emission_probs)