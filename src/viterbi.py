import numpy as np

def viterbi_algorithm(sentence, initial_probs, transition_probs, emission_probs):
    # List of unique tags (assuming that the transition_probs and emission_probs contain all possible tags)
    tags = list(initial_probs.keys())
    
    # Initialize the Viterbi table (dynamic programming table)
    V = [{} for _ in range(len(sentence))]  # V[i] stores the probabilities for each tag at word i
    backpointer = [{} for _ in range(len(sentence))]  # backpointer[i] stores the best tag at word i-1
    
    # Step 1: Initialization (for the first word)
    for tag in tags:
        V[0][tag] = initial_probs.get(tag, 0) * emission_probs.get(tag, {}).get(sentence[0], 0)
        backpointer[0][tag] = None  # No previous tag for the first word
    
    # Step 2: Recursion (for subsequent words)
    for t in range(1, len(sentence)):
        for current_tag in tags:
            # Find the best previous tag that maximizes the probability
            max_prob = -np.inf
            best_prev_tag = None
            for prev_tag in tags:
                prob = V[t-1][prev_tag] * transition_probs.get(prev_tag, {}).get(current_tag, 0) * emission_probs.get(current_tag, {}).get(sentence[t], 0)
                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = prev_tag
            V[t][current_tag] = max_prob
            backpointer[t][current_tag] = best_prev_tag
    
    # Step 3: Termination (find the best tag for the last word)
    best_final_tag = max(tags, key=lambda tag: V[len(sentence)-1].get(tag, 0))
    
    # Step 4: Backtracking (retrieve the sequence of tags)
    best_tags = [best_final_tag]
    for t in range(len(sentence)-1, 0, -1):
        best_tags.insert(0, backpointer[t][best_tags[0]])

    return best_tags
