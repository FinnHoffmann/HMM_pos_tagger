import conllu

def load_conllu_data(file_path):
    data = []
    
    # Open and parse the CoNLL-U file
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = conllu.parse(file.read())  # Parse the entire file
        
        # Loop through each sentence
        for sentence in sentences:
            sentence_words = []
            sentence_pos_tags = []
            
            # Loop through each token in the sentence
            for token in sentence:
                word = token['form']  # The word is in the 'form' key
                pos_tag = token['upostag']  # The POS tag is in the 'upostag' key
                
                sentence_words.append(word)
                sentence_pos_tags.append(pos_tag)
            
            # Append the sentence's words and POS tags as a tuple to the data
            data.append((sentence_words, sentence_pos_tags))
    
    return data



if __name__ == "__main__":
    file_path = "../data/UD_English-EWT/en_ewt-ud-train.conllu"

   # Load the data from the CoNLL-U file
    data = load_conllu_data(file_path)

    # Access the first sentence (first tuple in the list)
    first_sentence_words, first_sentence_pos_tags = data[0]

    # Print the words and POS tags for the first sentence
    print("Words:", first_sentence_words)
    print("POS Tags:", first_sentence_pos_tags)
