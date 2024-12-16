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

