import conllu
import os
import stanza

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


def load_reviews_with_ud_pos(folder_path):
    # Initialize the Stanza model
    stanza.download('en', verbose=False)  # Download the model (only needed once)
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', verbose=False)  # Enable sentence splitting and POS tagging

    data = []

    # Read all .txt files in the specified folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                text = file.read().strip()
                # Process the text
                doc = nlp(text)
                for sentence in doc.sentences:
                    # Split the sentence into words and tags
                    sentence_words = [word.text for word in sentence.words]
                    sentence_pos_tags = [word.upos for word in sentence.words]
                    # Append the sentence as a tuple (words, tags) to the data
                    data.append((sentence_words, sentence_pos_tags))

    return data