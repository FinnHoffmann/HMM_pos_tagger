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
    # Load Universal Dependencies model (English pipeline by default)
    stanza.download('en')  # Download the English model if not already done
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_no_ssplit=True)

    # Read all .txt files in the folder
    sentences = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                # Combine all lines in the file into a single string
                text = file.read().strip()
                # Split the text into sentences using Stanza's tokenizer
                doc = nlp(text)
                for sentence in doc.sentences:
                    # Collect tokens (words) for each sentence
                    words = [token.text for token in sentence.tokens]
                    sentences.append(words)

    # Annotate sentences with POS tags
    tagged_data = []
    for sentence in sentences:
        # Process the sentence using the Stanza pipeline
        doc = nlp(" ".join(sentence))  # Join words for processing as a single sentence
        words = []
        tags = []
        for sent in doc.sentences:
            for word in sent.words:
                words.append(word.text)
                tags.append(word.upos)  # Use Universal POS tags
        tagged_data.append((words, tags))

    # Return the tagged data
    return tagged_data
