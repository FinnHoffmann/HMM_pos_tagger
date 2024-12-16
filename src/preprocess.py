import os

def load_conllu_file(file_path):
    """
    Lädt eine CoNLL-U-Datei und gibt die Token- und Tag-Sequenzen zurück.
    """
    sentences = []
    sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Überspringe leere Zeilen oder Kommentare
            if line == '' or line.startswith('#'):
                continue
            
            # Teile die Zeile in Felder und stelle sicher, dass es mindestens 10 Felder gibt
            parts = line.split('\t')
            if len(parts) == 10:
                token = parts[1]    # Wort (Token)
                tag = parts[3]      # POS-Tag
                sentence.append((token, tag))
            else:
                # Falls weniger als 10 Felder, überspringe diese Zeile
                continue
            
            # Wenn der Satz abgeschlossen ist, füge ihn der Liste hinzu
            if line == '':
                if sentence:
                    sentences.append(sentence)
                    sentence = []
    
    # Falls der letzte Satz noch nicht hinzugefügt wurde, füge ihn hinzu
    if sentence:
        sentences.append(sentence)
    
    return sentences


def load_all_data(data_dir):
    """
    Lädt alle CoNLL-U-Dateien im angegebenen Verzeichnis und gibt die Token- und Tag-Sequenzen zurück.
    """
    all_data = {}
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.conllu'):
                file_path = os.path.join(root, file)
                print(f"Lade Datei: {file_path}")
                sentences = load_conllu_file(file_path)
                all_data[file] = sentences
    
    return all_data

def create_vocab(data):
    """
    Erstellt ein Vokabular aus den geladenen Daten und gibt es zurück.
    """
    vocab = set()
    for sentences in data.values():
        for sentence in sentences:
            for token, _ in sentence:
                vocab.add(token.lower())  # Normalisieren zu Kleinbuchstaben
    return vocab

def create_tagset(data):
    """
    Erstellt eine Tagset aus den geladenen Daten und gibt es zurück.
    """
    tagset = set()
    for sentences in data.values():
        for sentence in sentences:
            for _, tag in sentence:
                tagset.add(tag)
    return tagset

if __name__ == "__main__":
    # Beispiel: Verzeichnis mit den CoNLL-U-Daten
    data_dir = '../data/UD_English-EWT'
    file_path = r"C:\Users\Finn Hoffmann\Documents\Semester 11 (Spanien)\Computational Syntax\HMM_pos_tagger_assignment\data\UD_English-EWT\en_ewt-ud-train.conllu"

    # Alle Daten laden
    all_data = load_all_data(data_dir)
    
    # Vokabular und Tagset erstellen
    vocab = create_vocab(all_data)
    tagset = create_tagset(all_data)

    print(load_conllu_file(file_path=file_path)[:1])
    # print(f"Vokabular: {len(vocab)} Wörter")
    # print(f"Tagset: {len(tagset)} Tags")
