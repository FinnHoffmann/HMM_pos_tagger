from collections import defaultdict
import math
from preprocess import load_all_data, create_vocab, create_tagset

def compute_transition_probs(tag_sequences):
    """
    Berechnet die Übergangswahrscheinlichkeiten: P(tag_t | tag_(t-1)).
    """
    transition_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)

    for sentence in tag_sequences:
        previous_tag = None
        for token, tag, *rest in sentence:  # Entpacke nur (Token, Tag) und ignoriere den Rest
            if previous_tag is not None:
                transition_counts[previous_tag][tag] += 1
            tag_counts[tag] += 1
            previous_tag = tag

    # Berechne Wahrscheinlichkeiten
    transition_probs = defaultdict(dict)
    for prev_tag, next_tags in transition_counts.items():
        total_transitions = sum(next_tags.values())
        for next_tag, count in next_tags.items():
            transition_probs[prev_tag][next_tag] = count / total_transitions
    
    return transition_probs, tag_counts




def compute_emission_probs(data, vocab, tagset):
    """
    Berechnet die Emissionswahrscheinlichkeiten: P(token | tag).
    """
    emission_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)

    for sentences in data.values():
        for sentence in sentences:
            for token, tag in sentence:
                if token.lower() in vocab:  # Nur Vokabular berücksichtigen
                    emission_counts[tag][token.lower()] += 1
                    tag_counts[tag] += 1

    # Berechne Wahrscheinlichkeiten
    emission_probs = defaultdict(dict)
    for tag, token_counts in emission_counts.items():
        total_tag_count = tag_counts[tag]
        for token, count in token_counts.items():
            emission_probs[tag][token] = count / total_tag_count
    
    return emission_probs, tag_counts

def train(data_dir):
    """
    Trainiert das HMM-Modell und berechnet die Übergangs- und Emissionswahrscheinlichkeiten.
    """
    # Lade alle Daten
    all_data = load_all_data(data_dir)
    
    # Erstelle Vokabular und Tagset
    vocab = create_vocab(all_data)
    tagset = create_tagset(all_data)
    
    # Extrahiere alle Tag-Sequenzen aus den Sätzen
    tag_sequences = []
    for sentences in all_data.values():
        for sentence in sentences:
            tag_sequences.append([tag for _, tag in sentence])
    
    # Berechne die Übergangswahrscheinlichkeiten
    transition_probs, tag_counts = compute_transition_probs(tag_sequences)
    
    # Berechne die Emissionswahrscheinlichkeiten
    emission_probs, _ = compute_emission_probs(all_data, vocab, tagset)
    
    return transition_probs, emission_probs, tag_counts, vocab, tagset

if __name__ == "__main__":
    # Beispiel: Verzeichnis mit den CoNLL-U-Daten
    data_dir = '../data/UD_English-EWT'

    # Trainiere das Modell
    transition_probs, emission_probs, tag_counts, vocab, tagset = train(data_dir)

    # Zeige einige Beispiele der berechneten Wahrscheinlichkeiten
    print("Beispiel Transition-Wahrscheinlichkeiten:")
    print(dict(transition_probs.get('NOUN', {}).items())[:5])  # Beispiel: Übergänge von 'NOUN'
    
    print("Beispiel Emissions-Wahrscheinlichkeiten:")
    print(dict(emission_probs.get('VERB', {}).items())[:5])  # Beispiel: Emissionen für 'VERB'
    
    print(f"Vokabular-Größe: {len(vocab)}")
    print(f"Tagset-Größe: {len(tagset)}")
