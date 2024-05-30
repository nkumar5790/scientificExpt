## Funcation to check docs at diferent level

# Function to check Tense of the give doc
import spacy
from collections import Counter

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

def detect_tense(paragraph):
    # Process the paragraph
    doc = nlp(paragraph)
    
    # Initialize a counter for tenses
    tense_counter = Counter()
    
    # Iterate through tokens and identify verbs and their tenses
    for token in doc:
        if token.pos_ == 'VERB':
            if token.tag_ in ['VBD', 'VBN']:  # Past tense
                tense_counter['Past'] += 1
            elif token.tag_ in ['VB', 'VBP', 'VBZ']:  # Present tense
                tense_counter['Present'] += 1
            elif token.tag_ in ['MD', 'VB', 'VBG']:  # Future tense (simplified)
                # This is a simplified check; modals and 'going to' are future indicators
                if token.tag_ == 'MD' or (token.lemma_ == 'be' and token.dep_ == 'aux'):
                    tense_counter['Future'] += 1
                elif token.tag_ == 'VBG':
                    # Present continuous or future (with context, usually detected)
                    if any(child.dep_ == 'aux' and child.lemma_ == 'will' for child in token.children):
                        tense_counter['Future'] += 1
                    else:
                        tense_counter['Present'] += 1
    
    # Determine the most common tense
    if tense_counter:
        most_common_tense = tense_counter.most_common(1)[0][0]
    else:
        most_common_tense = 'Unknown'
    
    return most_common_tense

# Function to check any repeted n-gram words
from collections import defaultdict
from itertools import islice
def generate_ngrams(text, n):
    words = text.split()
    ngrams = zip(*(islice(words, i, None) for i in range(n)))
    return [' '.join(ngram) for ngram in ngrams]
def detect_repeated_phrases(text, min_ngram=3):
    phrases_count = defaultdict(int)
    ngram_size = min_ngram
    # Generate ngrams of size ngram_size or larger
    while True:
        ngrams = generate_ngrams(text, ngram_size)
        if not ngrams:
            break
        for ngram in ngrams:
            phrases_count[ngram] += 1
        ngram_size += 1
    repeated_phrases = {phrase: count for phrase, count in phrases_count.items() if count > 1}
    return repeated_phrases
