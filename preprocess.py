from collections import Counter
from nltk.stem import PorterStemmer

def clean_by_freq(tokenized_words, cut_off_count):
    vocab = Counter(tokenized_words)

    uncommon_words = {key for key, value in vocab.item if value <= cut_off_count}
    cleaned_words = [word for word in tokenized_words if word not in uncommon_words]

    return cleaned_words

def clean_by_len(tokenized_words, cut_off_len):
    cleaned_words = []
    for word in tokenized_words:
        if len(word) > cut_off_len:
            cleaned_words.append(word)

    return cleaned_words

def clean_by_stopwords(tokenized_words, stopwords_set):
    return [x for x in tokenized_words if x not in stopwords_set]

def stemming_by_porter(tokenized_words):
    porter_stemmer = PorterStemmer()
    porter_stemmed_words = []

    for word in tokenized_words:
        stem = porter_stemmer.stem(word)
        porter_stemmed_words.append(stem)
        
    return porter_stemmed_words