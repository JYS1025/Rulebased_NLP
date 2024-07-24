from collections import Counter
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn

lemmatizer = WordNetLemmatizer()

def clean_by_freq(tokenized_words, cut_off_count):
    vocab = Counter(tokenized_words)
    uncommon_words = {key for key, value in vocab.items() if value <= cut_off_count}
    cleaned_words = [word for word in tokenized_words if word not in uncommon_words]
    return cleaned_words

def clean_by_len(tokenized_words, cut_off_len):
    cleaned_words = [word for word in tokenized_words if len(word) > cut_off_len]
    return cleaned_words

def clean_by_stopwords(tokenized_words, stopwords_set):
    return [x for x in tokenized_words if x not in stopwords_set]

def stemming_by_porter(tokenized_words):
    porter_stemmer = PorterStemmer()
    return [porter_stemmer.stem(word) for word in tokenized_words]

def combine(sentence):
    return ' '.join(sentence)

def pos_tagger(tokenized_sent):
    pos_tagged_words = []
    for sentence in tokenized_sent:
        words = word_tokenize(sentence)
        pos_tagged = pos_tag(words)
        pos_tagged_words.extend(pos_tagged)
    return pos_tagged_words

def penn_to_wn(tag):
    if tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return None

def words_lemmatizer(pos_tagged_words):
    lemmatized_words = []
    for word, tag in pos_tagged_words:
        wn_tag = penn_to_wn(tag)
        if wn_tag in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
            lemmatized_words.append(lemmatizer.lemmatize(word, wn_tag))
        else:
            lemmatized_words.append(word)
    return lemmatized_words
