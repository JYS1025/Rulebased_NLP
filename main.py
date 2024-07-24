import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter

from preprocess import clean_by_freq
from preprocess import clean_by_len
from preprocess import clean_by_stopwords
from preprocess import stemming_by_porter
from preprocess import combine
from preprocess import pos_tagger
from preprocess import words_lemmatizer, idx_encoder

# NLTK 리소스 다운로드
nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('averaged_perceptron_tagger',quiet=True)
nltk.download('wordnet',quiet=True)

# 데이터프레임 읽기
df = pd.read_csv('imdb.tsv', delimiter='\\t', engine='python')

# Normalize
df['review'] = df['review'].str.lower()

# Tokenize
df['word_tokens'] = df['review'].apply(word_tokenize)
df['sent_tokens'] = df['review'].apply(sent_tokenize)

# pos-tagging
df['pos_tagged_tokens'] = df['sent_tokens'].apply(pos_tagger)

# 표제어 추출
df['lemmatized_tokens'] = df['pos_tagged_tokens'].apply(words_lemmatizer)

stopwords_set = set(stopwords.words('english'))

# 클린 작업
df['cleaned_tokens'] = df['lemmatized_tokens'].apply(lambda x: clean_by_freq(x, 1))
df['cleaned_tokens'] = df['cleaned_tokens'].apply(lambda x: clean_by_len(x, 2))
df['cleaned_tokens'] = df['cleaned_tokens'].apply(lambda x: clean_by_stopwords(x, stopwords_set))

# 어간 추출
df['stemmed_tokens'] = df['cleaned_tokens'].apply(stemming_by_porter)

# 문자열 통합
df['combined_corpus'] = df['stemmed_tokens'].apply(combine)

# Interger Encoding
tokens = sum(df['cleaned_tokens'], [])
vocab = Counter(tokens)
vocab = vocab.most_common()

#indexing
word2idx = {}
for i, (word, frequency) in enumerate(vocab):
    word2idx[word] = i+1

df['integer_encoded'] = df['cleaned_tokens'].apply(lambda x: idx_encoder(x, word2idx))


