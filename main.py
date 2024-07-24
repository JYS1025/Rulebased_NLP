import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

from preprocess import clean_by_freq
from preprocess import clean_by_len
from preprocess import clean_by_stopwords
from preprocess import stemming_by_porter

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('imdb.tsv', delimiter='\\t')

#Normalize
df['review'] = df['review'].str.lower()

#Tokenize
df['word_tokens'] = df['review'].apply(word_tokenize)

stopwords_set = set(stopwords.words('english'))

df['cleaned_tokens'] = df['word_tokens'].apply(lambda x: clean_by_freq(x, 1))
df['cleaned_tokens'] = df['cleaned_tokens'].apply(lambda x: clean_by_len(x, 2))
df['cleaned_tokens'] = df['cleaned_tokens'].apply(lambda x: clean_by_stopwords(x, stopwords_set))

# 어간 추출
df['stemmed_tokens'] = df['cleaned_tokens'].apply(stemming_by_porter)
