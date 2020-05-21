import numpy as np
import pandas as pd
import csv

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.pipeline import EntityLinker
import en_core_web_lg

from nltk.corpus import stopwords

import pyLDAvis

import tqdm
from tqdm.notebook import tqdm_notebook as tqdm
from pprint import pprint

from display import displayLDA

nytimes = pd.read_csv('include/nytimes.csv', sep=',', nrows=10)
df = nytimes['lead_paragraph']

#print(df.head(5))

nlp = spacy.load('en_core_web_lg')

# My list of stop words.
stop_list = ["Mrs.", "Ms.", "say", "The", "´s", "Mr."]

# Updates spaCy's default stop words list with my additional words.
nlp.Defaults.stop_words.update(stop_list)

# Iterates over the words in the stop words list and resets the "is_stop" flag.
for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True


def lemmatizer(doc):
    # This takes in a doc of tokens from the NER and lemmatizes them.
    # Pronouns (like "I" and "you" get lemmatized to '-PRON-', so I'm removing those.
    doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    doc = u' '.join(doc)
    return nlp.make_doc(doc)


def remove_stopwords(doc):
    # This will remove stopwords and punctuation.
    # Use token.text to return strings, which we'll need for Gensim.
    doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    return doc


# The add_pipe function appends our functions to the default pipeline.
nlp.add_pipe(lemmatizer, name='lemmatizer', after='ner')
nlp.add_pipe(remove_stopwords, name="stopwords", last=True)

doc_list = []
# Iterates through each article in the corpus.
for doc in tqdm(df):
    # Passes that article through the pipeline and adds to a new list.
    pr = nlp(doc)
    doc_list.append(pr)

# Creates, which is a mapping of word IDs to words.
words = corpora.Dictionary(doc_list)

# Turns each document into a bag of words.
corpus = [words.doc2bow(doc) for doc in doc_list]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=words,
                                            num_topics=10,
                                            random_state=2,
                                            update_every=1,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

pprint(lda_model.print_topics(num_topics=10))

#
#displayLDA(lda_model)

from lda2vec import Lda2vec

topics =Lda2vec.utils.prepare_topics(lda_model, corpus)
prepared = pyLDAvis.prepare(topics)
pyLDAvis.display(prepared)



#
#
# with open('test.txt', 'r', newline='\n') as csvfile:
#     #spamwriter = csv.writer(csvfile, delimiter=',',
#        #                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
#      reader = csv.DictReader(csvfile)
#      for row in reader:
#         print(row['box1'], row['box2'])
# # with open('test.txt', 'r') as myfile:
# #     data = myfile.read()
# #
