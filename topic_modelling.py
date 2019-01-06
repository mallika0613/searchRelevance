import spacy
spacy.load('en')
from spacy.lang.en import English
from nltk.corpus import wordnet as wn
from gensim import corpora
import gensim
import pandas as pd
import nltk
import warnings

parser = English()
nltk.download('wordnet')
nltk.download('stopwords')

warnings.filterwarnings("ignore", category=DeprecationWarning)


def tokenize(text):
    """
    Tokenizes the text
    :param text: Sentence that has to be tokenized
    :return: List of tokens
    """
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens



def get_lemma(word):
    """
    Lemmatize word with wordnet
    :param word: string
    :return: string lemmatized word
    """
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def get_commonwords():
    en_stop = set(nltk.corpus.stopwords.words('english'))
    color_set = [line.rstrip('\n').lower() for line in open('colors.txt')]
    verbs = pd.read_csv('verbs.csv')
    return en_stop,color_set, verbs


def prepare_text_for_lda(text, isVerb = False):
    """

    :param text: String Text that has to prepared
    :param isVerb: Boolean - If verbs has to removed from the text
    :return: List of token preapred
    """
    tokens = tokenize(text)
    en_stop, color_set, verbs = get_commonwords()
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [token for token in tokens if token not in color_set]
    tokens = [get_lemma(token) for token in tokens]
    if(isVerb):
        tokens = [token for token in tokens if token not in verbs['Word']]
        tokens = [token for token in tokens if token not in verbs['3singular']]
        tokens = [token for token in tokens if token not in verbs['Present Participle']]
        tokens = [token for token in tokens if token not in verbs['Simple Past']]
        tokens = [token for token in tokens if token not in verbs['Past Participle']]
    return tokens

def topic_model(text_data, NUM_TOPICS):
    """
    Topics compiutation
    :param text_data: List Prepared text data as tokens
    :param NUM_TOPICS: int -  Number of topics to be computed
    :return: corpus, ldamodel, dictionary
    """
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)

    return corpus, ldamodel, dictionary


