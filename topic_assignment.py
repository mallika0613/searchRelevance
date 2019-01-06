import pandas as pd
import random
import csv
import pickle
import gensim

import topic_modelling
location = "products.csv"
products_data = pd.read_csv(location)


def data_prepare():
    """
    Prepared description and title part of data for topic computation

    :return:
    text_data_title: List prepared title data
    text_data_desc: List prepared description
    """
    text_data_title = []
    text_data_desc = []
    with open('products.csv') as f:
        data = csv.reader(f)
        next(data, None)
        for line in data:
            tokens_title = topic_modelling.prepare_text_for_lda(line[1])
            tokens_desc = topic_modelling.prepare_text_for_lda(line[2], isVerb= True)
            text_data_title.append(tokens_title)
            text_data_desc.append(tokens_desc)

    return text_data_title, text_data_desc


def topics_extract(tokens_title, tokens_desc):
    """
    Topics Extracion

    :param tokens_title: List
    :param tokens_desc: List
    :return:
    """
    corpus_title, ldamodel_title, dictionary_title = topic_modelling.topic_model(tokens_title, NUM_TOPICS= 150)

    pickle.dump(corpus_title, open('corpus_title.pkl', 'wb'))
    dictionary_title.save('dictionary_title.gensim')
    ldamodel_title.save('model_title.gensim')

    corpus_desc, ldamodel_desc, dictionary_desc = topic_modelling.topic_model(tokens_desc, NUM_TOPICS=150)

    pickle.dump(corpus_desc, open('corpus_desc.pkl', 'wb'))
    dictionary_desc.save('dictionary_desc.gensim')
    ldamodel_desc.save('model_desc.gensim')


def topic_assignment(text_data_desc, text_data_title):
    """

    :param text_data_desc: List prepared title data
    :param text_data_title: List prepared description
    :return: Dataframe with assigned topics

    """
    dictionary_title = gensim.corpora.Dictionary.load('dictionary_title.gensim')
    dictionary_desc = gensim.corpora.Dictionary.load('dictionary_desc.gensim')

    lda_title = gensim.models.ldamodel.LdaModel.load('model_title.gensim')
    lda_desc = gensim.models.ldamodel.LdaModel.load('model_desc.gensim')

    products_data['topics_desc'] = None
    products_data['topics_title'] = None

    for index, doc in enumerate(text_data_desc):
        try:
            bow_desc = dictionary_desc.doc2bow(doc)
            topic_desc = lda_desc.get_document_topics(bow_desc)
            topic_desc.sort(key=lambda elem: elem[1], reverse=True)
            wp = lda_desc.show_topic(topic_desc[0][0])
            topic_keywords = ", ".join([word for word, prop in wp])
            products_data['topics_desc'][index] = topic_keywords

            bow_title = dictionary_title.doc2bow(text_data_title[index])
            topic_title = lda_title.get_document_topics(bow_title)
            topic_title.sort(key=lambda elem: elem[1], reverse=True)
            print(topic_title)
            wp = lda_title.show_topic(topic_title[0][0])
            topic_keywords = ", ".join([word for word, prop in wp])
            products_data['topics_title'][index] = topic_keywords
        except:
            continue


    return products_data


text_data_title, text_data_desc = data_prepare()
pickle.dump(text_data_desc, open('text_data_desc.pkl', 'wb'))
pickle.dump(text_data_title, open('text_data_title.pkl', 'wb'))
print('Data Prepared')
topics_extract(text_data_title, text_data_desc)
print('Topics extracted')

file = open("text_data_desc.pkl",'rb')
text_data_desc = pickle.load(file)

file = open("text_data_title.pkl",'rb')
text_data_title = pickle.load(file)
products_data= topic_assignment(text_data_desc, text_data_title)
print('Topics Assigned')
products_data.write_csv("products_data.csv")

