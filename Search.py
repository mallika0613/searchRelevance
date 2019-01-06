
import numpy as np
import pandas as pd
import re

def check_subset_regex(searchterm, data):

    if searchterm == '':
        return []

    pattern = '|'.join([' (?:'+x+') ' for x in searchterm.lower().split(' ')])
    return re.findall(pattern, data)


def rankfct(row):
    row['rank'] = row['price'].rank(ascending=False)
    return row


def relevance_sort(search_term):
    products_data= pd.read_csv('products_data.csv')
    word_exp = '|'.join(search_term.split(' '))
    score_card = products_data[products_data['title'].str.contains(search_term, case=False)]
    score_card['search_result'] = []

    if score_card.empty:
        score_card = products_data
        score_card['search_result'] = products_data.apply(
            lambda row: check_subset_regex(search_term, row['title'].lower()), axis=1)
        score_card = score_card.loc[np.array(list(map(len, score_card.search_result.values))) > 0]

    score_card['score'] = 0

    score = 0

    for index, doc in score_card.iterrows():
        res = check_subset_regex(search_term, doc['description'].lower())
        score = score + len(res)

        if len(doc['search_result']) > 1:
            score = score + 1
        if word_exp in doc['description'].lower():
            score = score + 1

        try:
            if word_exp in doc['topics_desc'].lower():
                score = score + 1
        except:
            continue
        try:
            if word_exp in doc['topics_title'].lower():
                score = score + 1
        except:
            continue
        doc['score'] = score

    score_card.sort_values(by=['score'], ascending=False)


    score_order = score_card.groupby(['score']).apply(rankfct).sort_values(['score', 'price'], ascending=[0, 0])
    score_order.sort_values(by=['rank'])

    search_result = score_order[['category', 'title', 'description', 'price']]

    return search_result


searchTerm = input("Enter Search Term")
searchResults = relevance_sort(searchTerm)
print(searchResults)






