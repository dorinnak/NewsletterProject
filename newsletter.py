'''In order to run the newsletter creation, several Python 
packages are requiered to be downloaded:

pip install feedparser 
pip install newspaper3k 
pip install DateTime 
pip install pandas 
pip install regex 
pip install nltk 
pip install scikit-learn 
pip install numpy 
pip install matplotlib 
pip install Flask 
pip install urllib5 
pip install Unidecode '''

import warnings
warnings.filterwarnings("ignore")
import feedparser as fp
import newspaper; from newspaper import Article
import time 
from time import mktime
from datetime import datetime
from datetime import date
import dateutil
from datetime import *; from dateutil.relativedelta import *
import calendar
import pandas as pd
from pandas.io.json import json_normalize
import json
import re
import nltk; from nltk.corpus import stopwords; from unidecode import unidecode
import string
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer #for creating count vectors
from sklearn.metrics.pairwise import cosine_similarity #cosine similarity calculator
import matplotlib.pyplot as plt
from matplotlib import style
from flask import render_template
import flask
from urllib.request import urlopen
import webbrowser
import os

nltk.download('wordnet')
nltk.download('punkt')

global today, yesterday
today = str(date.today()) 
yesterday = str(date.today() - timedelta(days=1))
date_html = datetime.strptime(today,'%Y-%m-%d')
datetime = date_html.strftime('%d, %b %Y')

def lemmatize_text(text):
    w_tokenizer, lemmatizer = nltk.tokenize.WhitespaceTokenizer() , nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

def create_newsletter(file): 

    print("Starting scraping...")
    with open(file) as data_file: #Loads the JSON files with news URLs
        companies = json.load(data_file)

    text_list, source_list, article_list, date_list, time_list, title_list, image_list, keywords_list, summaries_list = [], [], [], [], [], [], [], [], []

    for source, content in companies.items():
        source_list.append(source)
        source_list_unique = set(source_list)
        for rss, links in content.items():
            for url in content["rss"]:
                d = fp.parse(url)
                article = {}
                for entry in d.entries:
                    if hasattr(entry,'published') and (((dateutil.parser.parse(getattr(entry,'published'))).strftime("%Y-%m-%d") == today) or ((dateutil.parser.parse(getattr(entry, 'published'))).strftime("%Y-%m-%d") == yesterday)):
                        article['source'] = source
                        source_list.append(article['source'])

                        # getting the article URLs
                        article['link'] = entry.link
                        article_list.append(article['link'])

                        # getting the article published dates
                        date = (getattr(entry, 'published'))
                        date = dateutil.parser.parse(date)
                        date_formated = date.strftime("%Y-%m-%d")
                        time_formated = date.strftime("%H:%M:%S %Z") # hour, minute, timezone (converted)
                        date_list.append(date_formated)
                        time_list.append(time_formated)

                        content = Article(entry.link)
                        try:
                            content.download()
                            content.parse()  
                            content.nlp()
                        except Exception as e: 
                            # in case the download fails, it prints the error and immediatly continues with downloading the next article
                            print(e)
                            print("continuing...")

                        # save the "downloaded" content
                        title = content.title # extract article titles
                        image = content.top_image # extract article images
                        image_list.append(image)
                        keywords = content.keywords
                        keywords_list.append(keywords)
                        title_list.append(title)
                        text = content.text
                        text_list.append(text)
                        summaries = content.summary
                        summaries_list.append(summaries)

    source_dict = {'source':source_list}
    link_dict = {'link':article_list}
    date_dict = {'published_date':date_list}
    time_dict = {'published_time':time_list}
    title_dict = {'title':title_list}
    text_dict = {'text':text_list}
    keyword_dict = {'keywords':keywords_list}
    image_dict = {'image':image_list}
    summary_dict = {'summary':summaries_list}

    source_df = pd.DataFrame(source_dict, index=None)
    link_df = pd.DataFrame(link_dict, index=None)
    date_df = pd.DataFrame(date_dict, index=None)
    time_df = pd.DataFrame(time_dict, index=None)
    title_df = pd.DataFrame(title_dict, index=None)
    text_df = pd.DataFrame(text_dict, index=None)
    keyword_df = pd.DataFrame(keyword_dict, index=None)
    image_df = pd.DataFrame(image_dict, index=None)
    summary_df = pd.DataFrame(summary_dict, index=None)
    
    global news_df
    news_df = source_df.join(link_df).join(date_df).join(time_df).join(title_df).join(text_df).join(keyword_df).join(image_df).join(summary_df)

    for i in source_list_unique:
        print(source_list.count(i) , " articles downloaded from ", i)
    print("\n" , len(source_list) , " total articles downloaded", "\n")
    
    global news_df_daily    
    news_df_daily = news_df[news_df.title != ""]
    news_df_daily = news_df_daily[news_df_daily.text != ""]
    news_df_daily = news_df_daily[news_df_daily.image != ""]

    news_df_daily = news_df_daily[news_df_daily.title.str.count('\s+').ge(3)] #keep only titles having more than 3 spaces = length
    news_df_daily = news_df_daily[news_df_daily.text.str.count('\s+').ge(20)] #keep only titles having more than 20 spaces = length

    # 2 Make all letters lower case
    news_df_daily["clean_title"] = news_df_daily["title"].str.lower()

    # 3 Filter out the stopwords
    stop = stopwords.words('english')

    news_df_daily["clean_title"] = (news_df_daily["clean_title"].str.replace("'s",'').str.replace("’s",''))
    news_df_daily['clean_title'] = news_df_daily['clean_title'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

    # 4 Remove sources, punctuation ('[^\w\s]','') and numbers ('\d+', '')
    sources_list = (list(source_dict.values()))
    for i in sources_list:
        sources_set = set(i)
    sources_set = [x.lower() for x in sources_set]
    sources_to_replace = dict.fromkeys(sources_set, "") # replace every source with "" nothing

    news_df_daily["clean_title"] = (((news_df_daily["clean_title"].str.replace('[^\w\s]',''))
                                    .str.replace('\d+', '')).replace(sources_to_replace, regex=True))

    news_df_daily["clean_title"] = news_df_daily["clean_title"].apply(unidecode)

    news_df_daily["clean_title"] = (news_df_daily["clean_title"].apply(lemmatize_text).apply(lambda x: ' '.join([word for word in x])))

    news_df_daily["keywords"] = news_df_daily["keywords"].apply(lambda x: ' '.join([word for word in x]))

    # 6 Drop Duplicates

    news_df_daily = (news_df_daily.drop_duplicates(("title"))).sort_index()
    news_df_daily = (news_df_daily.drop_duplicates(("link"))).sort_index()

    news_df_daily = news_df_daily.reset_index(drop=True)

    pd.set_option('display.max_colwidth', 20)

    print("Scraping complete.")
    
    article_eucd(news_df_daily)
    
    print("Analysis complete.")
    
    global news_df_daily_cluster_1, news_df_daily_cluster_2, news_df_daily_cluster_3, news_df_daily_cluster_4, news_df_daily_cluster_5, news_df_daily_cluster_6
    
    article_filtering_eucd(eucl_dist_df)
    
    global rand_text_, rand_source, rand_link, rand_title, rand_img
    rand_text_, rand_source, rand_link, rand_title, rand_img = [], [], [], [], [] 
    unique_cosim_vals_1, unique_cosim_vals_2, unique_cosim_vals_3, unique_cosim_vals_4, unique_cosim_vals_5, unique_cosim_vals_6, unique_cosim_vals_7 = [], [], [], [], [], [], []
    unique_cosim_vals_1_filter, unique_cosim_vals_2_filter, unique_cosim_vals_3_filter, unique_cosim_vals_4_filter, unique_cosim_vals_5_filter, unique_cosim_vals_6_filter, unique_cosim_vals_7_filter = [], [], [], [], [], [], []
   
    count_vectorizer = CountVectorizer()

    if list(news_df_daily_cluster_1.shape)[0] > 2:
        clean_titles_list_1 = list(news_df_daily_cluster_1['clean_title']) 
        count_matrix_title_sparse_1 = count_vectorizer.fit_transform(clean_titles_list_1) 
        count_matrix_title_np_1 = count_matrix_title_sparse_1.todense()
        count_matrix_title_df_1 = pd.DataFrame(count_matrix_title_np_1, columns=count_vectorizer.get_feature_names())
        df_cosim_1 = pd.DataFrame(cosine_similarity(count_matrix_title_df_1, count_matrix_title_df_1))
        create_value_list(df_cosim_1, unique_cosim_vals_1)

        
        if len(unique_cosim_vals_1) != 0:
            dict_pos_cosim_val_1 = {elem: get_indexes(df_cosim_1, elem) for elem in ((np.array(unique_cosim_vals_1[0:1])).astype(str))}

            index_list_titles_clust1 = []
            find_indexes(dict_pos_cosim_val_1, index_list_titles_clust1)
            index_list_titles_clust1 = list(set(index_list_titles_clust1)) #creating a set for filtering out duplicate row and col indexes

            cosim_filter_cluster_1 = (news_df_daily_cluster_1.iloc[index_list_titles_clust1,:]) #select the articles based on the indexes
            
            if today in cosim_filter_cluster_1["published_date"].unique():
                cosim_filter_cluster_1 = (cosim_filter_cluster_1.loc[cosim_filter_cluster_1["published_date"] == today]).iloc[[0]]
            elif yesterday in cosim_filter_cluster_1["published_date"].unique():
                cosim_filter_cluster_1 = (cosim_filter_cluster_1.loc[cosim_filter_cluster_1["published_date"] == yesterday]).iloc[[0]]
            
            rand_info(cosim_filter_cluster_1, 1, 250) 
            
            # check whether there are other articles in cluster
            news_df_daily_cluster_1 = news_df_daily_cluster_1.reset_index(drop=True)
            news_df_daily_cluster_1_used = news_df_daily_cluster_1.index.isin(index_list_titles_clust1)
            news_df_daily_cluster_1_filter = news_df_daily_cluster_1[~news_df_daily_cluster_1_used]
            
            if list(news_df_daily_cluster_1_filter.shape)[0] > 1:            
                clean_titles_list_1_filter = list(news_df_daily_cluster_1_filter['clean_title']) 
                count_matrix_title_sparse_1_filter = count_vectorizer.fit_transform(clean_titles_list_1_filter) 
                count_matrix_title_np_1_filter = count_matrix_title_sparse_1_filter.todense()
                count_matrix_title_df_1_filter = pd.DataFrame(count_matrix_title_np_1_filter, columns=count_vectorizer.get_feature_names())
                df_cosim_1_filter = pd.DataFrame(cosine_similarity(count_matrix_title_df_1_filter, count_matrix_title_df_1_filter))
                create_value_list(df_cosim_1_filter, unique_cosim_vals_1_filter)
                
                dict_pos_cosim_val_1_filter = {elem: get_indexes(df_cosim_1_filter, elem) for elem in ((np.array(unique_cosim_vals_1_filter[0:1])).astype(str))}

                index_list_titles_clust1_filter = []
                find_indexes(dict_pos_cosim_val_1_filter, index_list_titles_clust1_filter) # apply the function for finding the indexes we got in the dataframe
                index_list_titles_clust1_filter = list(set(index_list_titles_clust1_filter))

                cosim_filter_cluster_1_filter = (news_df_daily_cluster_1_filter.iloc[index_list_titles_clust1_filter,:]) #select the articles based on the indexes
                if today in cosim_filter_cluster_1_filter["published_date"].unique():
                    cosim_filter_cluster_1_filter = (cosim_filter_cluster_1_filter.loc[cosim_filter_cluster_1_filter["published_date"] == today]).iloc[[0]]
                elif yesterday in cosim_filter_cluster_1_filter["published_date"].unique():
                    cosim_filter_cluster_1_filter = (cosim_filter_cluster_1_filter.loc[cosim_filter_cluster_1_filter["published_date"] == yesterday]).iloc[[0]]
                    
                rand_info(cosim_filter_cluster_1_filter, 1, 250) 
                
        else:
            rand_info(news_df_daily_cluster_1, 1, 250)
            
    else:
        rand_info(news_df_daily_cluster_1, 1, 250)

    if list(news_df_daily_cluster_2.shape)[0] > 2:
        clean_titles_list_2 = list(news_df_daily_cluster_2['clean_title']) 
        count_matrix_title_sparse_2 = count_vectorizer.fit_transform(clean_titles_list_2) 
        count_matrix_title_np_2 = count_matrix_title_sparse_2.todense()
        count_matrix_title_df_2 = pd.DataFrame(count_matrix_title_np_2, columns=count_vectorizer.get_feature_names())
        df_cosim_2 = pd.DataFrame(cosine_similarity(count_matrix_title_df_2, count_matrix_title_df_2))
        create_value_list(df_cosim_2, unique_cosim_vals_2)

        if len(unique_cosim_vals_2) != 0:
            dict_pos_cosim_val_2 = {elem: get_indexes(df_cosim_2, elem) for elem in ((np.array(unique_cosim_vals_2[0:1])).astype(str))}

            index_list_titles_clust2 = []
            find_indexes(dict_pos_cosim_val_2, index_list_titles_clust2) # apply the function for finding the indexes we got in the dataframe
            index_list_titles_clust2 = list(set(index_list_titles_clust2)) #creating a set for filtering out duplicate row and col indexes

            cosim_filter_cluster_2 = (news_df_daily_cluster_2.iloc[index_list_titles_clust2,:]) #select the articles based on the indexes
            
            if today in cosim_filter_cluster_2["published_date"].unique():
                cosim_filter_cluster_2 = (cosim_filter_cluster_2.loc[cosim_filter_cluster_2["published_date"] == today]).iloc[[0]]
            elif yesterday in cosim_filter_cluster_2["published_date"].unique():
                cosim_filter_cluster_2 = (cosim_filter_cluster_2.loc[cosim_filter_cluster_2["published_date"] == yesterday]).iloc[[0]]

            rand_info(cosim_filter_cluster_2, 1, 250)

            # check whether there are other articles in cluster
            news_df_daily_cluster_2 = news_df_daily_cluster_2.reset_index(drop=True)
            news_df_daily_cluster_2_used = news_df_daily_cluster_2.index.isin(index_list_titles_clust2)
            news_df_daily_cluster_2_filter = news_df_daily_cluster_2[~news_df_daily_cluster_2_used]
            
            if list(news_df_daily_cluster_2_filter.shape)[0] > 1:            
                clean_titles_list_2_filter = list(news_df_daily_cluster_2_filter['clean_title']) 
                count_matrix_title_sparse_2_filter = count_vectorizer.fit_transform(clean_titles_list_2_filter) 
                count_matrix_title_np_2_filter = count_matrix_title_sparse_2_filter.todense()
                count_matrix_title_df_2_filter = pd.DataFrame(count_matrix_title_np_2_filter, columns=count_vectorizer.get_feature_names())
                df_cosim_2_filter = pd.DataFrame(cosine_similarity(count_matrix_title_df_2_filter, count_matrix_title_df_2_filter))
                create_value_list(df_cosim_2_filter, unique_cosim_vals_2_filter)
                
                dict_pos_cosim_val_2_filter = {elem: get_indexes(df_cosim_2_filter, elem) for elem in ((np.array(unique_cosim_vals_2_filter[0:1])).astype(str))}

                index_list_titles_clust2_filter = []
                find_indexes(dict_pos_cosim_val_2_filter, index_list_titles_clust2_filter) # apply the function for finding the indexes we got in the dataframe
                index_list_titles_clust2_filter = list(set(index_list_titles_clust2_filter))

                cosim_filter_cluster_2_filter = (news_df_daily_cluster_2_filter.iloc[index_list_titles_clust2_filter,:]) #select the articles based on the indexes
                if today in cosim_filter_cluster_2_filter["published_date"].unique():
                    cosim_filter_cluster_2_filter = (cosim_filter_cluster_2_filter.loc[cosim_filter_cluster_2_filter["published_date"] == today]).iloc[[0]]
                elif yesterday in cosim_filter_cluster_2_filter["published_date"].unique():
                    cosim_filter_cluster_2_filter = (cosim_filter_cluster_2_filter.loc[cosim_filter_cluster_2_filter["published_date"] == yesterday]).iloc[[0]]
                    
                rand_info(cosim_filter_cluster_2_filter, 1, 250) 
                
        else:
            rand_info(news_df_daily_cluster_2, 1, 250)
            
    else:
        rand_info(news_df_daily_cluster_2, 1, 250)

    if list(news_df_daily_cluster_3.shape)[0] > 2:
        clean_titles_list_3 = list(news_df_daily_cluster_3['clean_title']) 
        count_matrix_title_sparse_3 = count_vectorizer.fit_transform(clean_titles_list_3) 
        count_matrix_title_np_3 = count_matrix_title_sparse_3.todense()
        count_matrix_title_df_3 = pd.DataFrame(count_matrix_title_np_3, columns=count_vectorizer.get_feature_names())
        df_cosim_3 = pd.DataFrame(cosine_similarity(count_matrix_title_df_3, count_matrix_title_df_3))
        create_value_list(df_cosim_3, unique_cosim_vals_3)
        
        if len(unique_cosim_vals_3) != 0:
            dict_pos_cosim_val_3 = {elem: get_indexes(df_cosim_3, elem) for elem in ((np.array(unique_cosim_vals_3[0:1])).astype(str))}

            index_list_titles_clust3 = []
            find_indexes(dict_pos_cosim_val_3, index_list_titles_clust3) # apply the function for finding the indexes we got in the dataframe
            index_list_titles_clust3 = list(set(index_list_titles_clust3)) #creating a set for filtering out duplicate row and col indexes

            cosim_filter_cluster_3 = (news_df_daily_cluster_3.iloc[index_list_titles_clust3,:]) #select the articles based on the indexes

            if today in cosim_filter_cluster_3["published_date"].unique():
                cosim_filter_cluster_3 = (cosim_filter_cluster_3.loc[cosim_filter_cluster_3["published_date"] == today]).iloc[[0]]
            elif yesterday in cosim_filter_cluster_3["published_date"].unique():
                cosim_filter_cluster_3 = (cosim_filter_cluster_3.loc[cosim_filter_cluster_3["published_date"] == yesterday]).iloc[[0]]

            rand_info(cosim_filter_cluster_3, 1, 250)

            # check whether there are other articles in cluster
            news_df_daily_cluster_3 = news_df_daily_cluster_3.reset_index(drop=True)
            news_df_daily_cluster_3_used = news_df_daily_cluster_3.index.isin(index_list_titles_clust3)
            news_df_daily_cluster_3_filter = news_df_daily_cluster_3[~news_df_daily_cluster_3_used]
            
            if list(news_df_daily_cluster_3_filter.shape)[0] > 1:            
                clean_titles_list_3_filter = list(news_df_daily_cluster_3_filter['clean_title']) 
                count_matrix_title_sparse_3_filter = count_vectorizer.fit_transform(clean_titles_list_3_filter) 
                count_matrix_title_np_3_filter = count_matrix_title_sparse_3_filter.todense()
                count_matrix_title_df_3_filter = pd.DataFrame(count_matrix_title_np_3_filter, columns=count_vectorizer.get_feature_names())
                df_cosim_3_filter = pd.DataFrame(cosine_similarity(count_matrix_title_df_3_filter, count_matrix_title_df_3_filter))
                create_value_list(df_cosim_3_filter, unique_cosim_vals_3_filter)
                
                dict_pos_cosim_val_3_filter = {elem: get_indexes(df_cosim_3_filter, elem) for elem in ((np.array(unique_cosim_vals_3_filter[0:1])).astype(str))}

                index_list_titles_clust3_filter = []
                find_indexes(dict_pos_cosim_val_3_filter, index_list_titles_clust3_filter) # apply the function for finding the indexes we got in the dataframe
                index_list_titles_clust3_filter = list(set(index_list_titles_clust3_filter))

                cosim_filter_cluster_3_filter = (news_df_daily_cluster_3_filter.iloc[index_list_titles_clust3_filter,:]) #select the articles based on the indexes
                if today in cosim_filter_cluster_3_filter["published_date"].unique():
                    cosim_filter_cluster_3_filter = (cosim_filter_cluster_3_filter.loc[cosim_filter_cluster_3_filter["published_date"] == today]).iloc[[0]]
                elif yesterday in cosim_filter_cluster_3_filter["published_date"].unique():
                    cosim_filter_cluster_3_filter = (cosim_filter_cluster_3_filter.loc[cosim_filter_cluster_3_filter["published_date"] == yesterday]).iloc[[0]]
                    
                rand_info(cosim_filter_cluster_3_filter, 1, 250) 
                
        else:
            rand_info(news_df_daily_cluster_3, 1, 250)
            
    else:
        rand_info(news_df_daily_cluster_3, 1, 250)


    if list(news_df_daily_cluster_4.shape)[0] > 2:
        clean_titles_list_4 = list(news_df_daily_cluster_4['clean_title']) 
        count_matrix_title_sparse_4 = count_vectorizer.fit_transform(clean_titles_list_4) 
        count_matrix_title_np_4 = count_matrix_title_sparse_4.todense()
        count_matrix_title_df_4 = pd.DataFrame(count_matrix_title_np_4, columns=count_vectorizer.get_feature_names())
        df_cosim_4 = pd.DataFrame(cosine_similarity(count_matrix_title_df_4, count_matrix_title_df_4))
        create_value_list(df_cosim_4, unique_cosim_vals_4)
        if len(unique_cosim_vals_4) != 0:
            dict_pos_cosim_val_4 = {elem: get_indexes(df_cosim_4, elem) for elem in ((np.array(unique_cosim_vals_4[0:1])).astype(str))}

            index_list_titles_clust4 = []
            find_indexes(dict_pos_cosim_val_4, index_list_titles_clust4) # apply the function for finding the indexes we got in the dataframe
            index_list_titles_clust4 = list(set(index_list_titles_clust4)) #creating a set for filtering out duplicate row and col indexes

            cosim_filter_cluster_4 = (news_df_daily_cluster_4.iloc[index_list_titles_clust4,:]) #select the articles based on the indexes

            if today in cosim_filter_cluster_4["published_date"].unique():
                cosim_filter_cluster_4 = (cosim_filter_cluster_4.loc[cosim_filter_cluster_4["published_date"] == today]).iloc[[0]]
            elif yesterday in cosim_filter_cluster_4["published_date"].unique():
                cosim_filter_cluster_4 = (cosim_filter_cluster_4.loc[cosim_filter_cluster_4["published_date"] == yesterday]).iloc[[0]]

            rand_info(cosim_filter_cluster_4, 1, 250)

            # check whether there are other articles in cluster
            news_df_daily_cluster_4 = news_df_daily_cluster_4.reset_index(drop=True)
            news_df_daily_cluster_4_used = news_df_daily_cluster_4.index.isin(index_list_titles_clust4)
            news_df_daily_cluster_4_filter = news_df_daily_cluster_4[~news_df_daily_cluster_4_used]
            
            if list(news_df_daily_cluster_4_filter.shape)[0] > 1:            
                clean_titles_list_4_filter = list(news_df_daily_cluster_4_filter['clean_title']) 
                count_matrix_title_sparse_4_filter = count_vectorizer.fit_transform(clean_titles_list_4_filter) 
                count_matrix_title_np_4_filter = count_matrix_title_sparse_4_filter.todense()
                count_matrix_title_df_4_filter = pd.DataFrame(count_matrix_title_np_4_filter, columns=count_vectorizer.get_feature_names())
                df_cosim_4_filter = pd.DataFrame(cosine_similarity(count_matrix_title_df_4_filter, count_matrix_title_df_4_filter))
                create_value_list(df_cosim_4_filter, unique_cosim_vals_4_filter)
                
                dict_pos_cosim_val_4_filter = {elem: get_indexes(df_cosim_4_filter, elem) for elem in ((np.array(unique_cosim_vals_4_filter[0:1])).astype(str))}

                index_list_titles_clust4_filter = []
                find_indexes(dict_pos_cosim_val_4_filter, index_list_titles_clust4_filter) # apply the function for finding the indexes we got in the dataframe
                index_list_titles_clust4_filter = list(set(index_list_titles_clust4_filter))

                cosim_filter_cluster_4_filter = (news_df_daily_cluster_4_filter.iloc[index_list_titles_clust4_filter,:]) #select the articles based on the indexes
                if today in cosim_filter_cluster_4_filter["published_date"].unique():
                    cosim_filter_cluster_4_filter = (cosim_filter_cluster_4_filter.loc[cosim_filter_cluster_4_filter["published_date"] == today]).iloc[[0]]
                elif yesterday in cosim_filter_cluster_4_filter["published_date"].unique():
                    cosim_filter_cluster_4_filter = (cosim_filter_cluster_4_filter.loc[cosim_filter_cluster_4_filter["published_date"] == yesterday]).iloc[[0]]
                    
                rand_info(cosim_filter_cluster_4_filter, 1, 250) 
        else:
            rand_info(news_df_daily_cluster_4, 1, 250)

    else:
        rand_info(news_df_daily_cluster_4, 1, 250)


    if list(news_df_daily_cluster_5.shape)[0] > 2:
        clean_titles_list_5 = list(news_df_daily_cluster_5['clean_title']) 
        count_matrix_title_sparse_5 = count_vectorizer.fit_transform(clean_titles_list_5) 
        count_matrix_title_np_5 = count_matrix_title_sparse_5.todense()
        count_matrix_title_df_5 = pd.DataFrame(count_matrix_title_np_5, columns=count_vectorizer.get_feature_names())
        df_cosim_5 = pd.DataFrame(cosine_similarity(count_matrix_title_df_5, count_matrix_title_df_5))
        create_value_list(df_cosim_5, unique_cosim_vals_5)

        if len(unique_cosim_vals_5) != 0:
            dict_pos_cosim_val_5 = {elem: get_indexes(df_cosim_5, elem) for elem in ((np.array(unique_cosim_vals_5[0:1])).astype(str))}

            index_list_titles_clust5 = []
            find_indexes(dict_pos_cosim_val_5, index_list_titles_clust5) # apply the function for finding the indexes we got in the dataframe
            index_list_titles_clust5 = list(set(index_list_titles_clust5)) #creating a set for filtering out duplicate row and col indexes

            cosim_filter_cluster_5 = (news_df_daily_cluster_5.iloc[index_list_titles_clust5,:]) #select the articles based on the indexes

            if today in cosim_filter_cluster_5["published_date"].unique():
                cosim_filter_cluster_5 = (cosim_filter_cluster_5.loc[cosim_filter_cluster_5["published_date"] == today]).iloc[[0]]
            elif yesterday in cosim_filter_cluster_5["published_date"].unique():
                cosim_filter_cluster_5 = (cosim_filter_cluster_5.loc[cosim_filter_cluster_5["published_date"] == yesterday]).iloc[[0]]

            rand_info(cosim_filter_cluster_5, 1, 250)

            # check whether there are other articles in cluster
            news_df_daily_cluster_5 = news_df_daily_cluster_5.reset_index(drop=True)
            news_df_daily_cluster_5_used = news_df_daily_cluster_5.index.isin(index_list_titles_clust5)
            news_df_daily_cluster_5_filter = news_df_daily_cluster_5[~news_df_daily_cluster_5_used]
            
            if list(news_df_daily_cluster_5_filter.shape)[0] > 1:            
                clean_titles_list_5_filter = list(news_df_daily_cluster_5_filter['clean_title']) 
                count_matrix_title_sparse_5_filter = count_vectorizer.fit_transform(clean_titles_list_5_filter) 
                count_matrix_title_np_5_filter = count_matrix_title_sparse_5_filter.todense()
                count_matrix_title_df_5_filter = pd.DataFrame(count_matrix_title_np_5_filter, columns=count_vectorizer.get_feature_names())
                df_cosim_5_filter = pd.DataFrame(cosine_similarity(count_matrix_title_df_5_filter, count_matrix_title_df_5_filter))
                create_value_list(df_cosim_5_filter, unique_cosim_vals_5_filter)
                
                dict_pos_cosim_val_5_filter = {elem: get_indexes(df_cosim_5_filter, elem) for elem in ((np.array(unique_cosim_vals_5_filter[0:1])).astype(str))}

                index_list_titles_clust5_filter = []
                find_indexes(dict_pos_cosim_val_5_filter, index_list_titles_clust5_filter) # apply the function for finding the indexes we got in the dataframe
                index_list_titles_clust5_filter = list(set(index_list_titles_clust5_filter))

                cosim_filter_cluster_5_filter = (news_df_daily_cluster_5_filter.iloc[index_list_titles_clust5_filter,:]) #select the articles based on the indexes
                if today in cosim_filter_cluster_5_filter["published_date"].unique():
                    cosim_filter_cluster_5_filter = (cosim_filter_cluster_5_filter.loc[cosim_filter_cluster_5_filter["published_date"] == today]).iloc[[0]]
                elif yesterday in cosim_filter_cluster_5_filter["published_date"].unique():
                    cosim_filter_cluster_5_filter = (cosim_filter_cluster_5_filter.loc[cosim_filter_cluster_5_filter["published_date"] == yesterday]).iloc[[0]]
                    
                rand_info(cosim_filter_cluster_5_filter, 1, 250) 

        else:
            rand_info(news_df_daily_cluster_5, 1, 250)

    else:
        rand_info(news_df_daily_cluster_5, 1, 250)
     
    if list(news_df_daily_cluster_6.shape)[0] > 2:
        clean_titles_list_6 = list(news_df_daily_cluster_6['clean_title']) 
        count_matrix_title_sparse_6 = count_vectorizer.fit_transform(clean_titles_list_6) 
        count_matrix_title_np_6 = count_matrix_title_sparse_6.todense()
        count_matrix_title_df_6 = pd.DataFrame(count_matrix_title_np_6, columns=count_vectorizer.get_feature_names())
        df_cosim_6 = pd.DataFrame(cosine_similarity(count_matrix_title_df_6, count_matrix_title_df_6))
        create_value_list(df_cosim_6, unique_cosim_vals_6)

        if len(unique_cosim_vals_6) != 0:
            dict_pos_cosim_val_6 = {elem: get_indexes(df_cosim_6, elem) for elem in ((np.array(unique_cosim_vals_6[0:1])).astype(str))}

            index_list_titles_clust6 = []
            find_indexes(dict_pos_cosim_val_6, index_list_titles_clust6) # apply the function for finding the indexes we got in the dataframe
            index_list_titles_clust6 = list(set(index_list_titles_clust6)) #creating a set for filtering out duplicate row and col indexes

            cosim_filter_cluster_6 = (news_df_daily_cluster_6.iloc[index_list_titles_clust6,:]) #select the articles based on the indexes

            if today in cosim_filter_cluster_6["published_date"].unique():
                cosim_filter_cluster_6 = (cosim_filter_cluster_6.loc[cosim_filter_cluster_6["published_date"] == today]).iloc[[0]]
            elif yesterday in cosim_filter_cluster_6["published_date"].unique():
                cosim_filter_cluster_6 = (cosim_filter_cluster_6.loc[cosim_filter_cluster_6["published_date"] == yesterday]).iloc[[0]]

            rand_info(cosim_filter_cluster_6, 1, 250)

            # check whether there are other articles in cluster
            news_df_daily_cluster_6 = news_df_daily_cluster_6.reset_index(drop=True)
            news_df_daily_cluster_6_used = news_df_daily_cluster_6.index.isin(index_list_titles_clust6)
            news_df_daily_cluster_6_filter = news_df_daily_cluster_6[~news_df_daily_cluster_6_used]
            
            if list(news_df_daily_cluster_6_filter.shape)[0] > 1:            
                clean_titles_list_6_filter = list(news_df_daily_cluster_6_filter['clean_title']) 
                count_matrix_title_sparse_6_filter = count_vectorizer.fit_transform(clean_titles_list_6_filter) 
                count_matrix_title_np_6_filter = count_matrix_title_sparse_6_filter.todense()
                count_matrix_title_df_6_filter = pd.DataFrame(count_matrix_title_np_6_filter, columns=count_vectorizer.get_feature_names())
                df_cosim_6_filter = pd.DataFrame(cosine_similarity(count_matrix_title_df_6_filter, count_matrix_title_df_6_filter))
                create_value_list(df_cosim_6_filter, unique_cosim_vals_6_filter)
                
                dict_pos_cosim_val_6_filter = {elem: get_indexes(df_cosim_6_filter, elem) for elem in ((np.array(unique_cosim_vals_6_filter[0:1])).astype(str))}

                index_list_titles_clust6_filter = []
                find_indexes(dict_pos_cosim_val_6_filter, index_list_titles_clust6_filter) # apply the function for finding the indexes we got in the dataframe
                index_list_titles_clust6_filter = list(set(index_list_titles_clust6_filter))

                cosim_filter_cluster_6_filter = (news_df_daily_cluster_6_filter.iloc[index_list_titles_clust6_filter,:]) #select the articles based on the indexes
                if today in cosim_filter_cluster_6_filter["published_date"].unique():
                    cosim_filter_cluster_6_filter = (cosim_filter_cluster_6_filter.loc[cosim_filter_cluster_6_filter["published_date"] == today]).iloc[[0]]
                elif yesterday in cosim_filter_cluster_6_filter["published_date"].unique():
                    cosim_filter_cluster_6_filter = (cosim_filter_cluster_6_filter.loc[cosim_filter_cluster_6_filter["published_date"] == yesterday]).iloc[[0]]
                    
                rand_info(cosim_filter_cluster_6_filter, 1, 250) 
        else:
            rand_info(news_df_daily_cluster_6, 1, 250)
    else:
        rand_info(news_df_daily_cluster_6, 1, 250)        

    global rand_text
    rand_text = [item + '...' for item in rand_text_]
    
    print("Article selection complete")

    print("Starting stock extraction...")
    extract_stocks()
    print("Stock extraction complete.")

    print("Creating HTML file...")
    create_HTML_file()
    print("HTML file creation complete.")

    print("HTML file saved in " + os.getcwd()+ " as " + str("NewsLetter_" + today + ".html"))

def article_eucd(df):
    print("Starting analysis...")

    # for analysis, we need a list of all the titles
    global clean_titles_list
    clean_titles_list = list(news_df_daily['clean_title'])

    count_vectorizer = CountVectorizer()
    count_matrix_title_sparse = count_vectorizer.fit_transform(clean_titles_list) # creates the count vector in sparse matrix
    count_matrix_title_np = count_matrix_title_sparse.todense() # creates numpy matrix out from all count vectors
    
    global eucl_dist_df
    eucl_dist_df = pd.DataFrame(euclidean_distances(count_matrix_title_np))

# general function to find the row and column index in a dataframe for a specific value
def get_indexes(dataframe, value):
    pos_list = list()
    for i in value:
        result = dataframe.isin([value]) # crete bool dataframe with True at positions where the given value exists
        series = result.any()
        column_names = list(series[series == True].index) # create list of columns that contain the value
        for col in column_names: # iterate over list of columns and fetch the rows indexes where value exists
            rows = list(result[col][result[col] == True].index)
            for row in rows:
                if row != col: # since matrix diagonal is always == 1, we exclude these results here
                    pos_list.append((row, col)) #creates a list of row, col position
        return pos_list # Return a list of tuples indicating the positions of value in the dataframe

# function for creating a list of the row indexes
def find_indexes(dict_pos, index_list):
    for key, value in dict_pos.items():
    #print(key, ' : ', value) # this prints the similarity values and its corresponding row and col indexes in the df
        for num in value:
            for firstnum in num:
                index_list.append(firstnum)
                
def article_filtering_eucd(eucl_dist_df):
    print("Starting article selection...")
    
    # defining filter criteria by all unique distance values appearing in the euclidean matrix
    cols = list(eucl_dist_df.columns)    
    unique_eucd_vals = sorted(list(pd.unique(eucl_dist_df[cols].values.ravel())))
    unique_eucd_vals.remove(float(0))
    
    # filter by euclidean distance values
    simval = (np.array(unique_eucd_vals[0:1])).astype(str) # first value in the unique list
    dict_pos_eucd_val = {elem: get_indexes(eucl_dist_df, elem) for elem in simval} 
    
    ### Cluster 1
    index_list_titles_clust1 = []
    find_indexes(dict_pos_eucd_val, index_list_titles_clust1) # apply the function for finding the indexes we got in the dataframe
    index_list_titles_clust1 = list(set(index_list_titles_clust1)) #creating a set for filtering out duplicate row and col indexes
    
    global news_df_daily_cluster_1
    news_df_daily_cluster_1 = (news_df_daily.iloc[index_list_titles_clust1, :]) #select the articles based on the indexes
    
    ### Cluster 2
    # Preparation
    news_df_daily_cluster_1_filtered = news_df_daily.index.isin(index_list_titles_clust1) # True/False of whichs rows are in the cluster 1
    news_df_daily_excl_clust_1 = news_df_daily[~news_df_daily_cluster_1_filtered] # Creating DF excluding the already selected articles
    news_df_daily_excl_clust_1 = news_df_daily_excl_clust_1.reset_index(drop=True)

    eucd_cluster_2 = (eucl_dist_df.drop(index=index_list_titles_clust1, columns=index_list_titles_clust1)).reset_index(drop=True) 
    col_rename = range(len(eucd_cluster_2.columns))
    eucd_cluster_2.columns = col_rename

    cols = list(eucd_cluster_2.columns)    
    unique_eucd_vals_2 = sorted(list(pd.unique(eucd_cluster_2[cols].values.ravel())))
    unique_eucd_vals_2.remove(float(0))

    simval_clust2 = (np.array(unique_eucd_vals_2[0:1])).astype(str)
    dict_pos_eucd_val2 = {elem: get_indexes(eucd_cluster_2, elem) for elem in simval_clust2}

    index_list_titles_clust2 = []
    find_indexes(dict_pos_eucd_val2, index_list_titles_clust2) # apply the function for finding the indexes we got in the dataframe
    index_list_titles_clust2 = list(set(index_list_titles_clust2)) #creating a set for filtering out duplicate row and col indexes

    global news_df_daily_cluster_2
    news_df_daily_cluster_2 = (news_df_daily_excl_clust_1.iloc[index_list_titles_clust2, :]) #select the articles based on the indexes
    
    ### Cluster 3
    # Preparation
    news_df_daily_cluster_2_filtered = news_df_daily_excl_clust_1.index.isin(index_list_titles_clust2) # True/False of whichs rows are in the cluster 1
    news_df_daily_excl_clust_1_2 = news_df_daily_excl_clust_1[~news_df_daily_cluster_2_filtered] # Creating DF excluding the already selected articles
    news_df_daily_excl_clust_1_2 = news_df_daily_excl_clust_1_2.reset_index(drop=True)

    eucd_cluster_3 = (eucd_cluster_2.drop(index=index_list_titles_clust2, columns=index_list_titles_clust2)).reset_index(drop=True) 
    col_rename = range(len(eucd_cluster_3.columns))
    eucd_cluster_3.columns = col_rename

    cols = list(eucd_cluster_3.columns)    
    unique_eucd_vals_3 = sorted(list(pd.unique(eucd_cluster_3[cols].values.ravel())))
    unique_eucd_vals_3.remove(float(0))
    
    simval_clust3 = (np.array(unique_eucd_vals_3[0:1])).astype(str)
    dict_pos_eucd_val3 = {elem: get_indexes(eucd_cluster_3, elem) for elem in simval_clust3}

    index_list_titles_clust3 = []
    find_indexes(dict_pos_eucd_val3, index_list_titles_clust3) # apply the function for finding the indexes we got in the dataframe
    index_list_titles_clust3 = list(set(index_list_titles_clust3)) #creating a set for filtering out duplicate row and col indexes
    
    global news_df_daily_cluster_3
    news_df_daily_cluster_3 = (news_df_daily_excl_clust_1_2.iloc[index_list_titles_clust3, :]) #select the articles based on the indexes 
    
    ### Cluster 4
    # Preparation Cluster 4
    news_df_daily_cluster_3_filtered = news_df_daily_excl_clust_1_2.index.isin(index_list_titles_clust3) # True/False of whichs rows are in the cluster 1
    news_df_daily_excl_clust_1_2_3 = news_df_daily_excl_clust_1_2[~news_df_daily_cluster_3_filtered] # Creating DF excluding the already selected articles
    news_df_daily_excl_clust_1_2_3 = news_df_daily_excl_clust_1_2_3.reset_index(drop=True)

    eucd_cluster_4 = (eucd_cluster_3.drop(index=index_list_titles_clust3, columns=index_list_titles_clust3)).reset_index(drop=True) 
    col_rename = range(len(eucd_cluster_4.columns))
    eucd_cluster_4.columns = col_rename

    cols = list(eucd_cluster_4.columns)    
    unique_eucd_vals_4 = sorted(list(pd.unique(eucd_cluster_4[cols].values.ravel())))
    unique_eucd_vals_4.remove(float(0))

    simval_clust4 = (np.array(unique_eucd_vals_4[0:1])).astype(str)
    dict_pos_eucd_val4 = {elem: get_indexes(eucd_cluster_4, elem) for elem in simval_clust4}

    index_list_titles_clust4 = []
    find_indexes(dict_pos_eucd_val4, index_list_titles_clust4) # apply the function for finding the indexes we got in the dataframe
    index_list_titles_clust4 = list(set(index_list_titles_clust4)) #creating a set for filtering out duplicate row and col indexes

    global news_df_daily_cluster_4
    news_df_daily_cluster_4 = (news_df_daily_excl_clust_1_2_3.iloc[index_list_titles_clust4, :])
    
    ### Cluster 5
    # Preparation Cluster 5
    news_df_daily_cluster_4_filtered = news_df_daily_excl_clust_1_2_3.index.isin(index_list_titles_clust4) # True/False of whichs rows are in the cluster 1
    news_df_daily_excl_clust_1_2_3_4 = news_df_daily_excl_clust_1_2_3[~news_df_daily_cluster_4_filtered] # Creating DF excluding the already selected articles
    news_df_daily_excl_clust_1_2_3_4 = news_df_daily_excl_clust_1_2_3_4.reset_index(drop=True)

    eucd_cluster_5 = (eucd_cluster_4.drop(index=index_list_titles_clust4, columns=index_list_titles_clust4)).reset_index(drop=True) 
    col_rename = range(len(eucd_cluster_5.columns))
    eucd_cluster_5.columns = col_rename

    cols = list(eucd_cluster_5.columns)    

    unique_eucd_vals_5 = sorted(list(pd.unique(eucd_cluster_5[cols].values.ravel())))
    unique_eucd_vals_5.remove(float(0))

    simval_clust5 = (np.array(unique_eucd_vals_5[0:1])).astype(str)
    dict_pos_eucd_val5 = {elem: get_indexes(eucd_cluster_5, elem) for elem in simval_clust5} # apply the function to get the indexes for the

    index_list_titles_clust5 = []
    find_indexes(dict_pos_eucd_val5, index_list_titles_clust5) # apply the function for finding the indexes we got in the dataframe
    index_list_titles_clust5 = list(set(index_list_titles_clust5)) #creating a set for filtering out duplicate row and col indexes

    global news_df_daily_cluster_5
    news_df_daily_cluster_5 = (news_df_daily_excl_clust_1_2_3_4.iloc[index_list_titles_clust5, :])
    
    ### Cluster 6
    # Preparation Cluster 6
    news_df_daily_cluster_5_filtered = news_df_daily_excl_clust_1_2_3_4.index.isin(index_list_titles_clust5) # True/False of whichs rows are in the cluster 1
    news_df_daily_excl_clust_1_2_3_4_5 = news_df_daily_excl_clust_1_2_3_4[~news_df_daily_cluster_5_filtered] # Creating DF excluding the already selected articles
    news_df_daily_excl_clust_1_2_3_4_5 = news_df_daily_excl_clust_1_2_3_4_5.reset_index(drop=True)

    eucd_cluster_6 = (eucd_cluster_5.drop(index=index_list_titles_clust5, columns=index_list_titles_clust5)).reset_index(drop=True) 
    col_rename = range(len(eucd_cluster_6.columns))
    eucd_cluster_6.columns = col_rename

    cols = list(eucd_cluster_6.columns)    
    unique_eucd_vals_6 = sorted(list(pd.unique(eucd_cluster_6[cols].values.ravel())))
    unique_eucd_vals_6.remove(float(0))

    simval_clust6 = (np.array(unique_eucd_vals_6[0:1])).astype(str)

    dict_pos_eucd_val6 = {elem: get_indexes(eucd_cluster_6, elem) for elem in simval_clust6}

    index_list_titles_clust6 = []
    find_indexes(dict_pos_eucd_val6, index_list_titles_clust6) # apply the function for finding the indexes we got in the dataframe
    index_list_titles_clust6 = list(set(index_list_titles_clust6)) #creating a set for filtering out duplicate row and col indexes

    global news_df_daily_cluster_6
    news_df_daily_cluster_6 = (news_df_daily_excl_clust_1_2_3_4_5.iloc[index_list_titles_clust6, :])
    
    ### Cluster 7
    # Preparation Cluster 7
    news_df_daily_cluster_6_filtered = news_df_daily_excl_clust_1_2_3_4_5.index.isin(index_list_titles_clust6) # True/False of whichs rows are in the cluster 1
    news_df_daily_excl_clust_1_2_3_4_5_6 = news_df_daily_excl_clust_1_2_3_4_5[~news_df_daily_cluster_6_filtered] # Creating DF excluding the already selected articles
    news_df_daily_excl_clust_1_2_3_4_5_6 = news_df_daily_excl_clust_1_2_3_4_5_6.reset_index(drop=True)

    eucd_cluster_7 = (eucd_cluster_6.drop(index=index_list_titles_clust6, columns=index_list_titles_clust6)).reset_index(drop=True) 
    col_rename = range(len(eucd_cluster_7.columns))
    eucd_cluster_7.columns = col_rename

    cols = list(eucd_cluster_7.columns)    
    unique_eucd_vals_7 = sorted(list(pd.unique(eucd_cluster_7[cols].values.ravel())))
    unique_eucd_vals_7.remove(float(0))

    simval_clust7 = (np.array(unique_eucd_vals_7[0:1])).astype(str)

    dict_pos_eucd_val7 = {elem: get_indexes(eucd_cluster_7, elem) for elem in simval_clust7}

    index_list_titles_clust7 = []
    find_indexes(dict_pos_eucd_val7, index_list_titles_clust7) # apply the function for finding the indexes we got in the dataframe
    index_list_titles_clust7 = list(set(index_list_titles_clust7)) #creating a set for filtering out duplicate row and col indexes

    global news_df_daily_cluster_7
    news_df_daily_cluster_7 = (news_df_daily_excl_clust_1_2_3_4_5_6.iloc[index_list_titles_clust7, :])

    
def rand_info(art_cluster_df, nr_of_art, max_chars):
    global rand_text_, rand_source, rand_link, rand_title, rand_img
    art_cluster_source = list(art_cluster_df['source'])
    art_cluster_link  = list(art_cluster_df['link'])
    art_cluster_title  = list(art_cluster_df['title'])
    art_cluster_text  = list(art_cluster_df['text'])
    art_cluster_img  = list(art_cluster_df['image'])
    
    art_nr_list = (list(art_cluster_df.shape))[0]
    
    random_art_nr = np.random.choice(art_nr_list, nr_of_art, replace=False)  # chosen randomly  
    for nr in random_art_nr:
        (rand_text_.append((art_cluster_text[nr])[:max_chars]))
        (rand_source.append(art_cluster_source[nr]))
        (rand_link.append(art_cluster_link[nr]))
        (rand_title.append(art_cluster_title[nr]))
        (rand_img.append(art_cluster_img[nr]))
    rand_text = [item + '...' for item in rand_text_]   

def create_value_list(df, value_list):
    cols = list(df.columns)   
    for i in (sorted(list(pd.unique(df[cols].values.ravel())), reverse=True)):
        if i < 0.9 and i > 0:
            value_list.append(i)

def extract_stocks():

    url = ("https://financialmodelingprep.com/api/v3/stock/losers")
    losers = get_jsonparsed_data(url)
    url = ("https://financialmodelingprep.com/api/v3/stock/gainers")
    gainers = get_jsonparsed_data(url)

    global df_losers, df_gainers
    df_losers = pd.DataFrame(losers.get("mostLoserStock"))
    df_gainers = pd.DataFrame(gainers.get("mostGainerStock"))
    df_losers = df_losers.sort_values('changesPercentage', ascending=False)
    df_gainers = df_gainers.sort_values('changesPercentage', ascending=False)

    df_losers = df_losers.reset_index()
    df_gainers = df_gainers.reset_index()

    #names of biggest loser and gainer companies
    loser = df_losers['ticker'].values[0]
    gainer = df_gainers['ticker'].values[0]

    #creates urls to access historical data of respective companies
    loser_url = ("https://financialmodelingprep.com/api/v3/historical-price-full/"+ loser +"?timeseries=30")
    gainer_url = ("https://financialmodelingprep.com/api/v3/historical-price-full/"+ gainer +"?timeseries=30")
    names = ["loser", "gainer"]
    name_stock = [loser, gainer]	
    c = 0

    #creates respective graphs for the close values of the last 30 days of the companies and saves it as png
    for i in loser_url, gainer_url:
        response = urlopen(i)
        data = response.read()
        hist = json.loads(data)
        hist.get("historical")
        df = pd.DataFrame(hist.get("historical"))
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date","close"]].copy()
        df.set_index('date', inplace=True)
        df.plot()		
        fig = plt.xlabel('Today`s biggest ' + names[c] + ': ' + name_stock[c] + ' last 30 Days [price]')
        fig = plt.gcf()
        savename = names[c] + "_plot.png"
        fig.savefig(savename, bbox_inches='tight')
        c =+ 1

def get_jsonparsed_data(url):

    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

def create_HTML_file():
    app = flask.Flask('Newsletter')
    global rendered

    with app.app_context():
        rendered = render_template('Newsletter.html',\
            todaysdate = datetime,\
            source00 = rand_source[0],\
            source01 = rand_source[1],\
            source02 = rand_source[2],\
            source10 = rand_source[3],\
            source11 = rand_source[4],\
            source12 = rand_source[5],\
            link00 = rand_link[0],\
            link01 = rand_link[1],\
            link02 = rand_link[2],\
            link10 = rand_link[3],\
            link11 = rand_link[4],\
            link12 = rand_link[5],\
            pic00 = rand_img[0],\
            pic01 = rand_img[1],\
            pic02 = rand_img[2],\
            pic10 = rand_img[3],\
            pic11 = rand_img[4],\
            pic12 = rand_img[5],\
            title00 = rand_title[0],\
            title01 = rand_title[1],\
            title02 = rand_title[2],\
            title10 = rand_title[3],\
            title11 = rand_title[4],\
            title12 = rand_title[5],\
            text00 = rand_text[0],\
            text01 = rand_text[1],\
            text02 = rand_text[2],\
            text10 = rand_text[3],\
            text11 = rand_text[4],\
            text12 = rand_text[5],\
            gainer1 = df_gainers.ticker[0], \
           gainer2 = df_gainers.ticker[1], \
           gainer3 = df_gainers.ticker[2], \
           gainer4 = df_gainers.ticker[3], \
           gainer5 = df_gainers.ticker[4], \
           gainer6 = df_gainers.ticker[5], \
           gainer7 = df_gainers.ticker[6], \
           gainer8 = df_gainers.ticker[7], \
           gainer9 = df_gainers.ticker[8], \
           gainer10 = df_gainers.ticker[9],\
           ch1 = df_gainers.changes[0], \
           ch2 = df_gainers.changes[1], \
           ch3 = df_gainers.changes[2], \
           ch4 = df_gainers.changes[3], \
           ch5 = df_gainers.changes[4], \
           ch6 = df_gainers.changes[5], \
           ch7 = df_gainers.changes[6], \
           ch8 = df_gainers.changes[7], \
           ch9 = df_gainers.changes[8], \
           ch10 = df_gainers.changes[9], \
           pch1 = df_gainers.changesPercentage[0], \
           pch2 = df_gainers.changesPercentage[1], \
           pch3 = df_gainers.changesPercentage[2], \
           pch4 = df_gainers.changesPercentage[3], \
           pch5 = df_gainers.changesPercentage[4], \
           pch6 = df_gainers.changesPercentage[5], \
           pch7 = df_gainers.changesPercentage[6], \
           pch8 = df_gainers.changesPercentage[7], \
           pch9 = df_gainers.changesPercentage[8], \
           pch10 = df_gainers.changesPercentage[9], \
            pp1 =  df_gainers.price[0] , \
           pp2 =  df_gainers.price[1] , \
           pp3 =  df_gainers.price[2] , \
           pp4 =  df_gainers.price[3] , \
           pp5 =  df_gainers.price[4] , \
           pp6 =  df_gainers.price[5] , \
           pp7 =  df_gainers.price[6] , \
           pp8 =  df_gainers.price[7] , \
           pp9 =  df_gainers.price[8] , \
           pp10 =  df_gainers.price[9] , \
            loser1 = df_losers.ticker[0], \
            loser2 = df_losers.ticker[1], \
            loser3 = df_losers.ticker[2], \
            loser4 = df_losers.ticker[3], \
            loser5 = df_losers.ticker[4], \
            loser6 = df_losers.ticker[5], \
            loser7 = df_losers.ticker[6], \
            loser8 = df_losers.ticker[7], \
            loser9 = df_losers.ticker[8], \
            loser10 = df_losers.ticker[9], \
            chan1 = df_losers.changes[0], \
           chan2 = df_losers.changes[1], \
           chan3 = df_losers.changes[2], \
           chan4 = df_losers.changes[3], \
           chan5 = df_losers.changes[4], \
           chan6 = df_losers.changes[5], \
           chan7 = df_losers.changes[6], \
           chan8 = df_losers.changes[7], \
           chan9 = df_losers.changes[8], \
           chan10 = df_losers.changes[9], \
            pchan1 = df_losers.changesPercentage[0], \
           pchan2 = df_losers.changesPercentage[1], \
           pchan3 = df_losers.changesPercentage[2], \
           pchan4 = df_losers.changesPercentage[3], \
           pchan5 = df_losers.changesPercentage[4], \
           pchan6 = df_losers.changesPercentage[5], \
           pchan7 = df_losers.changesPercentage[6], \
           pchan8 = df_losers.changesPercentage[7], \
           pchan9 = df_losers.changesPercentage[8], \
           pchan10 = df_losers.changesPercentage[9], \
            price1 = df_losers.price[0],\
            price2 = df_losers.price[1],\
            price3 = df_losers.price[2],\
            price4 = df_losers.price[3],\
            price5 = df_losers.price[4],\
            price6 = df_losers.price[5],\
            price7 = df_losers.price[6],\
            price8 = df_losers.price[7],\
            price9 = df_losers.price[8],\
            price10 = df_losers.price[9]

                                  )
    f = open(str("NewsLetter_" + today + ".html"),'w', encoding="utf-8")

    message = rendered

    f.write(message)
    f.close()

    #Change path to reflect file location
    filename = "file:///"+os.getcwd()+"/" + str("NewsLetter_" + today + ".html")

    webbrowser.open_new_tab(filename)