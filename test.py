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
import json
import re
import nltk; from nltk.corpus import stopwords; from unidecode import unidecode
import string
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer #for creating count vectors
nltk.download('wordnet')
import warnings
warnings.filterwarnings("ignore")

today = str(date.today()) 
yesterday = str(date.today() - timedelta(days=1))
a = 0

def lemmatize_text(text):
    w_tokenizer, lemmatizer = nltk.tokenize.WhitespaceTokenizer() , nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

def create_newspaper(file): 
    print("Starting scraping...")
    with open(file) as data_file: #Loads the JSON files with news URLs
        companies = json.load(data_file) # change to sources instead of companies

    text_list, source_list, article_list, date_list, time_list, title_list, image_list, keywords_list, summaries_list = [], [], [], [], [], [], [], [], []

    for source, content in companies.items():
        source_list.append(source)
        source_list_unique = set(source_list)
        for rss, links in content.items():
            for url in content["rss"]:
                d = fp.parse(url)
                article = {}
                for entry in d.entries:
                    if hasattr(entry, 'published') and (((dateutil.parser.parse(getattr(entry, 'published'))).strftime("%Y-%m-%d") == today)  or ((dateutil.parser.parse(getattr(entry, 'published'))).strftime("%Y-%m-%d") == yesterday)):
                        article['source'] = source
                        #print(article["source"]) ##
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

                        # "downloading" the articles
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
                        title = content.title #extract article titles
                        image = content.top_image #extract article images
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
    eucl_dist_df.head()
    
    print("Analysis complete.")
    
    article_filtering_eucd(eucl_dist_df)
    
    global art_nr_list, rand_text_, rand_source, rand_link, rand_title, rand_img
    global rand_source
    art_nr_list, rand_text_, rand_source, rand_link, rand_title, rand_img = [], [], [], [], [], []

    rand_info(news_df_daily_cluster_1, 1, 250)
    rand_info(news_df_daily_cluster_2, 1, 250)
    rand_info(news_df_daily_cluster_3, 1, 250)
    rand_info(news_df_daily_cluster_4, 1, 250)
    rand_info(news_df_daily_cluster_5, 1, 250)
    rand_info(news_df_daily_cluster_6, 3, 250)
    
    print("Article selection complete")
	
        
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

    
def rand_info(art_cluster_df, nr_of_art, max_chars):
    global rand_text
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

