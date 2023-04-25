import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer ,MinMaxScaler


def list_dict_encoding(data, colomn_name, new_id_col, new_name_col, id_key_name, name_key_name):
    ids = []
    names = []
    for row in data[colomn_name]:
        lists = eval(row)  # convert string to list of dictionaries
        d_ids = []
        d_names = []
        for D in lists:
            d_ids.append(D[id_key_name])
            d_names.append(D[name_key_name])
        ids.append(d_ids)
        names.append(d_names)
    data[new_id_col] = ids
    data[new_name_col] = names
    mlb = MultiLabelBinarizer()
    name_encoded = pd.DataFrame(mlb.fit_transform(data[new_name_col]), columns=mlb.classes_)
    data = pd.concat([data, name_encoded], axis=1)
    colo_to_drop = [colomn_name, new_id_col, new_name_col]
    data = data.drop(columns=colo_to_drop)
    return data

def Preprocessing(Movie_Data,testing_data):
    
    testing_data['runtime'] = testing_data['runtime'].fillna(Movie_Data['runtime'].mean())
    
    #handling the missing value in runtime:
    Movie_Data['runtime'] = Movie_Data['runtime'].fillna(Movie_Data['runtime'].mean())
    
    #handling the missing value in homepage:
    Movie_Data['homepage'] = Movie_Data['homepage'].fillna(
        'http://www.' + Movie_Data['original_title'].str.replace(' ', '').str.lower() + '.com/')
    
    testing_data['homepage'] = testing_data['homepage'].fillna(
        'http://www.' + testing_data['original_title'].str.replace(' ', '').str.lower() + '.com/')
    
    #handling the missing value in tagline,overview:
    Movie_Data.dropna(inplace=True)
    Movie_Data = Movie_Data.reset_index(drop=True)

    testing_data.dropna(inplace=True)
    testing_data = testing_data.reset_index(drop=True)
    
    # handle release date colomn:
    Movie_Data['release_date'] = pd.to_datetime(Movie_Data['release_date'])
    Movie_Data['release_day'] = Movie_Data['release_date'].dt.day
    Movie_Data['release_month'] = Movie_Data['release_date'].dt.month
    Movie_Data['release_year'] = Movie_Data['release_date'].dt.year
    Movie_Data.drop(columns=['release_date'], inplace=True)

    testing_data['release_date'] = pd.to_datetime(testing_data['release_date'])
    testing_data['release_day'] = testing_data['release_date'].dt.day
    testing_data['release_month'] = testing_data['release_date'].dt.month
    testing_data['release_year'] = testing_data['release_date'].dt.year
    testing_data.drop(columns=['release_date'], inplace=True)
    
    #Handeling overview column:
    sentiments = []
    for text in Movie_Data['overview']:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiments.append(2)
        elif polarity < 0:
            sentiments.append(1)
        else:
            sentiments.append(0)
    Movie_Data['overview'] = sentiments

    #Handeling overview column:
    sentiments = []
    for text in testing_data['overview']:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiments.append(2)
        elif polarity < 0:
            sentiments.append(1)
        else:
            sentiments.append(0)
    testing_data['overview'] = sentiments
    
    
    
    #Label Encoding:
    cols = ('status', 'original_language', 'original_title', 'tagline', 'homepage', 'title')
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(Movie_Data[c].values))
        testing_data[c] = testing_data[c].map(lambda s: '<unknown>' if s not in lbl.classes_ else s)
        lbl.classes_ = np.append(lbl.classes_, '<unknown>')
        Movie_Data[c] = lbl.transform(list(Movie_Data[c].values))
        testing_data[c] = lbl.transform(testing_data[c])
        
    
    
    Movie_Data = list_dict_encoding(Movie_Data, 'genres', 'genres_ids', 'genres_name', "id", "name")
    Movie_Data = list_dict_encoding(Movie_Data, 'spoken_languages', 'spoken_languages_ids', 'spoken_languages_name',
                                    "iso_639_1", "name")
    Movie_Data = list_dict_encoding(Movie_Data, 'production_countries', 'production_countries_ids',
                                    'production_countries_name', "iso_3166_1", "name")
    Movie_Data = list_dict_encoding(Movie_Data, 'production_companies', 'production_companies_ids',
                                    'production_companies_name', "id", "name")
    Movie_Data = list_dict_encoding(Movie_Data, 'keywords', 'keywords_ids', 'keywords_name', "id", "name")
    
    
    one_counts = Movie_Data.iloc[:,:].sum()
    cols_to_drop = one_counts[one_counts < Movie_Data.shape[0]/4].index
    Movie_Data = Movie_Data.drop(cols_to_drop, axis=1)
    
    

    columns2 = list(Movie_Data.columns.values)
    columns2.pop(columns2.index('vote_average'))
    Movie_Data = Movie_Data[columns2+['vote_average']]
    
    
    testing_data = list_dict_encoding(testing_data, 'genres', 'genres_ids', 'genres_name', "id", "name")
    testing_data = list_dict_encoding(testing_data, 'spoken_languages', 'spoken_languages_ids', 'spoken_languages_name',
                                    "iso_639_1", "name")
    testing_data = list_dict_encoding(testing_data, 'production_countries', 'production_countries_ids',
                                    'production_countries_name', "iso_3166_1", "name")
    testing_data = list_dict_encoding(testing_data, 'production_companies', 'production_companies_ids',
                                    'production_companies_name', "id", "name")
    testing_data = list_dict_encoding(testing_data, 'keywords', 'keywords_ids', 'keywords_name', "id", "name")

    testing_data = testing_data.drop([x for x in cols_to_drop if x in testing_data.columns], axis=1) #errors='ignore'#)
    
    one_counts2 = testing_data.iloc[:,:].sum()
    cols_to_drop2 = one_counts2[one_counts2 < Movie_Data.shape[0]/4].index
    testing_data = testing_data.drop([x for x in cols_to_drop2 if x !='status' and x != 'Action' and x != 'Comedy' and x != 'Drama' and x != 'Thriller' and x != 'English' and x != 'United States of America'], axis=1)
    
    columns = list(testing_data.columns.values)
    columns.pop(columns.index('vote_average'))
    testing_data = testing_data[columns+['vote_average']]
    
    
    
    #feature scaling:
    X_train = Movie_Data.iloc[:,0:-1]
    Y_train = Movie_Data.iloc[:,-1]
    X_test = testing_data.iloc[:,0:-1]
    Y_test = testing_data.iloc[:,-1]
    scaler = MinMaxScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
    X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
    Movie_Data = pd.concat([X_train, Y_train], axis=1, join="inner")
    testing_data = pd.concat([X_test, Y_test], axis=1, join="inner")
    

    
    #Feature Selection:
    corr = Movie_Data.corr()
    top_feature = corr.index[abs(corr['vote_average']) > 0.15]
    plt.subplots(figsize=(12, 8))
    top_corr = Movie_Data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    Movie_Data = Movie_Data[top_feature]
    testing_data = testing_data[top_feature]
    top_feature = top_feature.delete(-1)
    print("Number of top features:", len(top_feature))
    
    return Movie_Data,testing_data
