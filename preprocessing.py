import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


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


pd.set_option('display.max_rows', 100)


def Preprocessing(Movie_Data):
    Y = Movie_Data['vote_average']
    Movie_Data = Movie_Data.drop(columns=['vote_average'])
    
    #handling the missing value in runtime:
    Movie_Data['runtime'] = Movie_Data['runtime'].fillna(Movie_Data['runtime'].mean())
    
    # handling the missing value in homepage
    Movie_Data['homepage'] = Movie_Data['homepage'].fillna(
        'http://www.' + Movie_Data['original_title'].str.replace(' ', '').str.lower() + '.com/')
    
    # handle release date colomn
    Movie_Data['release_date'] = pd.to_datetime(Movie_Data['release_date'])
    Movie_Data['release_year'] = Movie_Data['release_date'].dt.year
    Movie_Data['release_month'] = Movie_Data['release_date'].dt.month
    Movie_Data['release_day'] = Movie_Data['release_date'].dt.day
    Movie_Data.drop(columns=['release_date'], inplace=True)
    Movie_Data.drop(columns=['overview'], inplace=True)
    
    Movie_Data.dropna(inplace=True)
    Movie_Data = Movie_Data.reset_index(drop=True)
    
    # use label encoder on these colomn
    cols = ('status', 'original_language', 'original_title', 'tagline', 'homepage', 'title')
    Movie_Data = Feature_Encoder(Movie_Data, cols)
    
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
    Movie_Data = pd.concat([Movie_Data, Y], axis=1, join="inner")
    corr = Movie_Data.corr()
    top_feature = corr.index[abs(corr['vote_average']) > 0.019]
    #Correlation plot
    plt.subplots(figsize=(12, 8))
    top_corr = Movie_Data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    Movie_Data = Movie_Data[top_feature]
    top_feature = top_feature.delete(-1)
    print("Number of top features:", len(top_feature))
    #scaling features
    X = Movie_Data[top_feature]
    X = featureScaling(X, 0, 1)
    X = pd.DataFrame(X,columns=top_feature)
    Movie_Data = pd.concat([X, Y], axis=1, join="inner")
    return Movie_Data
