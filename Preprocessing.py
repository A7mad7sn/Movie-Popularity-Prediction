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

def Preprocessing_Training_Regression(training_data):
    
    #reindexing:
    training_data = training_data.reset_index(drop=True)
    
    #handling the missing value in runtime:
    train_mean = training_data['runtime'].mean()
    training_data['runtime'] = training_data['runtime'].fillna(train_mean)
    
    #handling the missing value in homepage:
    training_data['homepage'] = training_data['homepage'].fillna(
        'http://www.' + training_data['original_title'].str.replace(' ', '').str.lower() + '.com/')
    
    #handling the missing value in tagline,overview:
    training_data.dropna(inplace=True)
    training_data = training_data.reset_index(drop=True)
    
    # handle release date colomn:
    training_data['release_date'] = pd.to_datetime(training_data['release_date'])
    training_data['release_day'] = training_data['release_date'].dt.day
    training_data['release_month'] = training_data['release_date'].dt.month
    training_data['release_year'] = training_data['release_date'].dt.year
    training_data.drop(columns=['release_date'], inplace=True)

    #Handeling overview column:
    sentiments = []
    for text in training_data['overview']:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiments.append(2)
        elif polarity < 0:
            sentiments.append(1)
        else:
            sentiments.append(0)
    training_data['overview'] = sentiments
        
    #Label Encoding :
    lbls = []
    cols = ('status', 'original_language', 'original_title', 'tagline', 'homepage', 'title')
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(training_data[c].values))
        training_data[c] = lbl.transform(list(training_data[c].values))
        lbls.append(lbl)
        
    #MultiLabelBinalizer (one hot encoding):
    training_data = list_dict_encoding(training_data, 'genres', 'genres_ids', 'genres_name', "id", "name")
    training_data = list_dict_encoding(training_data, 'spoken_languages', 'spoken_languages_ids', 'spoken_languages_name',
                                    "iso_639_1", "name")
    training_data = list_dict_encoding(training_data, 'production_countries', 'production_countries_ids',
                                    'production_countries_name', "iso_3166_1", "name")
    training_data = list_dict_encoding(training_data, 'production_companies', 'production_companies_ids',
                                    'production_companies_name', "id", "name")
    training_data = list_dict_encoding(training_data, 'keywords', 'keywords_ids', 'keywords_name', "id", "name")
    
    one_counts = training_data.iloc[:,:].sum()
    cols_to_drop = one_counts[one_counts < training_data.shape[0]/4].index
    training_data = training_data.drop(cols_to_drop, axis=1)

    columns = list(training_data.columns.values)
    columns.pop(columns.index('vote_average'))
    training_data = training_data[columns+['vote_average']]
    
    X_train = training_data.iloc[:,0:-1]
    Y_train = training_data.iloc[:,-1] 
    training_data = pd.concat([X_train, Y_train], axis=1, join="inner")
    
    #Feature Selection:
    corr = training_data.corr()
    top_feature = corr.index[abs(corr['vote_average']) > 0.15]
    plt.subplots(figsize=(12, 8))
    top_corr = training_data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    cur_top_feature = top_feature
    training_data = training_data[top_feature]
    top_feature = top_feature.delete(-1)
    print("Number of top features:", len(top_feature))
    
    #feature scaling:
    X_train = training_data.iloc[:,0:-1]
    Y_train = training_data.iloc[:,-1]
    scaler = MinMaxScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
    training_data = pd.concat([X_train, Y_train], axis=1, join="inner")
    
    Trained_variables = lbls,scaler,train_mean,cols_to_drop,cur_top_feature
        
    return training_data,Trained_variables



def Preprocessing_Testing_Regression(testing_data,Trained_variables):
    
    #retrieving needed variables form traning data:
    lbls,scaler,train_mean,cols_to_drop,top_feature = Trained_variables
    
    #reindexing:
    testing_data = testing_data.reset_index(drop=True)
    
    
    #handling the missing value in runtime:
    testing_data['runtime'] = testing_data['runtime'].fillna(train_mean)
    
    #handling the missing value in homepage:
    testing_data['homepage'] = testing_data['homepage'].fillna(
        'http://www.' + testing_data['original_title'].str.replace(' ', '').str.lower() + '.com/')
    
    #handling the missing value in tagline,overview:
    testing_data.dropna(inplace=True)
    testing_data = testing_data.reset_index(drop=True)
    
    # handle release date colomn:
    testing_data['release_date'] = pd.to_datetime(testing_data['release_date'])
    testing_data['release_day'] = testing_data['release_date'].dt.day
    testing_data['release_month'] = testing_data['release_date'].dt.month
    testing_data['release_year'] = testing_data['release_date'].dt.year
    testing_data.drop(columns=['release_date'], inplace=True)
    
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
    
    #Label Encoding :
    cols = ('status', 'original_language', 'original_title', 'tagline', 'homepage', 'title')
    i = 0
    for c in cols:
        lbl = lbls[i]
        testing_data[c] = testing_data[c].map(lambda s: '<others>' if s not in lbl.classes_ else s)
        lbl.classes_ = np.append(lbl.classes_, '<others>')
        testing_data[c] = lbl.transform(testing_data[c])
        i = i+1
        
    #MultiLabelBinalizer (one hot encoding):
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
    cols_to_drop2 = one_counts2[one_counts2 < testing_data.shape[0]/4].index
    testing_data = testing_data.drop([x for x in cols_to_drop2 if x not in top_feature], axis=1)
    
    columns = list(testing_data.columns.values)
    columns.pop(columns.index('vote_average'))
    testing_data = testing_data[columns+['vote_average']]
    
    #Feature Selection:
    testing_data = testing_data[top_feature]
    
    #feature scaling:
    X_test = testing_data.iloc[:,0:-1]
    Y_test = testing_data.iloc[:,-1]
    X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
    testing_data = pd.concat([X_test, Y_test], axis=1, join="inner")
    
    return testing_data

def Test_Script_Regression(testing_data,Trained_variables):
    
    #retrieving needed variables form traning data:
    lbls,scaler,train_mean,cols_to_drop,top_feature = Trained_variables
    
    #reindexing:
    testing_data = testing_data.reset_index(drop=True)
    
    
    #handling the missing value in runtime:
    testing_data['runtime'] = testing_data['runtime'].fillna(train_mean)
    
    #handling the missing value in homepage:
    testing_data['homepage'] = testing_data['homepage'].fillna(
        'http://www.' + testing_data['original_title'].str.replace(' ', '').str.lower() + '.com/')
    
    #handling the missing value in tagline,overview:
    testing_data.fillna('Undefined')
    
    
    # handle release date colomn:
    testing_data['release_date'] = pd.to_datetime(testing_data['release_date'])
    testing_data['release_day'] = testing_data['release_date'].dt.day
    testing_data['release_month'] = testing_data['release_date'].dt.month
    testing_data['release_year'] = testing_data['release_date'].dt.year
    testing_data.drop(columns=['release_date'], inplace=True)
    
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
    
    #Label Encoding :
    cols = ('status', 'original_language', 'original_title', 'tagline', 'homepage', 'title')
    i = 0
    for c in cols:
        lbl = lbls[i]
        testing_data[c] = testing_data[c].map(lambda s: '<others>' if s not in lbl.classes_ else s)
        lbl.classes_ = np.append(lbl.classes_, '<others>')
        testing_data[c] = lbl.transform(testing_data[c])
        i = i+1
        
    #MultiLabelBinalizer (one hot encoding):
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
    cols_to_drop2 = one_counts2[one_counts2 < testing_data.shape[0]/4].index
    testing_data = testing_data.drop([x for x in cols_to_drop2 if x not in top_feature], axis=1)
    
    columns = list(testing_data.columns.values)
    columns.pop(columns.index('vote_average'))
    testing_data = testing_data[columns+['vote_average']]
    
    
    #Feature Selection:
    testing_data = testing_data[top_feature]
    
    #feature scaling:
    X_test = testing_data.iloc[:,0:-1]
    Y_test = testing_data.iloc[:,-1]
    X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
    testing_data = pd.concat([X_test, Y_test], axis=1, join="inner")
    
    return testing_data

def Preprocessing_Training_Classification(training_data):

    #reindexing:
    training_data = training_data.reset_index(drop=True)
    
    #handling the missing value in runtime:
    train_mean = training_data['runtime'].mean()
    training_data['runtime'] = training_data['runtime'].fillna(train_mean)
    
    #handling the missing value in homepage:
    training_data['homepage'] = training_data['homepage'].fillna(
        'http://www.' + training_data['original_title'].str.replace(' ', '').str.lower() + '.com/')
    
    #handling the missing value in tagline,overview:
    training_data.dropna(inplace=True)
    training_data = training_data.reset_index(drop=True)
    
    # handle release date colomn:
    training_data['release_date'] = pd.to_datetime(training_data['release_date'])
    training_data['release_day'] = training_data['release_date'].dt.day
    training_data['release_month'] = training_data['release_date'].dt.month
    training_data['release_year'] = training_data['release_date'].dt.year
    training_data.drop(columns=['release_date'], inplace=True)

    #Handeling overview column:
    sentiments = []
    for text in training_data['overview']:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiments.append(2)
        elif polarity < 0:
            sentiments.append(1)
        else:
            sentiments.append(0)
    training_data['overview'] = sentiments
        
    #Label Encoding :
    lbls = []
    cols = ('status', 'original_language', 'original_title', 'tagline', 'homepage', 'title','Rate')
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(training_data[c].values))
        training_data[c] = lbl.transform(list(training_data[c].values))
        lbls.append(lbl)
        
    #MultiLabelBinalizer (one hot encoding):
    training_data = list_dict_encoding(training_data, 'genres', 'genres_ids', 'genres_name', "id", "name")
    training_data = list_dict_encoding(training_data, 'spoken_languages', 'spoken_languages_ids', 'spoken_languages_name',
                                    "iso_639_1", "name")
    training_data = list_dict_encoding(training_data, 'production_countries', 'production_countries_ids',
                                    'production_countries_name', "iso_3166_1", "name")
    training_data = list_dict_encoding(training_data, 'production_companies', 'production_companies_ids',
                                    'production_companies_name', "id", "name")
    training_data = list_dict_encoding(training_data, 'keywords', 'keywords_ids', 'keywords_name', "id", "name")
    
    one_counts = training_data.iloc[:,:].sum()
    cols_to_drop = one_counts[one_counts < training_data.shape[0]/4].index
    training_data = training_data.drop(cols_to_drop, axis=1)

    columns = list(training_data.columns.values)
    columns.pop(columns.index('Rate'))
    training_data = training_data[columns+['Rate']]
    
    
    X_train = training_data.iloc[:,0:-1]
    Y_train = training_data.iloc[:,-1]
    training_data = pd.concat([X_train, Y_train], axis=1, join="inner")
    
    #Feature Selection:
    corr = training_data.corr()
    top_feature = corr.index[abs(corr['Rate']) > 0.04]#0.07 may be better
    plt.subplots(figsize=(12, 8))
    top_corr = training_data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    cur_top_feature = top_feature
    training_data = training_data[top_feature]
    top_feature = top_feature.delete(-1)
    print("Number of top features:", len(top_feature))
    
    #feature scaling:
    X_train = training_data.iloc[:,0:-1]
    Y_train = training_data.iloc[:,-1]
    scaler = MinMaxScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
    training_data = pd.concat([X_train, Y_train], axis=1, join="inner")
    
    Trained_variables = lbls,scaler,train_mean,cols_to_drop,cur_top_feature
        
    return training_data,Trained_variables

def Preprocessing_Testing_Classification(testing_data,Trained_variables):
    
    #retrieving needed variables form traning data:
    lbls,scaler,train_mean,cols_to_drop,top_feature = Trained_variables
    
    #reindexing:
    testing_data = testing_data.reset_index(drop=True)
    
    
    #handling the missing value in runtime:
    testing_data['runtime'] = testing_data['runtime'].fillna(train_mean)
    
    #handling the missing value in homepage:
    testing_data['homepage'] = testing_data['homepage'].fillna(
        'http://www.' + testing_data['original_title'].str.replace(' ', '').str.lower() + '.com/')
    
    #handling the missing value in tagline,overview:
    testing_data.dropna(inplace=True)
    testing_data = testing_data.reset_index(drop=True)
    
    # handle release date colomn:
    testing_data['release_date'] = pd.to_datetime(testing_data['release_date'])
    testing_data['release_day'] = testing_data['release_date'].dt.day
    testing_data['release_month'] = testing_data['release_date'].dt.month
    testing_data['release_year'] = testing_data['release_date'].dt.year
    testing_data.drop(columns=['release_date'], inplace=True)
    
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
    
    #Label Encoding :
    cols = ('status', 'original_language', 'original_title', 'tagline', 'homepage', 'title' ,'Rate')
    i = 0
    for c in cols:
        lbl = lbls[i]
        testing_data[c] = testing_data[c].map(lambda s: '<others>' if s not in lbl.classes_ else s)
        lbl.classes_ = np.append(lbl.classes_, '<others>')
        testing_data[c] = lbl.transform(testing_data[c])
        i = i+1
        
    #MultiLabelBinalizer (one hot encoding):
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
    cols_to_drop2 = one_counts2[one_counts2 < testing_data.shape[0]/4].index
    testing_data = testing_data.drop([x for x in cols_to_drop2 if x not in top_feature], axis=1)
    
    columns = list(testing_data.columns.values)
    columns.pop(columns.index('Rate'))
    testing_data = testing_data[columns+['Rate']]
    
    
    #Feature Selection:
    testing_data = testing_data[top_feature]
    
    #feature scaling:
    X_test = testing_data.iloc[:,0:-1]
    Y_test = testing_data.iloc[:,-1]
    X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
    testing_data = pd.concat([X_test, Y_test], axis=1, join="inner")
    
    return testing_data

def Test_Script_Classification(testing_data,Trained_variables):
    
    #retrieving needed variables form traning data:
    lbls,scaler,train_mean,cols_to_drop,top_feature = Trained_variables
    
    #reindexing:
    testing_data = testing_data.reset_index(drop=True)
    
    
    #handling the missing value in runtime:
    testing_data['runtime'] = testing_data['runtime'].fillna(train_mean)
    
    #handling the missing value in homepage:
    testing_data['homepage'] = testing_data['homepage'].fillna(
        'http://www.' + testing_data['original_title'].str.replace(' ', '').str.lower() + '.com/')
    
    #handling the missing value in tagline,overview:
    testing_data.fillna('Undefined')
    
    # handle release date colomn:
    testing_data['release_date'] = pd.to_datetime(testing_data['release_date'])
    testing_data['release_day'] = testing_data['release_date'].dt.day
    testing_data['release_month'] = testing_data['release_date'].dt.month
    testing_data['release_year'] = testing_data['release_date'].dt.year
    testing_data.drop(columns=['release_date'], inplace=True)
    
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
    
    #Label Encoding :
    cols = ('status', 'original_language', 'original_title', 'tagline', 'homepage', 'title' ,'Rate')
    i = 0
    for c in cols:
        lbl = lbls[i]
        testing_data[c] = testing_data[c].map(lambda s: '<others>' if s not in lbl.classes_ else s)
        lbl.classes_ = np.append(lbl.classes_, '<others>')
        testing_data[c] = lbl.transform(testing_data[c])
        i = i+1
        
    #MultiLabelBinalizer (one hot encoding):
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
    cols_to_drop2 = one_counts2[one_counts2 < testing_data.shape[0]/4].index
    testing_data = testing_data.drop([x for x in cols_to_drop2 if x not in top_feature], axis=1)
    
    columns = list(testing_data.columns.values)
    columns.pop(columns.index('Rate'))
    testing_data = testing_data[columns+['Rate']]
    
    
    #Feature Selection:
    testing_data = testing_data[top_feature]
    
    #feature scaling:
    X_test = testing_data.iloc[:,0:-1]
    Y_test = testing_data.iloc[:,-1]
    X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
    testing_data = pd.concat([X_test, Y_test], axis=1, join="inner")
    
    return testing_data