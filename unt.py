import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

#=======================================================#

def options():
    pd.set_option('display.width', 300)
    pd.set_option('max_columns', 60)

def read_file(name: str):
    data = pd.read_csv('flavors_of_cacao.csv')
    return data

def print_rating(data: pd.DataFrame, rate: int):
    if (rate == 1):
        print("Least liked chocolates: ")
    if (rate == 5):
        print("Most liked chocolates: ")
    print(data[data['Rating'] == rate])

def dumb(X: pd.DataFrame, s: str):
    dummies = pd.get_dummies(X[s])
    dummies.columns = [s + '_' + k for k in dummies.columns.values]
    X = pd.concat([X, dummies], axis = 1)
    del X[s]
    return X

def train_train_train(X: pd.DataFrame, s: str):
    tf = TfidfVectorizer(ngram_range=(1,2))
    tf.fit(X[s])

    train_transformed = tf.transform(X[s])

    train_transformed = pd.DataFrame(data = train_transformed.todense(),
                                     index = X.index.values,
                                     columns = [s + '_' + k for k in tf.vocabulary_])

    X = pd.concat([X, train_transformed], axis=1)

    del X[s]
    return X

def print_some_data(data: pd.DataFrame, s1 : str, s2: str):
    print("Number of bars for " + s2 + ": " + str(data[data[s1] == s2].shape[0]))
    print(data[data[s1] == s2])
    print('------------------------------------')

def print_some_ratings(data: pd.DataFrame, s1: str, s2: str):
    print("Number of bars for " + s2 + ": " + str(data[data[s1] == s2].shape[0]))
    print("Average " + s2 + " bar rating: " + str(data[data[s1] == s2]['Rating'].mean()))
    print('------------------------------------')

def print_some_masks(data: pd.DataFrame, s1: str, s2: str):
    mask = data[s1].apply(lambda x: s2 in str(x).lower())
    print("Number of bars for " + s2 + ": " + str(data[mask].shape[0]))
    print(data[mask])
    print('------------------------------------')

#=======================================================#

def main():
    start_time = time.time()
    options()
    data = read_file('flavors_of_cacao.csv')
    
    for c in data.columns.values:
        if c != 'Rating':
            data[c] = data[c].apply(lambda x: str(x).replace(u'\xa0',''))

    #print("Number of records: " + str(data.shape[0]))
    #print("baseline Average Rating: " + str(data['Rating'].mean()))

    #print_rating(data, 1)
    #print_rating(data, 5)

    data['CocoaPercent'] = data['CocoaPercent'].apply(lambda x: float(str(x).replace('%','')))

    #=======================================================#
    
    #plot #1
    """
    sns.countplot(data['Rating']).set_title('Distribution Over Chocolate Ratings')
    """

    #plot #2
    """
    #sns.countplot(data['Review\Date']).set_title('Rating Volume Over Time')
    """
    
    #plot #3
    """
    sns.set_style("darkgrid")
    sns.regplot(x=data['ReviewDate'].apply(lambda x: float(x)),
                y=data['Rating'].apply(lambda x: float(x)))
    plt.title('Rating Over Time')
    """

    #plot #4
    """
    print("Total unique Cocoa Percent: " + str(len(data['CocoaPercent'].unique())))
    sns.set_style("darkgrid")
    sns.regplot(x=data['CocoaPercent'],
                y=data['Rating'])
    plt.title('Rating By Cocoa Percentage')
    """
    
    plt.show()

    #=======================================================#

    data.replace('',np.nan,).isnull().sum()
    
    data['BeanType'].fillna("Unknown", inplace = True)
    data['BeanType'].replace("nan", "Unknown", inplace = True)

    #=======================================================#

    """
    print('------------------------------------')
    print("Total unique bean types: " + str(len(data['BeanType'].unique())))
    f = {'Rating': ['size', 'mean', 'std']}
    print(data.groupby('BeanType').agg(f))

    print('------------------------------------')
    print("Total unique bean types: " + str(len(data['CompanyLocation'].unique())))
    f = {'Rating': ['size', 'mean', 'std']}
    print(data.groupby('CompanyLocation').agg(f))

    print('------------------------------------')
    print("Total unique Specific Bean Origin or Bar Name: " + str(len(data['SpecificOrigin/BarName'].unique())))
    print("Total unique REF: " + str(len(data['REF'].unique())))
    """

    #=======================================================#
    
    X = data.drop('Rating', axis = 1)
    y = data['Rating']

    X = dumb(X, 'Company')
    X = dumb(X, 'CompanyLocation')
    X = dumb(X, 'REF')
    X = dumb(X, 'ReviewDate')

    #=======================================================#

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

    X_train = train_train_train(X_train, 'SpecificOrigin/BarName')
    X_test = train_train_train(X_test, 'SpecificOrigin/BarName')

    X_train = train_train_train(X_train, 'BeanType')
    X_test = train_train_train(X_test, 'BeanType')

    X_train = train_train_train(X_train, 'BroadBeanOrigin')
    X_test = train_train_train(X_test, 'BroadBeanOrigin')

    #=======================================================#

    sc = StandardScaler()
    sc.fit(X_train['CocoaPercent'].values.reshape(-1,1))
    X_train['CocoaPercent'] = sc.transform(X_train['CocoaPercent'].values.reshape(-1,1))
    X_test['CocoaPercent'] = sc.transform(X_test['CocoaPercent'].values.reshape(-1,1))

    print(X_train.head())
    print('------------------------------------')

    #=======================================================#

    
    param_grid = {'n_estimators': [10, 30, 50, 90], 
                  'max_depth': [5, 10, 20, None]
                 }
    regr = RandomForestRegressor()
    print('1')
    print("%s seconds" % (time.time() - start_time))
    
    model = GridSearchCV(regr, param_grid, cv=3)
    print('2')
    print("%s seconds" % (time.time() - start_time))

    model.fit(np.matrix(X_train), y_train)
    print('3')
    print("%s seconds" % (time.time() - start_time))

    regr = model.best_estimator_
    print('4')
    print("%s seconds" % (time.time() - start_time))
    print(model.best_score_)
    

    #=======================================================#

    """
    sorted_indices = np.argsort(regr.feature_importances_)
    variables = regr.feature_importances_[sorted_indices]
    importance_rating = X_train.columns.values[sorted_indices]

    importances = pd.DataFrame({'variable': variables,
                                'importance': importance_rating})
    print(importances.tail(10))
    """
    
    #=======================================================#

    #print_some_masks(data, 'SpecificOrigin/BarName', 'honduras')
    #print_some_masks(data, 'SpecificOrigin/BarName', 'del toro')
    
    #=======================================================#

    #print_some_ratings(data, 'Company', 'Soma')
    #print_some_ratings(data, 'REF', '887')

    #=======================================================#

    """
    param_grid = {'alpha': [0.001, 0.01, 1, 3, 5, 10]
                  }
    regr = Ridge()
    model = GridSearchCV(regr, param_grid, cv=3)
    model.fit(np.matrix(X_train), y_train)
    regr = model.best_estimator_
    print(model.best_score_)

    #=======================================================#

    sorted_indices = np.argsort(regr.coef_)
    variables = regr.coef_[sorted_indices]
    importance_rating = X_train.columns.values[sorted_indices]
    importances = pd.DataFrame({'variable':variables, 'coefficient':importance_rating})
    print("Total non zero coefficients: " + str(len(importances[importances['coefficient'] != 0.0])))
    """

    #print(importances.head())

    #=======================================================#

    #print(importances.tail())

    #=======================================================#

    #print_some_data(data, 'Company', 'Callebaut')
    #print_some_data(data, 'Company', 'Amedei')
    #print_some_data(data, 'REF', '111')
    #print_some_data(data, 'Company', 'Patric')
    #print_some_data(data, 'Company', 'Cacao Sampaka')

    #=======================================================#

if __name__ == "__main__":
    main()
