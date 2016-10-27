#!/bin/python

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
plt.ion()

from vaderSentiment.vaderSentiment import sentiment

#metrics
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve

#preprocessing
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

#models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

#hyperparameter tuning
from sklearn.model_selection import GridSearchCV #upgrade to model_selection in new sklearn

def read_data():
    '''
    Read CSV files
    '''
    combined = pd.read_csv("Combined_News_DJIA.csv")
    stock = pd.read_csv("DJIA_table.csv")
    news = pd.read_csv("RedditNews.csv")

    return stock, news, combined

def prepare_stock_data(stock, frequency_in_days=1, start_col='Open', end_col='Open'):
    '''
    Input: raw stock data
    Output: targets to predict - one for each date at desired frequency
    '''
    stock = stock.copy()

    if len(stock["Date"].unique()) != stock.shape[0]:
           print "There are duplicates data in stock data. Exiting..."
           return None

    #sort by date and assign id to each data
    stock.sort("Date", ascending=True, inplace=True)
    stock['date_id'] = np.arange(stock.shape[0])
    stock.set_index('Date', inplace=True)

    #start col
    stock_prev = stock.shift(frequency_in_days) #for clarity
    stock_prev.columns = [col_name + "_prev" for col_name in stock_prev.columns]

    stock_curr = stock.copy() #copy
    stock_curr.columns = [col_name + "_curr" for col_name in stock_curr.columns]

    stock_diff = pd.concat([stock_prev, stock_curr], axis=1)
    stock_diff['target'] = stock_diff['{}_curr'.format(end_col)] / stock_diff['{}_prev'.format(start_col)] - 1
    stock_diff['date_id_diff'] = stock_diff['date_id_curr'] - stock_diff['date_id_prev']
    stock_diff['binary_target'] = stock_diff['target'].apply(lambda x: (x / np.abs(x) + 1)/2) #binary target - if % change > 0, binary = 1, else 0 (don't use if-else statements/pipelining)
    
    #drop nulls - number of dropped rows should equal frequency_in_days
    pre_rows = stock_diff.shape[0]
    stock_diff.dropna(inplace=True)
    post_rows = stock_diff.shape[0]

    #can get more dropped rows if target = 0 (2016-05-12 and 2016-05-13)
    #if pre_rows - post_rows != frequency_in_days:
    #    print "Dropping more nulls than expected..."
    #    return None

    unique_date_diff = stock_diff['date_id_diff'].unique()
    if len(unique_date_diff) > 1 or unique_date_diff[0] != frequency_in_days:
        print "Check steps above..."
        return None
    
    #keep relevant rows
    stock_diff = stock_diff[['{}_prev'.format(start_col), '{}_curr'.format(end_col), 'target', 'binary_target']]

    return stock, stock_prev, stock_curr, stock_diff

def prepare_news_data_vader(news):
    '''
    Get sentiment scores for all 25 headlines on each day
    Input: raw news table
    Output: rows = # unique dates, cols = 25 (one for each headline)
    '''

    news = news.copy()

    #sort by date and assign top id to each item
    news['News'] = news['News'].apply(lambda x: x.lower().replace(',','').replace('.','')) #convert to lower case and some trimming
    news.sort('Date', ascending=True, inplace=True)
    news['sentiment_score'] = news['News'].apply(lambda x: sentiment(x)['compound'])

    #rank index for each day
    news_np = np.array(news) #each row is [Date, news headline, score]
    news_np_score = []
    
    counter = 1
    current_date = -1
    for i in xrange(news_np.shape[0]):
        row = news_np[i]
        if row[0] != current_date:
            current_date = row[0]
            counter = 1
        else:
            counter += 1

        news_np_score.append(np.append(row, counter))

    news_score = pd.DataFrame(news_np_score, columns = ['Date', 'News', 'SentimentScore', 'Rank'])
    news_score = pd.pivot_table(news_score, values='SentimentScore', index='Date', columns='Rank')
    news_score = news_score.iloc[:,0:25] #two days have 50 articles
    news_score.columns = ['col_vader_{}'.format(i) for i in news_score.columns]

    return news_score

def prepare_news_data_tfidf(news, max_features = 1000, use_stop_words = False):
    '''
    Perform tfidf on news headlines
    Input: raw news headlines
    Output: tfidf vectors averaged and normalized across all 25 headlines for each day. #rows = #unique dates, #cols = max_features
    '''
    news = news.copy()

    news['News'] = news['News'].apply(lambda x: x.lower().replace(',','').replace('.','')) #convert to lower case and some trimming

    transformer = train_tfidf(news['News'], max_features = max_features, use_stop_words = use_stop_words)

    news_transformed = transformer.transform(news['News']).toarray()
    news_transformed = pd.DataFrame(news_transformed)
    news_transformed['Date'] = news['Date']

    news_transformed = news_transformed.groupby('Date').sum() #add the vectors for 25 articles on each day
    news_transformed = news_transformed.apply(lambda x: x / np.sqrt(np.sum(x**2)), axis=1)
    news_transformed.columns = ['col_tfidf_{}'.format(i) for i in news_transformed.columns]
    
    #pca = PCA()
    #news_score = pca.fit_transform(news_transformed)

    return news_transformed

def prepare_combined_dataset(stock_diff, news_score, target_col='binary_target', secondary_target_col='target'):
    '''
    Combine features - headlines + target - stock % return
    Pick how many dates to get headlines from and how to combined them
    Input: stock % returns, news vectors
    Output: #rows = #target dates (at a certain frequency), #cols = #features
    TODO: NEED TO CLEAN UP SO IT USES FREQUENCY
    '''
    #TODO - join on date
    stock_diff = stock_diff.copy()
    stock_diff.reset_index(inplace=True)
    stock_diff['Date_dt'] = pd.to_datetime(stock_diff['Date'])
    stock_diff['Date_dt_prev'] = stock_diff['Date_dt'].apply(lambda x: x - pd.DateOffset(days=1))
    stock_diff['Date_dt_prev_prev'] = stock_diff['Date_dt'].apply(lambda x: x - pd.DateOffset(days=2))
    stock_diff['Date_dt_prev_prev_prev'] = stock_diff['Date_dt'].apply(lambda x: x - pd.DateOffset(days=3))
    stock_diff['Date_dt_prev_prev_prev_prev'] = stock_diff['Date_dt'].apply(lambda x: x - pd.DateOffset(days=4))
    stock_diff.drop('Date', axis=1)

    news_score = news_score.copy()
    news_score.columns = [i + "_prev" for i in news_score.columns]
    news_score.reset_index(inplace=True)
    news_score['Date_dt'] = pd.to_datetime(news_score['Date'])

    #prev
    data = pd.merge(stock_diff, news_score, how='left', left_on='Date_dt_prev', right_on='Date_dt')

    #prev_prev
    #news_score.columns = [i + "_prev" if i.find("prev") > -1 else i for i in news_score.columns]
    #data = pd.merge(data, news_score, how='left', left_on='Date_dt_prev_prev', right_on='Date_dt')

    #prev_prev_prev
    #news_score.columns = [i + "_prev" if i.find("prev") > -1 else i for i in news_score.columns]
    #data = pd.merge(data, news_score, how='left', left_on='Date_dt_prev_prev_prev', right_on='Date_dt')

    #prev_prev_prev_prev
    #news_score.columns = [i + "_prev" if i.find("prev") > -1 else i for i in news_score.columns]
    #data = pd.merge(data, news_score, how='left', left_on='Date_dt_prev_prev_prev_prev', right_on='Date_dt')

    #keep headlines and target
    keep_cols = [i for i in data.columns if i.find("col")>-1] + [target_col, secondary_target_col]
            
    data = data[keep_cols]
    data = data.dropna()

    data_secondary_target = data[secondary_target_col]
    data.drop(secondary_target_col, axis=1, inplace=True)
    
    return data, data_secondary_target

def train_tfidf(document_list, max_features = None, use_stop_words=True):
    if use_stop_words:
        stop_words = stopwords.words('english')
    else:
        stop_words = None

    transformer = TfidfVectorizer(max_features = max_features, stop_words = stop_words)
    train_transform = transformer.fit_transform(document_list)
    
    return transformer
    
def create_train_validation_test(data, train_perc = 0.70, validation_perc = 0.20):
    '''
    Assumption: data is already sorted by "Date" in ascending order
    Each row consists of feature and target

    Since time-series, don't do random shuffling
    '''

    N_rows = float(data.shape[0])

    train_low = 0
    train_high = int(train_perc * N_rows)

    validation_low = train_high
    validation_high = validation_low + int(validation_perc * N_rows)

    test_low = validation_high
    test_high = int(N_rows)

    print train_low, train_high
    print validation_low, validation_high
    print test_low, test_high
    
    data_train = data.iloc[train_low:train_high]
    data_validation = data.iloc[validation_low:validation_high]
    data_test = data.iloc[test_low:test_high]

    return data_train, data_validation, data_test

def grid_search(data_train, data_validation, model, param_grid, score_func=recall_score, normalize=False):
    '''
    TODO: take target as argument
    '''
    def scorer(model, features, target, score_func=score_func):
        '''
        scorer implementation for grid search
        '''
        return score_func(target, model.predict(features))

    grid_search = GridSearchCV(model, param_grid, scoring = scorer, cv=3)

    grid_search.fit(data_train.drop('binary_target', axis=1), data_train['binary_target'])

    val_pred = grid_search.predict(data_validation.drop('binary_target', axis=1))

    print score_func(data_validation['binary_target'], val_pred)
    
    '''Precision-recall curve for best estimator'''
    target_true = data_validation['binary_target']
    target_proba_pred = grid_search.predict_proba(data_validation.drop('binary_target', axis=1))[:,0]
    precision, recall, thresholds = precision_recall_curve(target_true, target_proba_pred)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve for model = {}\n best params = {}".format(model.__class__.__name__, grid_search.best_params_))
    plt.savefig("plots/precision_recall_{}.png".format(model.__class__.__name__))

    results = {model.__class__.__name__: {'precision': precision,
                                          'recall': recall,
                                          'thresholds': thresholds}}

    return grid_search, results
    
def plot_precision_recall(data_train, data_test, score_func, feature_tag):
    gs_logreg, r_logreg = grid_search(data_train, data_test, LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [10**-5, 10**-1, 10, 10**2]}, score_func)

    gs_rf, r_rf = grid_search(data_train, data_test, RandomForestClassifier(), {'n_estimators': [10,30,60,90,150], 'max_depth': [5,10,15,20,30]}, score_func)
    
    gs_gbm, r_gbm = grid_search(data_train, data_test, GradientBoostingClassifier(), {'n_estimators': [10,30,60,90,150], 'max_depth': [5,10,15,20,30], 'loss': ['deviance', 'exponential']}, score_func)
    
    r = [r_logreg, r_rf, r_gbm]
    gs = [gs_logreg, gs_rf, gs_gbm]

    plt.figure()
    for result in r:
        name = result.keys()[0]
        p = result[name]['precision']
        r = result[name]['recall']

        plt.plot(r, p, label = name)
        plt.xlim([0,1])
        plt.ylim([0,1])
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    

    #plt.title("Precision-recall curve for model = {}\n best params = {}".format(model.__class__.__name__, grid_search.best_params_))
    plt.savefig("plots/precision_recall_curve_{}.png".format(feature_tag))
    return gs

def train_model(data_train, data_validation, data_test, target, model, score_func, predict_func, normalize=False):
    train_target = data_train[target]
    validation_target = data_validation[target]
    test_target = data_test[target]

    train_features = data_train.drop(target, axis=1)
    validation_features = data_validation.drop(target, axis=1)
    test_features = data_test.drop(target, axis=1)

    if normalize:
        train_mean = train_features.mean(axis=0)
        train_std = train_features.std(axis=0)

        train_features = (train_features - train_mean) / train_std
        validation_features = (validation_features - train_mean) / train_std
        test_features = (test_features - train_mean) / train_std

    overall_features = pd.concat([train_features, validation_features, test_features], axis=0) #do some feature engineering here - return what the model runs on
    overall_target = pd.concat([train_target, validation_target, test_target], axis=0)

    model.fit(train_features, train_target)
    
    train_pred = predict_func(train_features)
    validation_pred = predict_func(validation_features)
    test_pred = predict_func(test_features)
    
    if len(train_pred.shape) == 2:
        train_pred = [i[1] for i in predict_func(train_features)]
        validation_pred = [i[1] for i in predict_func(validation_features)]
        test_pred = [i[1] for i in predict_func(test_features)]
        
    train_score = score_func(train_target, train_pred)
    validation_score = score_func(validation_target, validation_pred)
    test_score = score_func(test_target, test_pred)
    
    print "TRAIN SCORE = {}".format(train_score)
    print "VALID SCORE = {}".format(validation_score)
    print "TEST  SCORE = {}".format(test_score)
    
    return model, overall_features

def simulation(capital, transaction_fee_rate, model, data_features, data_target, data_secondary_target, prob_tolerance):
    '''
    Simple simulation:
    if predict market is going to move up: buy fixed amount
    if predict market is going to move down: short fixed amount
    if can't predict for sure: do nothing

    always forced to act next time
    
    '''
    #amount_to_trade = 1000000
    #capital = 1000000
    amount_to_trade = capital

    #keep track of these variables
    amount_ts = [] #total capital at each time-step
    pnl_ts = [] #profit at each time-step
    cost_ts = [] #cost at each time step
    active_return_ts = [] #return of portfolio at each time-step
    benchmark_return_ts = [] #return of DJIA at each time-step

    data_features_np = np.array(data_features)
    pred = [i[1] for i in model.predict_proba(data_features_np)] #probability of predicting binary_target = 1
    actual = data_target.tolist()
    actual_return = data_secondary_target.tolist()
    
    for i in xrange(len(pred)):
        p = pred[i]

        if p < prob_tolerance:
            p = -1
        elif p > 1 - prob_tolerance:
            p = +1
        else:
            continue

        profit = p * (actual_return[i]) * amount_to_trade
        #profit = -actual_return[i] * amount_to_trade #make loss each time
        pnl_ts.append(profit)

        cost = transaction_fee_rate * amount_to_trade 
        cost_ts.append(cost)
        
        amount_to_trade += profit
        amount_ts.append(amount_to_trade)

        #only differ by signs
        active_return_ts.append(p*actual_return[i]) 
        benchmark_return_ts.append(actual_return[i])

        print i, pred[i], p, actual[i], actual_return[i], profit, amount_to_trade
        
    overall_return = 100 * (amount_ts[-1] - amount_ts[0]) / amount_ts[0]
    print "Overall return: {}".format(overall_return)

    overall_return_minus_cost = 100 * (amount_ts[-1] - np.sum(cost_ts) - amount_ts[0]) / amount_ts[0]
    print "Overall return minus cost: {}".format(overall_return_minus_cost)
    
    information_ratio = np.mean(np.array(active_return_ts) - np.array(benchmark_return_ts)) / np.std(active_return_ts)
    print "Information ratio: {}".format(information_ratio)

    information_ratio_abs = np.mean(np.array(active_return_ts) - np.array(np.abs(benchmark_return_ts))) / np.std(active_return_ts)
    print "Information ratio abs: {}".format(information_ratio_abs)

    return amount_ts, pnl_ts, cost_ts

def run_example(new_type='vader', max_tfidf_features=100):
    stock, news, combined = read_data()

    #stock data
    stock, stock_prev, stock_curr, stock_diff = prepare_stock_data(stock, frequency_in_days=frequency_in_days)

    #news data
    news_score_vader = prepare_news_data_vader(news) #one vector of length 25 for each day
    news_score_tfidf = prepare_news_data_tfidf(news, max_features = max_tfidf_features) #one vector of length max_tfidf_features for each day

    #combine news data
    if news_type=='vader':
        news_score = news_score_vader
    elif news_type=='tfidf':
        news_score = news_score_tfidf
    elif news_type=='combined':
        news_score = pd.concat([news_score_vader, news_score_tfidf], axis=1)     
    else:
        print "Please enter valid news type. Exiting..."

    #combined data
    data, secondary_target = prepare_combined_dataset(stock_diff, news_score)

    #validation
    data_train, data_val, data_test = create_train_validation_test(data, train_perc = 0.30, validation_perc = 0.30)

    #train - wasteful dual training but ok for now
    score_func = precision_score
    feature_tag = 'vader_precision'
    gs,r = grid_search(data_train, data_test, GradientBoostingClassifier(), {'n_estimators': [10,30,60,90,150], 'max_depth': [5,10,15,20,30], 'loss': ['deviance', 'exponential']}, score_func) #grid search over gradient boosted trees
 
    gs_models = plot_precision_recall(data_train, data_test, score_func, feature_tag)

    #run simulation with model
    capital = 1000000
    transaction_fee = 0.0005
    model = gs[0] #grid searched logistic regression
    tolerance = 0.45
    amount, pnl, cost = simulation(capital, transaction_fee, model, data.drop('binary_target', axis=1), data['binary_target'], secondary_target, tolerance)
    
    return model, data_train
