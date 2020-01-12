# -*- coding: utf-8 -*-
"""
Beta Version
@author: ayh9kim
"""
# libraries
import os; os.chdir(r'D:\Doc\Project\MLProject\WebDataHarvest')
import re, sys
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns

# modules
from technical_analysis import technical_analysis
import vix_data

### VIX Data
request_headers = {"User-Agent": "Test"} 
request_endpoints = ["http://vixcentral.com/ajax_get_contango_data/"]

resp = vix_data.request_url(request_endpoints, request_headers)
data = vix_data.vix_data_parser(resp)

# download
# data.to_csv('data_VIX_' + datetime.today().strftime('%Y%m%d') + '.csv')

### Indicators
TA = technical_analysis(data.copy())
cols = data.columns[1:]

for col in cols:
    # ewma
    TA.get_ewma(5, col)
    TA.get_ewma(10, col)
    TA.get_ewma(15, col)
    TA.get_ewma(25, col)
    TA.get_ewma(30, col)
    TA.get_ewma(40, col)
    TA.get_ewma(50, col)    
    # rsi
    TA.get_rsi(14, col)
    # bb
    TA.get_bollinger_band(20, 2, col)
    TA.get_bollinger_band(20, 3, col)
    TA.get_bollinger_band(20, 4, col)
    # macd
    TA.get_macd(10, 26, 9, col)
   
### END Indicators

# set df as our main dataframe
df = TA.df.copy()

### Data Cleaning
df.dtypes
df.info()

df['Date'] = pd.to_datetime(df['Date'])

# missing values
df.dropna(how='any', inplace=True)

# sort by date
df.sort_values(by=['Date'], inplace=True)
### Data Cleaning End

### Feature Extraction
## historical log-return: if skewed +ly, we can use attenuation band to scale down outliers
nPeriod = 10
df['LogRet_' + str(nPeriod) + 'D'] = np.log(df['Spot']) - np.log(df['Spot'].shift(nPeriod))
## historical log-return END

## quantiles
# spot level
_, initial_bins = pd.qcut(df['Spot'], 5, retbins=True)
spotBins = np.array([12.5, 14, 16, 19]) # approximated by rounded initial_bins
for i_bin in spotBins:
    df['Spot_Bin_' + str(i_bin)] = (df['Spot'] > i_bin)*1

# contango F1-F2
_, initial_bins = pd.qcut(df['F1-F2 Contango'], 5, retbins=True)
contangoBins = np.array([1.8, 5.7, 8.4, 11.4]) # approximated by rounded initial_bins
for i_bin in contangoBins:
    df['ContangoF1F2_Bin_' + str(i_bin)] = (df['F1-F2 Contango'] > i_bin)*1

# contango F4-F7
_, initial_bins = pd.qcut(df['F4-F7 Contango'], 5, retbins=True)
contangoBins = np.array([2.7, 5.1, 7.5, 10.3]) # approximated by rounded initial_bins
for i_bin in contangoBins:
    df['ContangoF4F7_Bin_' + str(i_bin)] = (df['F4-F7 Contango'] > i_bin)*1

# rsi
rsiBins = np.array([30, 50, 70]) # approximated by rounded initial_bins
rsiColumns = df.columns[df.columns.str.contains('_rsi_')]
for i_col in rsiColumns:
    for i_bin in rsiBins:
        df[i_col + '_Bin_' + str(i_bin)] = (df[i_col] > i_bin)*1
## quantiles END

## TS cross-overs: Spot, F1, F2, F4, F7, Contangos
lstTS = ['Spot', 'F1', 'F2', 'F4', 'F7', 'F1-F2 Contango', 'F4-F7 Contango']
for ts in lstTS:
    lstCols = df.columns[(df.columns.str.contains(ts + '_ewma')) & ~(df.columns.str.contains(ts + '_ewma_26'))]
    for col in lstCols:
        df[ts + '_GE_' + col] = (df[ts] > df[col])*1
## TS cross-overs END
        
## TA BB Breach
lstTS = ['Spot', 'F1', 'F2', 'F4', 'F7', 'F1-F2 Contango', 'F4-F7 Contango']
for ts in lstTS:
    df[ts + '_bb_ub_breach'] = (df[ts] > df[ts + '_bb_ub'])*1
    df[ts + '_bb_lb_breach'] = (df[ts] > df[ts + '_bb_lb'])*1      
## TA BB Breach END
    
### Defining Target Variable: Classification Problem - 2 Week Return > 30% when Spot < 16 
### This is an imbalanced classification problem, trying to predict the blow up in volatility

### The objective of this analysis is to see if it's possible to use a model to evaluate the probability
### of trader's VIX long call option trades being profitable, which can be for hedging or purely speculative.
nPeriod = 10
tmp_return = df['Spot'] > 20
tmp_return = tmp_return[np.where(~np.isnan(tmp_return))]
tmp_name = 'fwd_' + str(nPeriod) + 'd_return'
df[tmp_name] = np.nan
df[tmp_name].iloc[:len(tmp_return)] = tmp_return

# spot > x nPeriod later and spot < y now
minReturn = .30
maxSpot = 16
dfTarget = df[(df[tmp_name] > minReturn) & (df['Spot'] < maxSpot)][['Date', 'Spot', tmp_name]]
dfTarget.reset_index(inplace=True, drop=True)

idxNull = np.where(df[tmp_name].isnull())
df['Target'] = ((df[tmp_name] > minReturn) & (df['Spot'] < maxSpot))*1
df['Target'].iloc[idxNull] = np.nan

# delete
df.drop(columns=[tmp_name], axis=1, inplace=True)

# graph VIX and target
sns.set_style('whitegrid')
ax = sns.lineplot(data=df, x='Date', y='Spot')
target_date = df.loc[df['Target']==1, 'Date'].values

for xc in target_date:
    ax.axvline(x=xc, color='k', linestyle='-')

# drop NA columns
dfRest = df[df.isnull().any(axis=1)]
dfClean = df.dropna(how='any')
### END Feature Extraction

### Modelling
import matplotlib.pyplot as plt; import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight

# X, y
dfDate = dfClean['Date']
dfY = dfClean['Target']
dfX = dfClean.copy().drop(columns=['Target', 'Date'])

# TS Split
nSplit = 3
cvSplit = TimeSeriesSplit(n_splits=nSplit)

# Class Weights
classWeights = compute_class_weight('balanced', np.unique(dfY), dfY.values)
print('The weights are: %s' % classWeights + ' respectively for the classes: %s' % np.unique(dfY))

# Define Models
nMinSample = 1250
modelRF = RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample')
modelXGB = xgb.XGBClassifier(learning_rate=0.1, sample_weight=classWeights, nthread=-2)
#modelLogit_L1 = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear')
modelLogit_L2 = LogisticRegression(penalty='l2', class_weight='balanced', solver='newton-cg')
#modelLogit_EL = LogisticRegression(penalty='elasticnet', class_weight='balanced', solver='saga', l1_ratio=0.5, max_iter=10000)

dfResult = pd.DataFrame(index=dfX.index[(nMinSample+1):], 
                        columns=['RF_Pred', 'RF_Prob', 
                                 'XGB_Pred', 'XGB_Prob',
                                 'LogL1_Pred', 'LogL1_Prob',
                                 'LogL2_Pred', 'LogL2_Prob',
                                 'LogEL_Pred', 'LogEL_Prob', 'Actual'])
dfResult.fillna(2, inplace=True)
dfResult.reset_index(drop=True, inplace=True)

for i in range(nMinSample, dfX.shape[0]-1):
    # X, y
    tmpX, tmpY = dfX.iloc[:i,], dfY.iloc[:i,]
    tmpXTest, tmpYTest = pd.DataFrame(dfX.iloc[i+1,]).T, dfY.iloc[i+1,]
    
    # fit
    modelRF.fit(tmpX, tmpY)
    modelXGB.fit(tmpX, tmpY)    
    #modelLogit_L1.fit(tmpX, tmpY)    
    modelLogit_L2.fit(tmpX, tmpY)
    #modelLogit_EL.fit(tmpX, tmpY)
    
    # prediction
    tmpPred_RF = modelRF.predict(tmpXTest)[0]
    tmpPred_XGB = modelXGB.predict(tmpXTest)[0]
    #tmpPred_L1 = modelLogit_L1.predict(tmpXTest)[0]
    tmpPred_L2 = modelLogit_L2.predict(tmpXTest)[0]
    #tmpPred_EL = modelLogit_EL.predict(tmpXTest)[0]
    
    # prob == 0
    tmpProb_RF = modelRF.predict_proba(tmpXTest)[0][0]
    tmpProb_XGB = modelXGB.predict_proba(tmpXTest)[0][0]
    #tmpProb_L1 = modelLogit_L1.predict_proba(tmpXTest)[0][0]
    tmpProb_L2 = modelLogit_L2.predict_proba(tmpXTest)[0][0]
    #tmpProb_EL = modelLogit_EL.predict_proba(tmpXTest)[0][0]

    # save
    tmpIdx = i - nMinSample
    dfResult.loc[tmpIdx, 'RF_Pred'] = tmpPred_RF
    dfResult.loc[tmpIdx, 'RF_Prob'] = tmpProb_RF
    dfResult.loc[tmpIdx, 'XGB_Pred'] = tmpPred_XGB
    dfResult.loc[tmpIdx, 'XGB_Prob'] = tmpProb_XGB
    #dfResult.loc[tmpIdx, 'LogL1_Pred'] = tmpPred_L1
    #dfResult.loc[tmpIdx, 'LogL1_Prob'] = tmpProb_L1
    dfResult.loc[tmpIdx, 'LogL2_Pred'] = tmpPred_L2
    dfResult.loc[tmpIdx, 'LogL2_Prob'] = tmpProb_L2
    #dfResult.loc[tmpIdx, 'LogEL_Pred'] = tmpPred_EL
    #dfResult.loc[tmpIdx, 'LogEL_Prob'] = tmpProb_EL
    dfResult.loc[tmpIdx, 'Actual'] = tmpPred_XGB
    
    # notify
    print('\n\n\nFinished %s' % (tmpIdx+1) + ' backtest!')
    
# delete this line below if not needed    
#dfResult.drop(index=[1329], inplace=True)

dfResult.to_csv('model_output.csv')

dfResult = pd.read_csv('model_output.csv')

### Performance
colProb = 'XGB_Prob'
prob_threshold = 0.6
y = dfResult['Actual']
pred = dfResult[colProb].apply(lambda x: 0 if x > prob_threshold else 1)
prob = 1-dfResult[colProb]

## confusion matrix
matConf = confusion_matrix(y, pred) 
#'ha': 'center', 'va': 'center'
ax = sns.heatmap(matConf, annot=True, annot_kws={'size': 20, 'ha': 'center', 'va': 'center'}, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix: VIX')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.show()

# accuracy & error
accuracy = (matConf[0][0] + matConf[1][1])/matConf.sum()
error = 1 - accuracy
print('The accuracy is: %0.2f' % accuracy + ' and the error is: %0.2f' % error)

# classification report
target_names = ['VIX Stay', 'VIX Long']
print('\n\n' + classification_report(y, pred, target_names=target_names)) 

## ROC/AUC
# baseline
baseProb = [classWeights[0]/classWeights.sum() for _ in range(len(prob))]
baseFPR, baseTPR, baseThresholds = roc_curve(y, baseProb)
baseAUCScore = roc_auc_score(y, baseProb)
 
   
colProbList = ['XGB_Prob', 'RF_Prob', 'LogL2_Prob']
sns.set_style('whitegrid')
sns.lineplot(x=baseFPR, y=baseTPR, linestyle='--', label='Base')
for col in colProbList:
    # select 
    tmpProb = 1-dfResult[col]
    # fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y, tmpProb)
    # AUC score
    modelAUCScore = roc_auc_score(y, prob)
    print('Model ' + col + ' has AUC score of: %.3f' % modelAUCScore)
    # plot
    legendName = col[:(re.search('_', col).start())]    
    sns.lineplot(x=fpr, y=tpr, marker='.', label=legendName)
# plot format
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

## Precision-Recall Curve
baseLine = len(y[y==1]) / len(y)

colProbList = ['XGB_Prob', 'RF_Prob', 'LogL2_Prob']
sns.set_style('whitegrid')
sns.lineplot(x=[0, 1], y=[baseLine, baseLine], linestyle='--', label='Baseline')
for col in colProbList:
    # select
    tmpProb = 1-dfResult[col]
    tmpPred = dfResult[col].apply(lambda x: 0 if x > prob_threshold else 1)
    # prec, recall
    modelPrecision, modelRecall, _ = precision_recall_curve(y, tmpProb)
    # plot
    legendName = col[:(re.search('_', col).start())]
    sns.lineplot(modelRecall, modelPrecision, marker='.', label=legendName)
    # F1 Score
    modelF1 = f1_score(y, tmpPred)
    print(legendName + ' f1 score is: %.3f' % (modelF1))
# plot format
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
### END Performance 

# important features
# xgb
xgb.plot_importance(modelXGB, max_num_features=30)

# rf
importance = modelRF.feature_importances_
indices = np.argsort(importance)[::-1][:30]

sns.barplot(x=[i for i in range(len(indices))], y=importance[indices], color='red')
feature_names = dfX.columns # e.g. ['A', 'B', 'C', 'D', 'E']
plt.xticks(range(len(indices)), feature_names, rotation=90)
### END Modelling

### Application
# future projection
dfForecast = dfRest.copy()
forecast = modelXGB.predict(df.iloc[:, 1:-1])
dfForecast['Target'] = forecast

# plot
dfForecast = pd.concat([dfClean, dfForecast], axis=0)
target_date = dfForecast.loc[dfForecast['Target']==1, 'Date'].values

sns.set_style('whitegrid')
sns.lineplot(data=dfForecast, x='Date', y='Spot')

for xc in target_date:
    ax.axvline(x=xc, color='k', linestyle='-')
### END Application