# -*- coding: utf-8 -*-
"""
@author: ayh9kim

Examples:
y = technical_analysis(data.copy())
y.get_ma(10, 'Adj Close')
y.get_ewma(12, 'Adj Close')
y.get_macd(10,26,9, 'Adj Close')
y.get_bollinger_band(21, 20, 'Adj Close')
y.get_rsi(14, 'Adj Close')
y.get_adx(14, 'Adj Close', 'High', 'Low')
y.get_slow_sto(14, 3, 3, 'Adj Close', 'High', 'Low')

"""

import sys
eps = sys.float_info.epsilon
import re
import datetime as dt
import numpy as np
import pandas as pd

def generate_weekly_data_daily(df=pd.DataFrame()):
    ## df: Date, Open, High, Low, Close
    
    ## Find the first full week
    # Monday indicator
    df['Monday'] = (df['Date'].dt.weekday == 0)*1 # 1 if Monday
    # First Monday
    idx_first_Monday = df['Monday'].idxmax()
    # Clear every record before the first Monday
    df.iloc[0:idx_first_Monday] = np.nan
    
    ## Week high/low are cumulative max/min of the daily high/low
    # group by year & week
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.week
    df_grp = df.groupby(['Year', 'Week'])
    # cummax/cummin for high/low
    df_weekly_high = pd.DataFrame(df_grp['High'].cummax())
    df_weekly_low = pd.DataFrame(df_grp['Low'].cummin())
    df_weekly_high_low = pd.concat([df_weekly_high, df_weekly_low], axis=1)
    df_weekly_high_low.columns = ['Weekly_High', 'Weekly_Low']
    
    ## Week open is the open value on Monday and does not change throughout the week
    df_weekly_open = df_grp['Open'].first().reset_index() # or just select the Monday == 1
    tmpCol = df_weekly_open.columns.tolist(); tmpCol[2] = 'Weekly_Open'
    df_weekly_open.columns = tmpCol
    df_weekly_open = pd.merge(df, df_weekly_open, how='left', on=['Year', 'Week'])
    df_weekly_open = df_weekly_open['Weekly_Open']
    
    ## Week close is the close value on each day (i.e. weekly close == daily close through out the week)
    df_weekly_close = pd.DataFrame(df['Close'])
    df_weekly_close.columns = ['Weekly_Close']

    ## Combine
    df_weekly = pd.concat([df['Date'], df_weekly_open, df_weekly_high_low, df_weekly_close], axis=1)
    return(df_weekly)

def generate_heinkin_ashi_data(df=pd.DataFrame()):
    # open = [open(t-1) + close(t-1)]/2
    heinkin_ashi_open = (df['Open'].shift(periods=1) + df['Close'].shift(periods=1))/2
    
    # close = [open + high + low + close]/4
    heinkin_ashi_close = (df['Open'] + df['High'] + df['Low'] + df['Close'])/4
    
    # high and low are the same
    df['Open'] = heinkin_ashi_open
    df['Close'] = heinkin_ashi_close
    
    # return
    return(df)
    
def generate_candle_stick_data(df=pd.DataFrame()):
    # check
    if df.shape[0] == 0:
        print("The dataframe is empty!")
        return()
    ## range
    # high vs. low
    df['Range_high_to_low'] = df['High'] - df['Low']
    # range vs. open
    df['Range_high_to_open'] = df['High'] - df['Open']
    df['Range_open_to_low'] = df['Open'] - df['Low']
    # range vs. close
    df['Range_high_to_close'] = df['High'] - df['Close']
    df['Range_close_to_low'] = df['Close'] - df['Low']
    # open vs. close
    df['Range_close_to_open'] = df['Close'] - df['Open']
    
    ## indicator
    # up/down intraday
    df['Up_intraday'] = (df['Close'] > df['Open'])*1
    # up/down previous day
    df['Up_prevday'] = (df['Close'] > df['Close'].shift(periods=1))*1
    df['Up_prevday'].iloc[0] = np.nan
    
    ## special pattern indicator
    reg_param_doji = 3 # need at least x times the tail length compared to the body
    # doji - open/close are nearby with high and low extended
    df['Doji'] = df['Up_intraday']*((df['Range_high_to_close'] > reg_param_doji*df['Range_close_to_open']) &
                                    (df['Range_open_to_low'] > reg_param_doji*df['Range_close_to_open'])) + \
                 (1-df['Up_intraday'])*((df['Range_high_to_open'] > reg_param_doji*df['Range_close_to_open'].abs()) &
                                        (df['Range_close_to_low'] > reg_param_doji*df['Range_close_to_open'].abs()))
    # star - opposite of hangman
    reg_param_star = 2
    df['Star'] = df['Up_intraday']*((df['Range_high_to_close'] > reg_param_star*df['Range_close_to_open']) &
                                    (reg_param_star*df['Range_open_to_low'] < df['Range_close_to_open'])) + \
                    (1-df['Up_intraday'])*((df['Range_high_to_open'] > reg_param_star*df['Range_close_to_open'].abs()) &
                                           (reg_param_star*df['Range_close_to_low'] < df['Range_close_to_open'].abs()))
    # hangman - either open/close is near high with a big gap between close/open and low
    reg_param_hang = 2
    df['Hangman'] = df['Up_intraday']*((df['Range_open_to_low'] > reg_param_hang*df['Range_close_to_open']) &
                                       (reg_param_hang*df['Range_high_to_close'] < df['Range_close_to_open'])) + \
                    (1-df['Up_intraday'])*((df['Range_close_to_low'] > reg_param_hang*df['Range_close_to_open'].abs()) &
                                           (reg_param_hang*df['Range_high_to_open'] < df['Range_close_to_open'].abs()))
    
    ## enhancements can include multi-day patterns such as bearish gulf                    

class technical_analysis():
    """This is a collection of technical analysis tools indicating: volatility, momentum, trend and volume"""
    def __init__(self, df):
        
        self.df = df
    
    def get_ma(self, lag, col_name):
        
        self.df[col_name+'_ma_'+str(lag)] = self.df[col_name].rolling(window=lag).mean()
        
    def get_ewma(self, lag, col_name):
        
        self.df[col_name+'_ewma_'+str(lag)] = self.df[col_name].ewm(span=lag).mean()
    
    def get_macd(self, lag_short=10, lag_long=26, lag_signal=9, col_name=''):
        
        self.get_ewma(lag_short, col_name)
        self.get_ewma(lag_long, col_name)
        
        tmp_macd_name = col_name+'_macd_'+str(lag_short)+'_'+str(lag_long)
        self.df[tmp_macd_name] = \
        self.df[col_name+'_ewma_'+str(lag_short)] - self.df[col_name+'_ewma_'+str(lag_long)]
        
        tmp_signal_name = tmp_macd_name + '_ewma_'+str(lag_signal)
        self.get_ewma(lag_signal, tmp_macd_name)
        
        tmp_sig_diff_name = col_name+'_macd_signal_'+str(lag_short)+'_'+str(lag_long)+'_'+str(lag_signal)
        self.df[tmp_sig_diff_name] = self.df[tmp_macd_name] - self.df[tmp_signal_name] 
        
    # enhancement: provide features in relative form (i.e. ub/close or lb/close))
    def get_bollinger_band(self, lag_ma, lag_std, col_name):
        
        tmp_sd = self.df[col_name].rolling(window=lag_std).std()
        self.get_ma(lag_ma, col_name)
        self.df[col_name+'_bb_ub'] = self.df[col_name+'_ma_'+str(lag_ma)] + tmp_sd*2
        self.df[col_name+'_bb_lb'] = self.df[col_name+'_ma_'+str(lag_ma)] - tmp_sd*2
    
    def get_rsi(self, lag, col_name):
        # get delta
        tmp_delta = self.df[col_name].diff()
        
        # up & down vectors
        tmp_up, tmp_dn = tmp_delta.copy(), tmp_delta.copy()
        tmp_up[tmp_up < 0] = 0; tmp_dn[tmp_dn > 0] = 0
        
        # rolling
        tmp_avg_gain = tmp_up.rolling(window=lag).mean()
        tmp_avg_loss = tmp_dn.rolling(window=lag).mean().abs()
        
        # relative strength
        tmp_RS = pd.Series([np.nan]*len(tmp_avg_gain))
        tmp_RS[lag] = (tmp_avg_gain[lag]+eps)/(tmp_avg_loss[lag]+eps)
        
        tmp_RS[(lag+1):] = (tmp_avg_gain[lag:-1].values*(lag-1) + tmp_avg_gain.values[(lag+1):] + eps) \
                           /(tmp_avg_loss[lag:-1].values*(lag-1) + tmp_avg_loss.values[(lag+1):] + eps)
        
        tmp_RS = 100.0 - (100.0/(1.0 + tmp_RS))
        
        self.df[col_name+'_rsi_'+str(lag)] = tmp_RS.values
    
    def get_true_range(self, col_close, col_high, col_low):
        
        tmp_high_to_close_t_1 = np.maximum(self.df[col_high].values[1:], self.df[col_close].values[:-1])
        tmp_low_to_close_t_1 = np.minimum(self.df[col_low].values[1:], self.df[col_close].values[:-1])
        
        tmp_true_range = tmp_high_to_close_t_1 - tmp_low_to_close_t_1
                
        self.df[col_close+'_tr'] = np.append([np.nan],tmp_true_range)
    
    def wilders_smooth(self, idx_start, lag, series):
        # init
        tmp_output = pd.Series([np.nan]*len(series))
        # first value
        tmp_output[idx_start] = series[:idx_start].mean()
        # subsequent value
        for i in range((idx_start+1), len(series)):
            tmp_output[i] = ((lag-1)*tmp_output[(i-1)] + series[i])/lag
        
        return tmp_output
    
    def get_adx(self, lag, col_close, col_high, col_low):
        # get true range
        self.get_true_range(col_close, col_high, col_low)
        
        # get average true range
        tmp_atr = self.wilders_smooth(lag, lag, self.df[col_close+'_tr'])
        
        # pos & neg dm
        tmp_high_delta = self.df[col_high].diff()
        tmp_low_delta = -self.df[col_low].diff()
  
        tmp_pos_dm, tmp_neg_dm = tmp_high_delta.copy(), tmp_low_delta.copy()
        tmp_pos_dm[~((tmp_pos_dm > tmp_low_delta) & (tmp_pos_dm > 0))] = 0
        tmp_neg_dm[~((tmp_neg_dm > tmp_high_delta) & (tmp_neg_dm > 0))] = 0
        
        # pos & neg dir index
        tmp_pos_di = 100*self.wilders_smooth(lag, lag, tmp_pos_dm).values / tmp_atr.values
        tmp_neg_di = 100*self.wilders_smooth(lag, lag, tmp_neg_dm).values / tmp_atr.values
        
        # adx
        tmp_dx = 100*np.abs(tmp_pos_di-tmp_neg_di)/(tmp_pos_di+tmp_neg_di)
        tmp_adx = self.wilders_smooth(2*lag-1, lag, pd.Series(tmp_dx))
        
        self.df[col_close+'_adx_'+str(lag)] = tmp_adx.values

    def get_fast_sto(self, lag_stoch, lag_sma, col_close, col_high, col_low):
        
        tmp_max_high = self.df[col_high].rolling(window=lag_stoch).max()
        tmp_min_low = self.df[col_low].rolling(window=lag_stoch).min()
        
        tmp_K = 100*(self.df[col_close].values - tmp_min_low.values)/(tmp_max_high.values - tmp_min_low.values)
        tmp_K = pd.Series(tmp_K)
        tmp_D = tmp_K.rolling(window=lag_sma).mean()        
        
        self.df[col_close+'_fast_sto_'+str(lag_stoch)] = tmp_K.values
        self.df[col_close+'_fast_sto_'+str(lag_stoch)+'_'+str(lag_sma)] = tmp_D.values
        
    def get_slow_sto(self, lag_stoch, lag_sma, lag_sma2, col_close, col_high, col_low):
        # make fast sto
        self.get_fast_sto(lag_stoch, lag_sma, col_close, col_high, col_low)
        # make slow sto
        tmp_slow_K = self.df[col_close+'_fast_sto_'+str(lag_stoch)+'_'+str(lag_sma)]   
        tmp_slow_D = tmp_slow_K.rolling(window=lag_sma2).mean()  
        
        self.df[col_close+'_slow_sto_'+str(lag_stoch)+'_'+str(lag_sma)] = tmp_slow_K.values
        self.df[col_close+'_slow_sto_'+str(lag_stoch)+'_'+str(lag_sma)+'_'+str(lag_sma2)] = tmp_slow_D.values    
    
    # parabolic SAR
    
    # ichimoku cloud
    
    # calculate weekly & monthly
    
    # volume indicators






















