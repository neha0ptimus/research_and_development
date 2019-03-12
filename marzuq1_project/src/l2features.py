import re
import time
import pickle
import os
import numpy as np
import pandas as pd
from itertools import *
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from textwrap import dedent
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import datetime
import math
from plotnine import *
from sklearn.metrics import brier_score_loss
from sklearn.utils import column_or_1d


from sklearn.base import TransformerMixin
# Imputes missing values
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if ( (X[c].dtype == np.dtype('O')) | (X[c].dtype.name =='category')) else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)
    

def get_vh_60(data, election_date):
    ''' 
    Before doing this change target election col to y 
    Currently, only works for general elections
    '''
    vh_cols_all = [x for x in data.columns.values.tolist() if x.startswith('vh_')]
    
    vh_cols_want_old_names = [x for x in data.columns.values.tolist() 
           if (
               (x.startswith('vh_') and int(x[3:7]) < election_date['year'] and not (re.search('democratic',x) or re.search('republican',x) or re.search('other',x) ) )
               or (x.startswith('vh_') and int(x[3:7]) == election_date['year'] and x[7] != 'g' and not (re.search('democratic',x) or re.search('republican',x) or re.search('other',x) ))
           )][0:60]
    
    vh_cols_drop = list(set(vh_cols_all) - set(vh_cols_want_old_names))
    
    data = data.drop(vh_cols_drop, axis=1)    
    
    vh_cols_want_new_names = ['_'.join(['vh',str(x)]) for x in range(1,61)]
    
    data.rename(columns=dict(zip(vh_cols_want_old_names, vh_cols_want_new_names)),inplace=True)
    
    return data


def get_percent_turnout(data, geo_unit, election_date):
    
    col = [x for x in data.columns.values.tolist() 
           if (x.startswith('vh_') and int(x[3:7]) == election_date['year'] - 4 and x[7] == 'g')][0]
    t1 = data.groupby(geo_unit)[col].sum()/data.groupby(geo_unit)[col].count()
    new_col_name = '_'.join([geo_unit,'percent_turnout_4g'])
    t2 = pd.DataFrame(columns=[geo_unit, new_col_name])
    t2[geo_unit] = t1.index
    t2[new_col_name] = t1.values
    data = data.merge(t2, how='left', on=geo_unit)
    
    return data

def normalize_population(data, col):
    '''
    Feature engineered to estimate relative population density as per col e.g reg_zip
    density = population in a zip code/ total population in all zip codes
    Returns the data file with a new column norm_population_by_col
    '''
    new_col_name = '_'.join(['norm_population','by',col])
    t1 = (data.groupby(col)[col].count())/data[col].dropna().count()
    t2 = pd.DataFrame(columns=[col, new_col_name])
    t2[col] = t1.index
    t2[new_col_name] = t1.values
    data = data.merge(t2, how='left', on=col)
    return data


def date_to_nums(data, col, from_date):
    '''
    Feature engineered to estimate number years since from_date of a datetime col
    Returns the data file with a new column norm_population_by_col
    '''
    from datetime import date
    new_col_name = '_'.join([col.replace('_date',''),'years'])
    data[new_col_name] = (((date(from_date['year'],from_date['month'],from_date['day']) - data[col]).dt.days).values)/365
    
    return data


def get_max_index(values): 
    '''
    Identify the zip/precinct by majority party or ethnicity voters
    '''    
    # 
    return values.value_counts().index[0],(values.value_counts().values[0])/len(values) 
    
def get_proportion_not_white(values):    
    if not 'White' in values.value_counts().index:
        d = 1.0
        return d
    
    d = 1- (values[values == 'White'].value_counts()/values.value_counts().sum())
    if d.empty:
        d = 1.0
    return  d 

def get_proportion_not_republican(values):    
    if not 'Republican' in values.value_counts().index:
        d = 1.0
        return d
    
    d = 1- (values[values == 'Republican'].value_counts()/values.value_counts().sum())
    if d.empty:
        d = 1.0
    return  d 
    
def proportion_or_max_index(data, geo_unit, feature):
    '''
    Find the majority or not_our_base in a geographical unit by a certain feature
    So, it could be find proportion of non-republicans in a precinct
    or which is the most occuring income group in a dma
    '''
        
    if feature == 'party':
        t1 = data.groupby([geo_unit])[feature].agg(get_proportion_not_republican)
        new_col_name = '_'.join([geo_unit,'not_republican'])
        
    elif feature == 'ethnicity':
        t1 = data.groupby([geo_unit])[feature].agg(get_proportion_not_white)
        new_col_name = '_'.join([geo_unit, 'not_white'])
        
    else:
        t1 = data.groupby([geo_unit])[feature].agg(get_max_index)
        new_col_name = '_'.join([geo_unit, feature, 'majority'])
     
    t2 = pd.DataFrame(columns=[geo_unit, new_col_name])
    t2[geo_unit] = t1.index
    t2[new_col_name] = t1.values
    data = data.merge(t2, how='left', on=geo_unit)
    
    return data

def add_features_date(data, election_date):
    
    data = date_to_nums(data,'birth_date', election_date)
    
    data['birth_years_squared'] = data['birth_years']**2
    
    data['birth_years_cubed'] = data['birth_years']**3
    
    # convert age into a categorical variable by making age bins
    data['birth_years_group'] = pd.cut(data['birth_years'], [1, 18, 25,35, 55, 1000],
                                       labels=['-18','18-24','25-34','35-54','55+'])
    
    data = date_to_nums(data,'calculated_reg_date', election_date)
    
    data['calculated_reg_years_group'] = pd.cut(data['calculated_reg_years'], [-4,0,1,2,4,8,1000],
                                       labels=['-0','0-1','1-2','2-4','4-8','8+'])
    
    data = get_percent_turnout(data, 'precinct', election_date)
    
    return data    

def add_features(data):
       
    # by reg_zip
    data = normalize_population(data,'reg_zip')
    data = proportion_or_max_index(data,'reg_zip','party')
    data = proportion_or_max_index(data,'reg_zip','ethnicity')
    
    # by precinct
    data = normalize_population(data,'precinct')
    data = proportion_or_max_index(data,'precinct','party')
    data = proportion_or_max_index(data,'precinct','ethnicity')
    return data


def first_time_voters(data, flags, first_time_voter_flag):
    
    print('Total number of voters in the pre-election voterfile = ', data.shape[0])
    # subset first time voters
    data = data[data['lalvoterid'].isin(flags[flags[first_time_voter_flag]== True]['lalvoterid'])]
    print('Total number of First time voters in proxy election = ', data.shape[0])

    print('Filter first time voters file to exclude voters below the age of 18 by proxy election date')
    # exclude voters below the age of 18
    data = data[data['birth_years'] >= 18]
    print('First time voters above the age of 18 by proxy election date =', data.shape[0])

    return data
    