import glob
from loguru import logger

import pandas as pd
import numpy as np

class LoadNetflixData():
    def __init__(self, path: str='./assets', demo: bool=True):
        self.filepaths = glob.glob(f'{path}/combined_data*.txt')
        self.demo = demo
        
    
    def _data_cleaning(self) -> pd.DataFrame:
        '''
        Add Movie_Id column
        '''
        df_nan = pd.DataFrame(pd.isnull(self.df.Rating))
        df_nan = df_nan[df_nan['Rating'] == True]
        df_nan = df_nan.reset_index()

        movie_np = []
        movie_id = 1

        for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
            # numpy approach
            temp = np.full((1,i-j-1), movie_id)
            movie_np = np.append(movie_np, temp)
            movie_id += 1

        # Account for last record and corresponding length
        last_record = np.full((1, len(self.df) - df_nan.iloc[-1, 0] - 1), movie_id)
        movie_np = np.append(movie_np, last_record)

        # remove those Movie ID rows
        df = self.df[pd.notnull(self.df['Rating'])]

        df['Movie_Id'] = movie_np.astype(int)
        df['Cust_Id'] = df['Cust_Id'].astype(int)

        return df
    
    def _data_slicing(self, df) -> pd.DataFrame:
        '''
        Reduce data size
        '''
        f = ['count','mean']

        df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
        df_movie_summary.index = df_movie_summary.index.map(int)
        movie_benchmark = round(df_movie_summary['count'].quantile(0.7),0)
        drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

        df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
        df_cust_summary.index = df_cust_summary.index.map(int)
        cust_benchmark = round(df_cust_summary['count'].quantile(0.7),0)
        drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

        df = df[~df['Movie_Id'].isin(drop_movie_list)]
        df = df[~df['Cust_Id'].isin(drop_cust_list)]

        return df

    def load(self) -> pd.DataFrame:
        '''
        Load Netflix data, with data cleaning and slicing, into DataFrame
        '''
        if self.demo == True:
            self.df = pd.read_csv(self.filepaths[0], header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
            self.df['Rating'] = self.df['Rating'].astype(float)
        else:
            self.df = pd.concat([pd.read_csv(f, header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1]) for f in self.filepaths])
            self.df['Rating'] = self.df['Rating'].astype(float)
            self.df.index = np.arange(0,len(self.df))
        
        df = self._data_cleaning()
        df = self._data_slicing(df)

        df = df.pivot_table(index=['Cust_Id'], columns=['Movie_Id'], values='Rating')

        return df