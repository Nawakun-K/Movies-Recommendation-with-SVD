from loguru import logger

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

class SVDModel():
    def __init__(self, k: int=100) -> None:
        self.k = k
    
    def fit(self, df: pd.DataFrame):
        self.actual_df = df
        df.fillna(0, inplace=True)        

        # Apply SVD into actual DataFrame
        U, sigma, Vt = svds(df, k=self.k)
        sigma_diag_matrix = np.diag(sigma)

        all_user_predicted_score = np.dot(np.dot(U, sigma_diag_matrix), Vt)
        preds_df = pd.DataFrame(all_user_predicted_score, columns = df.columns, index=df.index)

        self.preds_df = preds_df
    
    def _remove_watched(self, id:int):

        # Get actual watched
        watched_df = self.actual_df[self.actual_df.index == id]
        watched_df = watched_df.melt(value_vars=None, var_name='Movie_Id', value_name='Rating', ignore_index = False)
        watched_df = watched_df[watched_df['Rating'] > 0]

        # Get predicted
        preds_df = self.preds_df[self.preds_df.index == id]
        preds_df = preds_df.melt(value_vars=None, var_name='Movie_Id', value_name='Estimate_Score', ignore_index = False)

        # Actual result
        preds_df = preds_df[~preds_df['Movie_Id'].isin(watched_df['Movie_Id'])]
        self.pred_result = preds_df
    
    def predict(self, id:int) -> pd.DataFrame:

        self._remove_watched(id=id)
        return self.pred_result