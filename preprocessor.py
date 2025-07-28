# preprocessor.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FraudPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Frequency encode CUSTOMER_ID and TERMINAL_ID
        df['CUSTOMER_ID_FREQ'] = df['CUSTOMER_ID'].map(df['CUSTOMER_ID'].value_counts())
        df['TERMINAL_ID_FREQ'] = df['TERMINAL_ID'].map(df['TERMINAL_ID'].value_counts())

        # Drop original ID columns and TRANSACTION_ID
        df = df.drop(['TRANSACTION_ID', 'CUSTOMER_ID', 'TERMINAL_ID'], axis=1)

        # Convert TX_DATETIME to Features
        df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
        df['TX_DAY'] = df['TX_DATETIME'].dt.day
        df = df.drop('TX_DATETIME', axis=1)

        # Ensure all Columns are Numeric
        df['TX_TIME_SECONDS'] = df['TX_TIME_SECONDS'].astype(float)
        df['TX_TIME_DAYS'] = df['TX_TIME_DAYS'].astype(float)

        return df
