import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

def preprocess(df_original):
    df = df_original.dropna(subset=['mh_PHQ_S']).copy()
    df = df[df['age'] >= 15]

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols.remove('mh_PHQ_S')
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    imputer_num = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[numeric_cols])
    poly_feature_names = poly.get_feature_names_out(numeric_cols)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

    df = pd.concat([df, poly_df], axis=1)
    return df, numeric_cols, categorical_cols
