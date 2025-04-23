import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def preprocess_data(df, numerical_features, categorical_features):
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    for col in numerical_features:
        df[col] = num_imputer.fit_transform(df[[col]])

    for col in categorical_features:
        df[col] = cat_imputer.fit_transform(df[[col]])

    # Example feature engineering
    df['debt_to_income'] = df['credit card debt'] / (df['annual Salary'] + 1e-6)
    df['debt_to_networth'] = df['credit card debt'] / (df['net worth'] + 1e-6)

    numerical_features += ['debt_to_income', 'debt_to_networth']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    X = df.drop('car purchase amount', axis=1)
    y = df['car purchase amount']

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor
