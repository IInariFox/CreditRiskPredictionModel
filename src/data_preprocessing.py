import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath, header=1)
    # Remove any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print("Columns in the DataFrame:", df.columns.tolist())
    return df

def preprocess_data(df):
    # Rename target column
    df = df.rename(columns={'Y': 'default'})
    # Convert data types
    for col in df.columns:
        if col != 'default':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['default'] = pd.to_numeric(df['default'], errors='coerce').astype(int)
    # Handle missing values
    df = df.dropna()
    # Drop irrelevant columns if any
    X = df.drop(['default'], axis=1)
    y = df['default']
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data('data/credit_data.csv')
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)



