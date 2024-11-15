import pandas as pd

def load_data(file_path='data/credit_data.csv'):
    df = pd.read_csv(file_path)
    print(f"Columns in the DataFrame: {df.columns.tolist()}")
    return df

def preprocess_data(df, columns_to_keep=None):
    # Ensure the target column is correctly identified
    df['default'] = pd.to_numeric(df['Y'], errors='coerce').fillna(0).astype(int)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    categorical_columns = ['X2', 'X3', 'X4']
    df = pd.get_dummies(df, columns=categorical_columns)

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.to_numeric(df[column], errors='coerce')

    df = df.fillna(0)

    if columns_to_keep is not None:
        # Convert columns_to_keep to a list if it is a Pandas Index
        if isinstance(columns_to_keep, pd.Index):
            columns_to_keep = columns_to_keep.tolist()

        # Ensure 'default' is always present
        if 'default' not in columns_to_keep:
            columns_to_keep.append('default')

        for col in columns_to_keep:
            if col not in df:
                df[col] = 0  # Add missing columns with 0
        df = df[columns_to_keep]  # Reorder columns to match training set

    # Check if 'default' column is in the DataFrame
    if 'default' in df.columns:
        X = df.drop(columns=['default'])
        y = df['default']
    else:
        raise KeyError("'default' column not found in DataFrame after processing")

    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df)
    print("Data preprocessing complete. Features and target variables are ready.")


# import pandas as pd
# from sklearn.model_selection import train_test_split

# def load_data(filepath):
#     df = pd.read_csv(filepath, header=1)
#     # Remove any unnamed columns
#     df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#     print("Columns in the DataFrame:", df.columns.tolist())
#     return df

# def preprocess_data(df):
#     # Rename target column
#     df = df.rename(columns={'Y': 'default'})
#     # Convert data types
#     for col in df.columns:
#         if col != 'default':
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#     df['default'] = pd.to_numeric(df['default'], errors='coerce').astype(int)
#     # Handle missing values
#     df = df.dropna()
#     # Drop irrelevant columns if any
#     X = df.drop(['default'], axis=1)
#     y = df['default']
#     return X, y

# def split_data(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
#     return X_train, X_test, y_train, y_test

# if __name__ == "__main__":
#     df = load_data('data/credit_data.csv')
#     X, y = preprocess_data(df)
#     X_train, X_test, y_train, y_test = split_data(X, y)



