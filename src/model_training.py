import os
import joblib
from data_preprocessing import load_data, preprocess_data, split_data
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(rf, 'models/random_forest_model.pkl')
    return rf

def train_xgboost(X_train, y_train):
    xgb = XGBClassifier(
        use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42
    )
    xgb.fit(X_train, y_train)
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(xgb, 'models/xgboost_model.pkl')
    return xgb

if __name__ == "__main__":
    df = load_data('data/credit_data.csv')
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    print("Random Forest model trained and saved.")

    print("Training XGBoost model...")
    xgb_model = train_xgboost(X_train, y_train)
    print("XGBoost model trained and saved.")
