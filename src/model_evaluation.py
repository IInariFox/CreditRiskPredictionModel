# src/model_evaluation.py

import joblib
from data_preprocessing import load_data, preprocess_data, split_data
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if __name__ == "__main__":
    df = load_data('data/credit_data.csv')
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    rf_model = joblib.load('models/random_forest_model.pkl')
    xgb_model = joblib.load('models/xgboost_model.pkl')

    evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    evaluate_model(xgb_model, X_test, y_test, 'XGBoost')

