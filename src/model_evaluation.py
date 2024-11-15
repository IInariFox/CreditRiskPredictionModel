# src/model_evaluation.py

from data_preprocessing import load_data, preprocess_data
import joblib
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    model = joblib.load('models/random_forest_model.pkl')
    training_columns = joblib.load('models/training_columns.pkl')

    # Load and preprocess the evaluation dataset
    df = load_data()
    X, y = preprocess_data(df, columns_to_keep=training_columns)

    if 'default' not in df.columns:
        raise KeyError("'default' column not found in the evaluation DataFrame")

    # Predict using the loaded model
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model accuracy on the evaluation set: {accuracy:.2f}")



