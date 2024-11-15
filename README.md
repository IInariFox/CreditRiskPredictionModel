# Credit Risk Prediction Model

## Overview
This project is a machine learning-based credit risk prediction system designed to evaluate the likelihood of a loan applicant defaulting on their payment. It provides an automated approach to assess creditworthiness, thereby supporting financial institutions in making informed lending decisions.

## Goals
- **Accurately predict credit risk**: Use historical loan and financial data to train a model capable of distinguishing between low and high-risk loan applicants.
- **Support decision-making**: Assist financial institutions in evaluating potential risks, improving the reliability of credit assessments.
- **Scalability**: Implement a scalable data pipeline that can be updated with new data for continuous model training and improvement.

## Capabilities and Features
- **Data Preprocessing**:
  - Cleans and preprocesses raw financial data to prepare it for model training.
  - Handles missing values, drops unnecessary columns, and one-hot encodes categorical variables for robust data preparation.
- **Model Training**:
  - Utilizes a Random Forest Classifier (or other machine learning models) to train on preprocessed data.
  - Splits data into training and testing sets to evaluate model performance.
  - Saves trained models and feature columns for consistent evaluation and prediction.
- **Model Evaluation**:
  - Evaluates the trained model on unseen data and reports performance metrics such as accuracy.
  - Ensures that the feature columns used during training are applied during evaluation for consistency.
- **Custom Predictions**:
  - Can be adapted to load new data and generate predictions using the trained model.
  - Ensures that new input data is preprocessed consistently with the training data.

## How to Run the Project
1. **Environment Setup**:
   - Ensure you have Python installed, and set up a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # macOS/Linux
     .\venv\Scripts\activate   # Windows
     ```
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run the Data Preprocessing**:
   - Preprocess the data by running the `data_preprocessing.py` script:
     ```bash
     python src\data_preprocessing.py
     ```
   - This script loads and preprocesses the input data and prepares it for training.

3. **Train the Model**:
   - Train the model using the preprocessed data:
     ```bash
     python src\model_training.py
     ```
   - The script trains the model, reports accuracy, and saves the model and training columns for later use.

4. **Evaluate the Model**:
   - Run the `model_evaluation.py` script to assess the model's performance:
     ```bash
     python src\model_evaluation.py
     ```
   - The evaluation script reports the model's accuracy on the evaluation set.

5. **Run Custom Predictions**:
   - Modify or create a script to load the trained model and predict outcomes on new data.
