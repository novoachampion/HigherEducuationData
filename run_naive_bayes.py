import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def run_naive_bayes():
    # Load the data from an Excel file
    input_path = "./processedData/ADJUSTED_DATA.xlsx"
    data = pd.read_excel(input_path)
    data_numeric = data.select_dtypes(include=[np.number])  # Use only numeric data

    if 'Grade' in data.columns:
        # Prepare the features and target variable
        X = data_numeric.drop('Grade', axis=1)
        y = data['Grade']
        
        # Encode target variable if it's categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Standardize the features to have zero mean and unit variance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Instantiate the Gaussian Naive Bayes classifier
        nb = GaussianNB()

        # Set up the GridSearchCV to tune the var_smoothing parameter
        param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
        clf = GridSearchCV(nb, param_grid, cv=5, scoring='accuracy')
        clf.fit(X_train, y_train)

        # Predict on the testing set
        y_pred = clf.predict(X_test)

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Store the results in a DataFrame
        results_df = pd.DataFrame({
            'Model': ['Naive Bayes']*4,
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        # Write the results to an Excel file without overwriting existing data
        with pd.ExcelWriter(input_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            if 'Algorithm Metrics' in writer.book.sheetnames:
                book = writer.book
                sheet = book['Algorithm Metrics']
                startrow = sheet.max_row
                results_df.to_excel(writer, sheet_name='Algorithm Metrics', startrow=startrow, index=False, header=False)
            else:
                results_df.to_excel(writer, sheet_name='Algorithm Metrics', index=False)

        # Print the best parameters and return a success message
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        return "Naive Bayes metrics added to the Excel file."

    else:
        return "Grade column not found in the data."

if __name__ == "__main__":
    print(run_naive_bayes())
