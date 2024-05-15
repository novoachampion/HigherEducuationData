#run_random_forest.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def run_random_forest():
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

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the parameters for GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 500],  # Change these values to tune the number of trees
            'max_depth': [None, 10, 100],      # Change these values to tune the maximum depth of each tree
            'min_samples_split': [2, 4, 5],  # Change these values to tune the minimum number of samples required to split a node
            'min_samples_leaf': [1, 2, 8]     # Change these values to tune the minimum number of samples required to be at a leaf node
        }

        # Instantiate the RandomForestClassifier
        rf = RandomForestClassifier(random_state=42)

        # Set up the GridSearchCV to tune the parameters
        clf = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
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
            'Model': ['Random Forest']*4,
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
        return "Random Forest metrics added to the Excel file."

    else:
        return "Grade column not found in the data."

if __name__ == "__main__":
    print(run_random_forest())
