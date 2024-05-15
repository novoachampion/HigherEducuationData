#run_decision_tree.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os

def run_decision_tree():
    input_path = "./processedData/ADJUSTED_DATA.xlsx"
    data = pd.read_excel(input_path)
    data_numeric = data.select_dtypes(include=[np.number])
    
    if 'Grade' in data.columns:
        X = data_numeric.drop('Grade', axis=1)
        y = data['Grade']

        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        results_df = pd.DataFrame({
            'Model': ['Decision Tree']*4,
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        with pd.ExcelWriter(input_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            if 'Algorithm Metrics' in writer.book.sheetnames:
                book = writer.book
                sheet = book['Algorithm Metrics']
                startrow = sheet.max_row
                results_df.to_excel(writer, sheet_name='Algorithm Metrics', startrow=startrow, index=False, header=False)
            else:
                results_df.to_excel(writer, sheet_name='Algorithm Metrics', index=False)

        return "Decision Tree metrics added to the Excel file."

    else:
        return "Grade column not found in the data."

if __name__ == "__main__":
    print(run_decision_tree())
