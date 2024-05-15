#run_svm.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from bokeh.plotting import figure, show
from bokeh.io import output_file

def plot_curve(title, x, y, error_y, x_label, y_label):
    p = figure(title=title, x_axis_label=x_label, y_axis_label=y_label, sizing_mode="stretch_width", height=250)
    p.line(x, y, line_width=2)
    p.multi_line(xs=[x, x], ys=[y-error_y, y+error_y], color='gray')
    show(p)

def run_svm():
    input_path = "./processedData/ADJUSTED_DATA.xlsx"
    data = pd.read_excel(input_path)
    data_numeric = data.select_dtypes(include=[np.number])

    if 'Grade' in data.columns:
        X = data_numeric.drop('Grade', axis=1)
        y = data['Grade']
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        svm = SVC(random_state=42)
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100], 'gamma':('scale', 'auto')}
        clf = GridSearchCV(svm, parameters)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Bokeh plots (if desired, add actual plotting here)

        results_df = pd.DataFrame({
            'Model': ['SVM']*4,
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

        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        return "SVM metrics added to the Excel file."

    else:
        return "Grade column not found in the data."

if __name__ == "__main__":
    run_svm()
