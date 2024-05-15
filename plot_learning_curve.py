import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression  # Example classifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot
from bokeh.io import output_notebook

def plot_learning_curve():
    # Load the dataset
    input_path = "./processedData/ADJUSTED_DATA.xlsx"
    data = pd.read_excel(input_path)
    # Ensure 'Grade' is your target and is included in the dataset
    X = data.drop('Grade', axis=1)  # Assuming 'Grade' is the target
    y = data['Grade']

    # Creating a pipeline for preprocessing and modeling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scale data
        ('classifier', LogisticRegression())  # Use Logistic Regression as an example model
    ])

    # Define training sizes and calculate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy', n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plotting the results
    p = figure(title="Learning Curve", x_axis_label='Training examples', y_axis_label='Score', width=800, height=400)
    p.line(train_sizes, train_scores_mean, color='blue', legend_label='Train score')
    p.line(train_sizes, test_scores_mean, color='green', legend_label='Cross-validation score')

    # Drawing bands for the standard deviation of the mean
    p.varea(x=train_sizes, y1=train_scores_mean - train_scores_std, y2=train_scores_mean + train_scores_std, fill_alpha=0.1, color='blue')
    p.varea(x=train_sizes, y1=test_scores_mean - test_scores_std, y2=test_scores_mean + test_scores_std, fill_alpha=0.1, color='green')

    p.legend.location = 'bottom_right'
    output_file("learning_curve.html")  # Save the plot as an HTML file
    show(p)  # Show the plot

if __name__ == "__main__":
    plot_learning_curve()
