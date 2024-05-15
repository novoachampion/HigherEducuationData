from run_decision_tree import run_decision_tree
from run_svm import run_svm
from run_naive_bayes import run_naive_bayes
from adjust_data import adjust_data
from plot_learning_curve import plot_learning_curve

def main():
    file_path = "data/DATA.csv"
    adjusted_data_path = adjust_data(file_path)
    print(f"Adjusted data saved to: {adjusted_data_path}")

    decision_tree_results = run_decision_tree()
    print(decision_tree_results)

    svm_results = run_svm()
    print(svm_results)

    naive_bayes_results = run_naive_bayes()
    print(naive_bayes_results)

    # Visualize the learning curve for the model
    plot_learning_curve()

if __name__ == "__main__":
    main()
