#main.py
from run_decision_tree import run_decision_tree
from run_svm import run_svm
from adjust_data import adjust_data

def main():
    # Path to the original data, assuming it's in the same directory as this script
    file_path = "data/DATA.csv"

    # Adjust data
    adjusted_data_path = adjust_data(file_path)
    print(f"Adjusted data saved to: {adjusted_data_path}")

    # Run Decision Tree algorithm
    decision_tree_results = run_decision_tree()
    print(decision_tree_results)

    # Run SVM algorithm
    svm_results = run_svm()
    print(svm_results)

    # Additional processing functions can be added here

if __name__ == "__main__":
    main()
