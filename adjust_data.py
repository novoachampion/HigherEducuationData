#adjust_data.py
import pandas as pd
import numpy as np
import os

def adjust_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Check and adjust the columns where the minimum value is not zero and the data type is numeric
    for column in data.select_dtypes(include=[np.number]).columns:
        if data[column].min() != 0:
            data[column] -= 1

    # Ensure the output directory exists
    output_directory = "./processedData"
    os.makedirs(output_directory, exist_ok=True)

    # Save the adjusted data to an Excel file in the 'processedData' subdirectory
    output_path = os.path.join(output_directory, "ADJUSTED_DATA.xlsx")
    data.to_excel(output_path, index=False)

    return output_path

if __name__ == "__main__":
    file_path = "DATA.csv"  # Assuming the DATA.csv is in the same directory as this script
    print(adjust_data(file_path))
