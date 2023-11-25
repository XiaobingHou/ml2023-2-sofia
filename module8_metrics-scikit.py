import numpy as np
from sklearn.metrics import precision_score, recall_score
def get_user_input():
    N = int(input("Enter the number of data points (N): "))
    return N
def get_data_points(N):
    x_values = []
    y_values = []

    for i in range(N):
        x = int(input(f"Enter the x value for data point {i + 1} (0 or 1): "))
        y = int(input(f"Enter the y value for data point {i + 1} (0 or 1): "))
        x_values.append(x)
        y_values.append(y)

    return np.array(x_values), np.array(y_values)
def compute_precision_recall(x_values, y_values):
    precision = precision_score(x_values, y_values)
    recall = recall_score(x_values, y_values)
    return precision, recall
def main():
    N = get_user_input()
    x_values, y_values = get_data_points(N)

    precision, recall = compute_precision_recall(x_values, y_values)

    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

if __name__ == "__main__":
    main()
