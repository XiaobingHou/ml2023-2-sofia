import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def get_input_pairs(num_pairs, set_type):
    print(f"Enter {num_pairs} (x, y) pairs for {set_type} set:")
    x_values = np.array([float(input(f"Enter x value for pair {i+1}: ")) for i in range(num_pairs)])
    y_values = np.array([int(input(f"Enter y value for pair {i+1}: ")) for i in range(num_pairs)])
    return x_values, y_values
def main():
    N = int(input("Enter the number of training pairs (N): "))
    train_x, train_y = get_input_pairs(N, "training")
    M = int(input("\nEnter the number of test pairs (M): "))
    test_x, test_y = get_input_pairs(M, "test")
    k_values = range(1, 11)
    best_k = None
    best_accuracy = 0
    for k in k_values:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        train_x_reshaped = train_x.reshape(-1, 1)
        knn_classifier.fit(train_x_reshaped, train_y)
        test_x_reshaped = test_x.reshape(-1, 1)
        predictions = knn_classifier.predict(test_x_reshaped)
        accuracy = accuracy_score(test_y, predictions)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
    print(f"\nBest k for kNN Classification: {best_k}")
    print(f"Corresponding Test Accuracy: {best_accuracy:.2%}")
if __name__ == "__main__":
    main()
