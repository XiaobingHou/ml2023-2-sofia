import numpy as np
n = int(input("Enter the number of points N: "))
k = int(input("Enter the value of k: "))
x_train = []
y_train = []
for i in range(n):
    x, y = map(float, input("Enter x and y coordinates separated by space: ").split())
    x_train.append(x)
    y_train.append(y)
x_train = np.array(x_train)
y_train = np.array(y_train)
test_x = float(input("Enter the test point x: "))
distances = np.sqrt(np.sum((x_train - test_x)**2, axis=1))
indices = np.argsort(distances)[:k]
weights = 1.0 / distances[indices]**2
y_pred = np.dot(weights, y_train[indices])
if k > n:
    print("Error: k must be less than or equal to N")
else:
    print("The predicted value of y is:", y_pred)
