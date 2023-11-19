import numpy as np  
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.metrics import r2_score  
  
def knn_regression(X_train, y_train, X_test, k):  
    knn_model = KNeighborsRegressor(n_neighbors=k)  
    knn_model.fit(X_train, y_train)  
    y_pred = knn_model.predict(X_test)  
    r2 = r2_score(y_train, y_pred)  
    return y_pred, r2  
  
N = int(input("Enter the value of N (positive integer): "))  
k = int(input("Enter the value of k (positive integer): "))  
points = []  
for i in range(N):  
    x = float(input(f"Enter x value for point {i + 1}: "))  
    y = float(input(f"Enter y value for point {i + 1}: "))  
    points.append((x, y))  
data = np.array(points)  
X = data[:, 0].reshape(-1, 1)  
y = data[:, 1]  
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
  
if k <= len(X_train):   
    result, coefficient_of_determination = knn_regression(X_train, y_train, X_test, k)  
    print(f"Result (Y) of k-NN Regression: {result}")  
    print(f"Coefficient of Determination: {coefficient_of_determination}")  
else:  
    print("Error: k should be less than or equal to the number of training samples.")
