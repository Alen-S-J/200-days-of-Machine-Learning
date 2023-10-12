'''LinearRegression class with methods for fitting the model and making predictions. The fit method uses the least squares method to calculate the coefficients for the linear model, and the predict method predicts target values based on the input data and the learned coefficients.'''



import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None  # Coefficients for the linear model
        self.intercept_ = None  # Intercept for the linear model

    def fit(self, X, y):
        """
        Fit the linear regression model to the given training data.

        Parameters:
        X (numpy.ndarray): Training data, shape (n_samples, n_features)
        y (numpy.ndarray): Target values, shape (n_samples,)

        Returns:
        self
        """
        # Add a column of ones to represent the intercept term
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]

        # Calculate coefficients using the least squares formula
        coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

        # Extract coefficients
        self.intercept_ = coefficients[0]
        self.coef_ = coefficients[1:]

        return self

    def predict(self, X):
        """
        Predict target values for the given input data.

        Parameters:
        X (numpy.ndarray): Input data, shape (n_samples, n_features)

        Returns:
        numpy.ndarray: Predicted target values, shape (n_samples,)
        """
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        return X @ self.coef_ + self.intercept_

# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)  # Random data
    y = 3 * X.squeeze() + 2 + np.random.randn(100)  # Linear relationship with noise

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Print the coefficients
    print("Intercept:", model.intercept_)
    print("Coefficient:", model.coef_)

    # Predict using the model
    X_test = np.array([[1.5], [3.0]])  # Test data
    y_pred = model.predict(X_test)
    print("Predicted values:", y_pred)
