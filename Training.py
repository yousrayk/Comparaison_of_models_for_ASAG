from sklearn.linear_model import LinearRegression, Ridge
from sklearn.isotonic import IsotonicRegression
import numpy as np

class RegressionAnalyzer:
    def __init__(self, training_features, training_labels, test_features):
        self.training_features = training_features
        self.training_labels = training_labels
        self.test_features = test_features

    def handle_nan_values(self, array):
        nan_indices = np.isnan(array)
        array[nan_indices] = 0
        return array

    def linear_regression(self):
        linear_model = LinearRegression()

        train_features = self.handle_nan_values(self.training_features.to_numpy().reshape(-1, 1))
        train_labels = self.handle_nan_values(self.training_labels.to_numpy().reshape(-1, 1))
        test_features = self.handle_nan_values(self.test_features.to_numpy().reshape(-1, 1))

        linear_model.fit(train_features, train_labels)
        predicted_labels = linear_model.predict(test_features)
        return predicted_labels

    def ridge_regression(self):
        ridge_model = Ridge()

        train_features = self.handle_nan_values(self.training_features.to_numpy().reshape(-1, 1))
        train_labels = self.handle_nan_values(self.training_labels.to_numpy().reshape(-1, 1))
        test_features = self.handle_nan_values(self.test_features.to_numpy().reshape(-1, 1))

        ridge_model.fit(train_features, train_labels)
        predicted_labels = ridge_model.predict(test_features)
        return predicted_labels
    
    def isotonic_regression(self):
        isotonic_model = IsotonicRegression(out_of_bounds='clip')

        train_features = self.handle_nan_values(self.training_features.to_numpy().reshape(-1, 1))
        train_labels = self.handle_nan_values(self.training_labels.to_numpy().reshape(-1, 1))
        test_features = self.handle_nan_values(self.test_features.to_numpy().reshape(-1, 1))

        isotonic_model.fit(train_features.flatten(), train_labels.flatten())
        predicted_labels = isotonic_model.predict(test_features.flatten())
        return predicted_labels
