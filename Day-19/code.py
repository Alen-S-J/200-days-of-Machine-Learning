import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset as an example
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create three diverse base models
base_model1 = RandomForestClassifier(n_estimators=100, random_state=42)
base_model2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
base_model3 = LogisticRegression(max_iter=10000)

# Train the base models
base_model1.fit(X_train, y_train)
base_model2.fit(X_train, y_train)
base_model3.fit(X_train, y_train)

# Get predictions from the base models
predictions1 = base_model1.predict(X_test)
predictions2 = base_model2.predict(X_test)
predictions3 = base_model3.predict(X_test)

# Create a meta-learner (e.g., Logistic Regression) and use base models' predictions as input
meta_learner = LogisticRegression(max_iter=10000)
meta_learner_input = np.column_stack((predictions1, predictions2, predictions3))
meta_learner.fit(meta_learner_input, y_test)

# Make final predictions using the meta-learner
final_predictions = meta_learner.predict(meta_learner_input)

# Evaluate the performance of the stacked model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, final_predictions)
print("Stacking Accuracy:", accuracy)
