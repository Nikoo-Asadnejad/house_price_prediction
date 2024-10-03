import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'Meter': [80, 100, 120, 150, 200, 250],
    'Bedrooms': [1, 2, 2, 3, 4, 4],
    'Price': [4000000000, 10000000000, 13000000000, 18000000000, 22000000000, 26000000000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features (SquareFeet and Bedrooms) and Target (Price)
X = df[['Meter', 'Bedrooms']]  # Independent Variables (X)
y = df['Price']                     # Dependent Variable (y)

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Print the model coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Check the model accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")