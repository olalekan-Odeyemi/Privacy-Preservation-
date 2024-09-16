import pandas as pd
import numpy as np
import random
import diffprivlib.models
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from diffprivlib.models import LinearRegression as DPLinearRegression
from diffprivlib.mechanisms import Laplace

# Ensure the path is correct
dataset_path = r'C:\Users\DELL\Documents\hourlyCalories_heartrate_data.csv'

# Read the dataset
df = pd.read_csv(dataset_path)

# Print column names to verify
print(df.columns)

# Assuming 'FirstColumn' and 'SecondColumn' are the actual column names in your CSV
column_x = 'Heartrate'
column_y = 'Calories'

# Drop rows with NaN values
df = df.dropna(subset=[column_x, column_y])

x = df[column_x].values.reshape(-1, 1)
y = df[column_y]

# Calculate bounds for X and y
bounds_x = (x.min(axis=0), x.max(axis=0))  # Bounds for each feature
bounds_y = (y.min(), y.max())              # Bounds for the target variable

# Calculate mean and variance without noise
mean_without_noise = np.mean(y)
variance_without_noise = np.var(y)

print("\nMean without noise:", mean_without_noise)
print("Variance without noise:", variance_without_noise)

# Fit and display the standard Linear Regression model
print("\nStandard Linear Regression Model:")
std_model = LinearRegression()
std_model.fit(x, y)
std_coef = std_model.coef_
std_intercept = std_model.intercept_
std_r_squared = std_model.score(x, y)

print(f"  Coefficients: {std_coef}")
print(f"  Intercept: {std_intercept}")
print(f"  R-squared: {std_r_squared}")

# Prepare the first plot for standard regression
plt.figure(figsize=(12, 7))
plt.plot(x, std_model.predict(x), color='blue', label='Standard Linear Regression')

# Define the differential privacy levels
Desired_privacy = ['low']
itr_list = []
eps_list = []
colors = ['green', 'yellow', 'red', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'gray']

# Iterate over different numbers of iterations and privacy levels
for iterations in range(1000, 3000, 500):
    for idx, desired_privacy in enumerate(Desired_privacy):
        print(f'\nCalculating results for {desired_privacy} privacy with {iterations} iterations.')
        
        # Determine epsilon based on privacy level
        if desired_privacy == "very_high":
            epsilon = round(random.uniform(0.01, 0.1), 2)
        elif desired_privacy == "high":
            epsilon = round(random.uniform(0.1, 0.25), 2)
        elif desired_privacy == "moderate":
            epsilon = round(random.uniform(0.25, 0.50), 2)
        elif desired_privacy == "low":
            epsilon = round(random.uniform(0.51, 0.75), 2)
        elif desired_privacy == "very_low":
            epsilon = round(random.uniform(0.75, 1.0), 2)

        # Add Laplace noise to y values
        laplace_mechanism = Laplace(epsilon=epsilon, sensitivity=1.0)
        y_noisy = np.array([laplace_mechanism.randomise(value) for value in y])

        # Calculate mean and variance with noise
        mean_with_noise = np.mean(y_noisy)
        variance_with_noise = np.var(y_noisy)

        print(f"  Mean with noise: {mean_with_noise}")
        print(f"  Variance with noise: {variance_with_noise}")
        
        # Use diffprivlib's differentially private Linear Regression
        dp_model = DPLinearRegression(epsilon=epsilon, bounds_X=bounds_x, bounds_y=bounds_y)
        dp_model.fit(x, y_noisy)
        
        dp_coef = dp_model.coef_
        dp_intercept = dp_model.intercept_
        dp_r_squared = dp_model.score(x, y_noisy)
        
        print(f"  Differential Privacy Epsilon: {epsilon}")
        print(f"  Coefficients: {dp_coef}")
        print(f"  Intercept: {dp_intercept}")
        print(f"  R-squared: {dp_r_squared}")
        
        # Plot the differentially private model with unique color for each iteration
        plt.plot(x, dp_model.predict(x), color=colors[iterations % len(colors)], 
                 label=f'DP Linear Regression (eps={epsilon}, iters={iterations})')
        
        itr_list.append(iterations)
        eps_list.append(epsilon)

# Customize and display the first plot
plt.xlabel(column_x)
plt.ylabel(column_y)
plt.title('Linear Regression with and without Differential Privacy')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plot the mean with and without noise
plt.figure(figsize=(12, 7))
bars = plt.bar(['Mean without Noise', 'Mean with Noise'], [mean_without_noise, mean_with_noise], color=['blue', 'red'])
plt.title('Comparison of Means')

# Display the values on the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.show()

# Plot the variance with and without noise
plt.figure(figsize=(12, 7))
bars = plt.bar(['Variance without Noise', 'Variance with Noise'], [variance_without_noise, variance_with_noise], color=['blue', 'red'])
plt.title('Comparison of Variances')

# Display the values on the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.show()

# Predict new values using the last fitted DP model
x_new = np.linspace(x.min(), x.max(), num=10).reshape(-1, 1)
y_new = dp_model.predict(x_new)
print(f"\nNew data for prediction:\n{x_new}")
print(f"Predicted response for new data:\n{y_new}")

# Prepare the second plot
plt.figure(figsize=(12, 7))
plt.plot(x, dp_model.predict(x), color='purple', label='Old Predictions')
plt.scatter(x_new, y_new, color='red', label='New Predictions')
plt.xlabel(column_x)
plt.ylabel(column_y)
plt.title('Old and New Predictions with Differentially Private Linear Regression')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
