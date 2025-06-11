import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Generate synthetic dataset
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 3 * x + 5

# Step 2: Add noise to the data
noise = np.random.normal(0, 3, size=y.shape)
y_noisy = y + noise

# Step 3: Fit a linear regression model
x_reshaped = x.reshape(-1, 1) # Reshape for sklearn
model = LinearRegression()
model.fit(x_reshaped, y_noisy)

# Get the predicted line
y_pred = model.predict(x_reshaped)

# Step 4: Plot the data and the fitted line
plt.figure(figsize=(8, 6))
plt.scatter(x, y_noisy, color='blue', label='Noisy data')
plt.plot(x, y, color='green', linestyle='--', label='True line (y=3x+5)')
plt.plot(x, y_pred, color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression on Noisy Data')
plt.legend()
#plt.show() # show() will not work on the cluster, since there is no GUI
plt.savefig('./dsworkflow/src/Lecture_1/linear_reg_plot.png')