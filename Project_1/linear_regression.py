# This project aims to create a simple linear regression model in machine learning

import numpy as np
import matplotlib.pyplot as plt



def calculate_predicitions(x, m, w, b):
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    
    return f_wb


# Now look for the gradient descent to find the best error cost
def cost_function(x, y, m, w, b):
    
    cost_sum = 0
    f_wb = calculate_predicitions(x, m, w, b)
    for i in range(m):
        cost = (f_wb[i] - y[i]) ** 2
        cost_sum += cost

    total_cost = (1/(2*m)) * cost_sum
    
    return total_cost


# Calculate the gradient descent to find the best values of w and b that will represent the data
def calculate_gradient(x, y, m, w, b):
    d_dw  = 0
    d_db = 0

    f_wb = calculate_predicitions(x, m, w, b)
    for i in range(m): 
        d_dw_i = (f_wb[i] - y[i]) * x[i]
        d_db_i = f_wb[i] - y[i]

        d_dw += d_dw_i
        d_db += d_db_i
    d_dw /= m
    d_db /= m

    return d_dw, d_db


def gradient_descent(x, y, m, alpha, w, b, num_iter):
    temp_w = w
    temp_b = b
    
    for i in range(num_iter):
        # Claculate the gradient 
        d_dw, d_db = calculate_gradient(x, y, m, temp_w, temp_b)

        # Find the new values of w and b
        temp_w = temp_w - alpha * d_dw
        temp_b = temp_b - alpha * d_db
            
    return temp_w, temp_b

# Load the training data
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# Get the number of training examples stored as m
m = x_train.shape[0]
print(f'Size of training example: {m}')

# Define the intial values of w and b 
w = 0
b = 0

# State the larning rate between 0 - 1
learning_rate = 0.010
itterations = 10000

# Using gradient descent function calculate the best w and b that best represents the data
w_final, b_final = gradient_descent(x_train, y_train, m, learning_rate, w, b, itterations)

# Debug
print(f"(w,b) found by gradient descent: ({w_final}, {b_final})")

f_wb = calculate_predicitions(x_train, m, w_final, b_final)
cost = cost_function(x_train, y_train, m, w_final, b_final)

print(f'w: {w_final}, b: {b_final} ',
      f'error cost: {cost}')

# Plot the graph
plt.plot(x_train, f_wb, c='b', label="Predicition")
plt.scatter(x_train, y_train, marker='x', c='r')

plt.title('Table Data Name')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.show()

