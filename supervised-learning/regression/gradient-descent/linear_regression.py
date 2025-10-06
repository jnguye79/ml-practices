import pandas as pd
import matplotlib.pyplot as plt
import time

def compute_errors(points, b, m):
    """
    Computes the gradients for linear regression
    """
    # Initalize the error
    error = 0
    # Loop through all the points
    for i in range(len(points)):
        # Get the x and y points
        x = points[i, 0]
        y = points[i, 1]
        # We'll use the error formula. To compute the gradient, based off of the error.
        # Little explanation of the formula. Essentially it's y = mx + b. But we moved the y over to the same side. So it's [y - ((mx + b)^2)].
        error += (y - (m * x + b)) ** 2
        # We also have the (1/N) to account for. So will follow up with that in our return function.
        return error / float(len(points))

def update_parameters(points, starting_b, starting_m, learn_rate, num_iterations):
    """
    Updates the parameters for linear regression
    """
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        # Compute the gradients
        gradients = compute_errors(points, b, m, learn_rate)
        # Update the parameters
        b -= learn_rate * gradients[0]
        m -= learn_rate * gradients[1]
    return b, m

def gradient_desc(b_current, m_current, points, learn_rate):
    # Initialize gradient points
    b_gradient = 0
    m_gradient = 0

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 0]
        # Compute partial derivatives of our error function
        b_gradient += -(2 / len(points)) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / len(points)) * x * (y - ((m_current * x) + b_current))

        # Update the parameters
        new_b = b_current - (learn_rate * b_gradient)
        new_m = m_current - (learn_rate * m_gradient)

        # Return the updated parameters
        return new_b, new_m

def run():
    """
    Runs the program.

    Returns:
        A visual representation of the linear regression model.
    """

    # Load dataset
    points = pd.read_csv('points.csv', delimiter=',')

    # Define hyperparameters -- 'learn_rate' and 'num_iterations'
    # Learning rate is how fast we want to converge to the optimal solution.
    learning_rate = 0.00001
    num_iterations = 1000

    # Use the y = mx + b formula since it's a linear regression.
    init_b = 0
    init_m = 0
    m = len(points)
    # Get all points located in first column for 'x', second column for 'y'
    x = points.iloc[:, 0].values
    y = points.iloc[:, 1].values

    print 'starting gradient descent at b = {0}, m = {1}, error = {2}'.format(init_b, init_m, compute_errors(points.values, init_b, init_m))
    [b , m] = gradient_desc(init_b, init_m, points.values, learn_rate)
    print 'ending gradient descent at b = {0}, m = {1}, error = {2}'.format(b, m, compute_errors(points.values, b, m))

if __name__ == "__main__":
    run()