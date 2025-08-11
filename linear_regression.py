import pandas as pd
import matplotlib.pyplot as plt
import time

def compute_gradients(x, y, init_b, init_m):
    """
    Computes the gradients for linear regression
    """
    m = len(x)
    y_pred = init_m * x + init_b
    gradients = {
        'm': (-2/m) * sum(x * (y - y_pred)),
        'b': (-2/m) * sum(y - y_pred)
    }
    return gradients

def update_parameters(gradients, learn_rate):
    """
    Updates the parameters for linear regression
    """
    global init_b, init_m
    init_b -= learn_rate * gradients['b']
    init_m -= learn_rate * gradients['m']

def run():
    """
    Runs the program

    Returns:
        A visual compilation of the linear regression model
    """

    # Load dataset
    points = pd.read_csv('points.csv', delimiter=',')

    # Define hyperparameters -- 'learn_rate' and 'num_iterations'
    learn_rate = 0.00001
    num_iterations = 1000

    # Use the y = mx + b. Going to plot the first graph
    init_b = 0
    init_m = 0
    m = len(points)
    # Get all points located in first column for 'x', second column for 'y'
    x = points.iloc[:, 0].values
    y = points.iloc[:, 1].values

    for i in range(num_iterations):
        # Compute the gradient
        gradients = compute_gradients(points, learn_rate)
        # Update the parameters
        update_parameters(gradients, learn_rate)

if __name__ == "__main__":
    run()