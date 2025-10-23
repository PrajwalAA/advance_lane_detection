import numpy as np
import matplotlib.pyplot as plt

# The original error suggests you tried to access 'best_fit' on a visual line object.
# We correct this by performing the calculation on the *data* first.

def plot_with_best_fit(x_data, y_data):
    """
    Calculates and plots the line of best fit (linear regression) along with the raw data.
    """
    print("--- Calculating Linear Regression ---")

    # 1. Calculate the Best Fit (Linear Regression)
    # np.polyfit(x, y, degree) returns the coefficients [m, c] for the line y = m*x + c
    # The degree=1 specifies a linear (straight) line.
    try:
        coefficients = np.polyfit(x_data, y_data, 1)
        # Extract slope (m) and y-intercept (c)
        m, c = coefficients
        print(f"Coefficients found: Slope (m) = {m:.4f}, Intercept (c) = {c:.4f}")

    except Exception as e:
        print(f"Error during polyfit calculation: {e}")
        return

    # 2. Generate the predicted y-values using the calculated line equation
    # y_fit = m * x + c
    y_fit = m * x_data + c

    # 3. Plotting the results
    plt.figure(figsize=(8, 5))

    # Plot the original data points
    plt.scatter(x_data, y_data, label='Raw Data Points', color='royalblue')

    # Plot the calculated line of best fit
    # This line object (line_fit) is what you might have incorrectly tried to access 'best_fit' on
    line_fit, = plt.plot(x_data, y_fit, color='darkorange', linewidth=2, 
                         label=f'Best Fit Line: y = {m:.2f}x + {c:.2f}')

    plt.title('Data with Calculated Line of Best Fit')
    plt.xlabel('X Data')
    plt.ylabel('Y Data')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()

# --- Example Usage ---
# Dummy data (replace with your actual data)
np.random.seed(42)
X = np.arange(1, 11)
# Create data that is roughly linear, plus some random noise
Y = 2 * X + 5 + np.random.randn(10) * 3

# Run the function with the data
plot_with_best_fit(X, Y)

# Now, if you inspect 'line_fit' object from matplotlib, it still won't have a 'best_fit' 
# attribute, because the calculation was done beforehand using numpy.
# The variable 'line_fit' is just the visual representation of the calculated line.
