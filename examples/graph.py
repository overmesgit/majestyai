import numpy as np
import matplotlib.pyplot as plt

def calculate_probability(num1, num2):
    if num1 < 0 or num1 > 100 or num2 < 0 or num2 > 100:
        raise ValueError("Both numbers must be between 0 and 100")

    probability = num2 / (num1 + num2)
    return probability

# Create a meshgrid of values for num1 and num2
num1_values, num2_values = np.meshgrid(np.linspace(1, 100, 100), np.linspace(1, 100, 100))

# Calculate the probabilities for each combination of num1 and num2
probabilities = np.vectorize(calculate_probability)(num1_values, num2_values)

# Create a contour plot of the probabilities
plt.contourf(num1_values, num2_values, probabilities, levels=20, cmap='viridis')
plt.colorbar(label='Probability')

# Set the axis labels
plt.xlabel('Number 1')
plt.ylabel('Number 2')

# Set the title
plt.title('Probability Based on Number 1 and Number 2')

# Display the plot
plt.show()
