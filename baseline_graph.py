import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('output_novel.csv')

# Plot the data
plt.figure(figsize=(10, 6))

# Plot Loss
plt.plot(df['Iteration'], df['Loss'], label='Loss', color='blue', marker='o')

# Plot BPC
plt.plot(df['Iteration'], df['BPC'], label='BPC', color='red', marker='x')

# Adding titles and labels
plt.title('Loss and BPC over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Values')
plt.legend()

# Show grid
plt.grid(True)

# Display the plot
plt.show()
