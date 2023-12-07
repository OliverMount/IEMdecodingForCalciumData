### Conceptual figure making

import numpy as np
import matplotlib.pyplot as plt

# Set a seed for reproducibility
#np.random.seed(42)

# Mean values
mean_values = np.arange(22.5, 360, 45)
num_points=len(mean_values)
# Number of neruons per stimulus
num_curves = 2 

# Initialize an array to store Gaussian curves
gaussian_curves = np.zeros((num_points, num_points * num_curves))



fig, ax = plt.subplots(1, figsize=(17, 5))


# Generate and plot Gaussian curves
#colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
colors=  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']

for idx, mean in enumerate(mean_values):
    for curve_num in range(num_curves):
        # Randomly choose standard deviation in the range (10, 20)
        std_dev = np.random.uniform(40, 50)

        # Randomly choose peak amplitude in the range (1, 10)
        peak_amplitude = np.random.uniform(1, 10)

        # Use mean_values directly instead of generating x values
        x = mean_values

        # Generate Gaussian curve
        y = peak_amplitude * np.exp(-(x - mean)**2 / (2 * std_dev**2))
        
        # Store the Gaussian curve in the array
        gaussian_curves[:, idx * num_curves + curve_num] = y

        # Plot only the data points at the mean with a smaller and unfilled marker
        ax.scatter(x, y, color=colors[idx], marker='o', s=25)

        # Plot the entire curve
        ax.plot(x, y, color=colors[idx], alpha=0.8,lw=3)

# Set x-axis ticks to show mean values
ax.set_xticks([])

# Add vertical lines to mark mean values
for mean in mean_values:
    ax.axvline(x=mean, color='gray', linestyle='--', alpha=0.5) 

# Set y-axis limits to ensure zero points are clearly visible
ax.set_ylim(-0.1,10)

# Set labels and title
#ax.set_xlabel('Mean motion direction')
#ax.set_ylabel('Tuning curve amplitude (a.u)')
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_linewidth(3)
ax.tick_params(axis='both', which='major', left=False, right=False, labelleft=False)

# Show the plot
plt.show() 


given_stimulus=3
row_to_plot = gaussian_curves[given_stimulus]   
averaged_row = np.mean(row_to_plot.reshape(-1, 2), axis=1)
reshaped_values = row_to_plot.reshape(-1, 2)
flattened_values = reshaped_values.flatten()

x_positions = np.arange(0, 8)


fig, ax = plt.subplots(1, figsize=(17, 5))
# Plot the scatter plot
for k in range(len(x_positions)):
	ax.scatter(np.repeat(x_positions[k], 2), flattened_values[2*k:2*k+2], color=colors[k], marker='o')
ax.plot(averaged_row, color='k', linestyle='-', linewidth=2,marker='o',alpha=0.3)

ax.set_xticks([])
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_linewidth(3)
ax.tick_params(axis='both', which='major', left=False, right=False, labelleft=False)
ax.set_ylim(-0.1,10)

# Show the plot
plt.show()

