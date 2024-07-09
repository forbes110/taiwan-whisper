import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define the data
# MER
x1 = [1.0, 0.8, 0.6, 0.4, 0.2]             # 0.5
y1 = [94.83, 74.26, 64.15, 55.10, 21.50] # 69.13
# PER
x2 = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2]             #0.5
y2 = [89.80, 75.72, 71.62, 68.06, 61.67, 31.17] #71.62
# ngram+PER
x3 = [1.0, 0.8, 0.6, 0.4, 0.2]                  #0.5
y3 = [90.01, 87.56, 84.21, 74.54, 46.33] # 80.79

# Convert to DataFrame
sns.set_theme()
# Create the dot plot
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax = sns.stripplot(x='Model Parameters (in millions)', y='Mix Error Rate (%)', data=df, jitter=True)
# Customize the color of the leftmost point

# Add text labels for each data point
# ax.plot(x1, y1, label='MER', color='blue', linestyle='-', linewidth=2, marker='x', markersize=5, markerfacecolor='blue')
ax.plot(x1, y1, label='MER', linestyle='-', linewidth=4, marker='x', markersize=10)
ax.plot(x2, y2, label='PER', linestyle='-', linewidth=4, marker='x', markersize=10)
ax.plot(x3, y3, label='ngram+PER', linestyle='-', linewidth=4, marker='x', markersize=10)
ax.set_title("Data Remaining Percentage with Different Filtering Methods", fontsize=18)
ax.set_xlabel("Threshold α", fontsize=18)
ax.set_ylabel("Data Remaining Percentage (%)", fontsize=18)
ax.set_xlim(1.0, 0.2)
ax.set_ylim(0, 100)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(fontsize=18)

# Add titles and labels
# plt.title("Mix Error Rate vs. Model Parameters")
plt.tight_layout()
# plt.subplots_adjust(wspace=0.2)  # Adjust the width space between subplots
# Show the plot
plt.show()
plt.savefig("figure3.pdf")