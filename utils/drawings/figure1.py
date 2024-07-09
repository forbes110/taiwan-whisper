import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define the data
data_ood = {
    'Model Parameters (in millions)': [50, 244, 756, 769, 1550, 1600],
    'Mix Error Rate (%)': [14, 32.07, 17.86, 31.83, 25.69, 35],
    'Model': ["", 'Whisper Small', 'Ours (K²D)', 'Whisper Medium', 'Whisper Large-v2', ""]  # Optional labels for models
}

# Convert to DataFrame
df_ood = pd.DataFrame(data_ood)
sns.set_theme()
# Create the dot plot
fig, axs = plt.subplots(1, 2, figsize=(10, 5.4))
# ax = sns.stripplot(x='Model Parameters (in millions)', y='Mix Error Rate (%)', data=df, jitter=True)
# Customize the color of the leftmost point

# Add text labels for each data point
ax = axs[1]
for i in range(df_ood.shape[0]):
    x, y, model = df_ood['Model Parameters (in millions)'][i], df_ood['Mix Error Rate (%)'][i], df_ood['Model'][i]
    if model == 'Ours (K²D)':
        color = 'red'
        s = 100
    elif model == "":
        color = 'black'
        s = 0
    else:
        color = 'blue'
        s = 50
    ax.scatter(x, y, color=color, s=s, zorder=2)
    if model:
        weight = "normal"
        if model == "Ours (K²D)":
            weight = "bold"
        align = 'center'
        if model == "Whisper Small":
            align = 'left'
        if model == "Whisper Medium":
            align = 'left'
            y -= 1.5
        if model == "Whisper Large-v2":
            align = 'right'
        ax.text(
            x=x,
            y=y + 0.5,  # Adjust the y position slightly for better readability
            s=model,
            zorder=5,
            horizontalalignment=align,
            size=16,
            color='black',
            weight=weight,
        )
ax.set_title("Out-of-Domain", fontsize=18)
ax.set_xlabel("Model Parameters (in millions)", fontsize=17)
ax.set_ylabel("Mix Error Rate (%)", fontsize=17)
ax.set_xlim(200, 1600)
ax.set_ylim(16, 35)

data_id = {
    'Model Parameters (in millions)': [60, 244, 756, 769, 1550, 1700],
    'Mix Error Rate (%)': [10, 26.47, 11.44, 23.06, 13.96, 27],
    'Model': ["", 'Whisper Small', 'Ours (K²D)', 'Whisper Medium', 'Whisper Large-v2', ""]  # Optional labels for models
}
df_id = pd.DataFrame(data_id)

ax = axs[0]
for i in range(df_ood.shape[0]):
    x, y, model = df_id['Model Parameters (in millions)'][i], df_id['Mix Error Rate (%)'][i], df_id['Model'][i]
    if model == 'Ours (K²D)':
        color = 'red'
        s = 100
    elif model == "":
        color = 'black'
        s = 0
    else:
        color = 'green'
        s = 50
    ax.scatter(x, y, color=color, s=s, zorder=2)
    if model:
        weight = "normal"
        if model == "Ours (K²D)":
            weight = "bold"
        align = 'center'
        if model == "Whisper Small":
            align = 'left'
        if model == "Whisper Large-v2":
            align = 'right'
        ax.text(
            x=x,
            y=y + 0.5,  # Adjust the y position slightly for better readability
            s=model,
            zorder=5,
            horizontalalignment=align,
            size=16,
            color='black',
            weight=weight,
        )
ax.set_title("In-Domain", fontsize=18)
ax.set_xlabel("Model Parameters (in millions)", fontsize=17)
ax.set_ylabel("Mix Error Rate (%)", fontsize=17)
ax.set_xlim(200, 1600)
ax.set_ylim(10, 29)
# Add titles and labels
# plt.title("Mix Error Rate vs. Model Parameters")
plt.tight_layout()
plt.subplots_adjust(wspace=0.2)  # Adjust the width space between subplots
# Show the plot
plt.show()
plt.savefig("figure1.pdf")