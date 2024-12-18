import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

outpath = "images/logging.png"

logging_df = pd.read_feather("results/logging.feather")
logging_df["Episode"] = logging_df.index.astype(int)
# Create subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 12.5))

# Plot 1: episode vs epsilon
axs[0].plot(logging_df["Episode"], logging_df["Epsilon"], label="Epsilon", color='gray')
axs[0].set_title("Episode vs Epsilon")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Epsilon")
axs[0].legend()

# Plot 2: episode vs minQ, maxQ with cumulative Q on secondary Y-axis
ax2 = axs[1].twinx()
axs[1].plot(logging_df["Episode"], logging_df["Min Q"], label="Min Q", color='tab:blue')
axs[1].plot(logging_df["Episode"], logging_df["Max Q"], label="Max Q", color='tab:orange')
ax2.plot(logging_df["Episode"], logging_df["Cumulative Q"], label="Cumulative Q", color='tab:green')
axs[1].set_title("Episode vs Min Q, Max Q with Cumulative Q on Secondary Y-axis")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Min Q / Max Q")
ax2.set_ylabel("Cumulative Q")
axs[1].legend(loc='upper left')
ax2.legend(loc='upper right')

# Plot 3: episode vs minR, maxR with cumulative R on secondary Y-axis
ax3 = axs[2].twinx()
axs[2].plot(logging_df["Episode"], logging_df["Min R"], label="Min R", color='tab:blue')
axs[2].plot(logging_df["Episode"], logging_df["Max R"], label="Max R", color='tab:orange')
ax3.plot(logging_df["Episode"], logging_df["Cumulative Reward"], label="Cumulative Reward", color='tab:green')
axs[2].set_title("Episode vs Min R, Max R with Cumulative Reward on Secondary Y-axis")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Min R / Max R")
ax3.set_ylabel("Cumulative Reward")
axs[2].legend(loc='upper left')
ax3.legend(loc='upper right')

# Plot 4: scatterplot of endX, endY - colored by episode
scatter = axs[3].scatter(logging_df["endX"], logging_df["endY"], c=logging_df["Episode"], cmap='viridis')
axs[3].set_title("Scatterplot of endX, endY - Colored by Episode")
axs[3].set_xlabel("endX")
axs[3].set_ylabel("endY")
axs[3].invert_yaxis()  # Reverse the Y-axis limits
fig.colorbar(scatter, ax=axs[3], label="Episode")

# Adjust layout
plt.tight_layout()
fig.savefig(outpath, dpi=200)
plt.close(fig)