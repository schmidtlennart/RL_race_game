import matplotlib.pyplot as plt
import pandas as pd
import sys

outfolder = sys.argv[1]

outpath = f"images/{outfolder}/logging.png"

logging_df = pd.read_feather(f"results/{outfolder}/logging.feather")
logging_df["Episode"] = logging_df.index.astype(int)
 
logging_df.tail()



# Create figure
fig = plt.figure(figsize=(10, 15))

# Plot 1: epsilon, cumulative Q
ax0 = plt.subplot(711)
ax1 = ax0.twinx()
ax0.plot(logging_df["Episode"], logging_df["Epsilon"], label="Epsilon", color='gray')
ax1.plot(logging_df["Episode"], logging_df["Cumulative Q"] / logging_df["Steps"], label="Q", color='darkgreen')
ax0.set_title("Epsilon, Cumulative Q scaled by n steps")
ax0.set_xlabel("Episode")
ax0.set_ylabel("Epsilon")
ax0.legend(loc='upper left')
ax1.set_ylabel("Scaled Cumulative Q")
ax1.legend(loc='upper right')

# Plot 2: epsilon, cumulative Reward
ax2 = plt.subplot(712)
ax3 = ax2.twinx()
ax2.plot(logging_df["Episode"], logging_df["Steps"], label="Steps", color='gray')
ax3.plot(logging_df["Episode"], logging_df["Cumulative Reward"] / logging_df["Steps"], label="Reward", color='lightgreen')
ax2.set_title("n Steps, Cumulative Reward scaled by n steps")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Epsilon")
ax2.legend(loc='upper left')
ax3.set_ylabel("Scaled Cumulative R")
ax3.legend(loc='upper right')

# Plot 3: episode vs epsilon and scaled TD error
ax4_0 = plt.subplot(713)
ax4_1 = ax4_0.twinx()
ax4_0.plot(logging_df["Episode"], logging_df["Epsilon"], label="Epsilon", color='gray')
ax4_1.plot(logging_df["Episode"], logging_df["Cumulative TD-Error"] / logging_df["Steps"], label="Cumulative TD Error", color='purple')
ax4_0.set_title("epsilon, Cumulative TD Error scaled by n steps")
ax4_0.set_xlabel("Episode")
ax4_0.set_ylabel("Epsilon")
ax4_1.set_ylabel("Scaled Cumulative TDE")
ax4_0.legend(loc='upper left')

# Plot 3: episode vs minQ, maxQ with cumulative Q on secondary Y-axis
ax4 = plt.subplot(714)
ax4.plot(logging_df["Episode"], logging_df["Max R"], label="Max Reward", color='tab:blue')
ax4.plot(logging_df["Episode"], logging_df["Max Q"], label="Max Q-Value", color='tab:orange')
ax4.set_title("Maximum Reward vs Q")
ax4.set_xlabel("Episode")
ax4.set_ylabel("Max R Q")
ax4.legend(loc='upper left')

# Plot 4: min R, min Q
ax6 = plt.subplot(715)
ax6.plot(logging_df["Episode"], logging_df["Min Q"], label="Min Q-Value", color='tab:orange')
ax6.plot(logging_df["Episode"], logging_df["Min R"], label="Min Reward", color='tab:blue', linestyle="dashed", alpha=0.7)
ax6.set_title("Minimum Reward vs Q")
ax6.set_xlabel("Episode")
ax6.set_ylabel("Min R / Q")
ax6.legend(loc='upper left')

# Plot 5: scatterplot of endX, endY - colored by episode
ax8 = plt.subplot(716)
scatter = ax8.scatter(logging_df["endX"], logging_df["endY"], c=logging_df["Episode"], cmap='viridis')
ax8.set_title("endX, endY by Episode")
ax8.set_xlabel("endX")
ax8.set_ylabel("endY")
ax8.invert_yaxis()  # Reverse the Y-axis limits
fig.colorbar(scatter, ax=ax8, label="Episode")

# endY~Episode
ax8 = plt.subplot(717)
scatter = ax8.scatter(logging_df["Episode"], logging_df["endY"], alpha=0.7, color="gray")
ax8.set_title("endY by Episode")
ax8.set_xlabel("Episode")
ax8.set_ylabel("endY")
ax8.invert_yaxis()  # Reverse the Y-axis limits

# Adjust layout
plt.tight_layout()
fig.savefig(outpath, dpi=200)
plt.close(fig)