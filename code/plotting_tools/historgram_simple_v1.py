import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: Enter your dataset here
# ----------------------------
peak_values = [
    0.06875,
    0.0640625,
    0.071875,
    0.067187,
    0.0703125,
    0.0640625,
    0.0625,
    0.0609375
]

# ----------------------------
# Step 2: Calculate statistics
# ----------------------------
mean_val = np.mean(peak_values)
std_val = np.std(peak_values)

print(f"Mean of Peak Values: {mean_val:.6f}")
print(f"Standard Deviation: {std_val:.6f}")

# ----------------------------
# Step 3: Plot histogram
# ----------------------------
plt.figure(figsize=(8, 5))
n, bins, patches = plt.hist(peak_values, bins=6, color="skyblue", edgecolor="black", alpha=0.8)

# Label frequencies on top of bars
for i in range(len(n)):
    plt.text((bins[i] + bins[i+1]) / 2, n[i] + 0.02, str(int(n[i])), ha='center', fontsize=10)

# Add labels and title
plt.xlabel("Average Peak Location (Voltage)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Tile 21 Repeatability Test", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Show plot
plt.show()
