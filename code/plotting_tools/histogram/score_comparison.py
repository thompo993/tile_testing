import pandas as pd
import matplotlib.pyplot as plt

# Set your file path
file_path = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\Scintillating Tile Tests\pmt_rig_250825\spreadsheets\Data\score_v_peak_ral_resin_250929.csv"   # <-- change to your actual file path

# Read the data (assumes CSV with headers like "voltage_peak" and "total_score")
data = pd.read_csv(file_path)

# Extract the needed columns
voltage_peak = data['voltage_peak']
total_score  = data['total_score']
import numpy as np

# Calculate Pearson correlation coefficient
corr_coef = np.corrcoef(voltage_peak, total_score)[0, 1]

print(f"Correlation coefficient (Voltage Peak vs Total Score): {corr_coef:.4f}")
# Plot
plt.figure(figsize=(8, 5))
plt.scatter(voltage_peak, total_score, marker='o', linestyle='-', linewidth=2)

plt.xlabel('Voltage Peak (decimal number)')
plt.ylabel('Total Score')
plt.title('Voltage Peak vs. Total Score')
plt.grid(True)

plt.show()
