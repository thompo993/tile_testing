import pandas as pd 
import matplotlib.pyplot as plt

filepath = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\Scintillating Tile Tests\dual_pmt_rig_251112\calibration\gain_match_csv.xlsx.csv"

df = pd.read_csv(filepath, on_bad_lines='skip', delimiter=",")
print(df.columns)

bias = df["bias"]
lhs_peak = df["lhs_peak"]
rhs_peak = df["rhs_peak"]



fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(lhs_peak, bias, label = "LHS PMT")
ax.plot(rhs_peak, bias, label = "RHS PMT")




ax.set_xlabel("Photon Peak")
ax.set_ylabel("Bias")
ax.set_title("Gain Matching of Dual PMT Rig - 251113")
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()