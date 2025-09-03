import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
folder_path = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\Scintillating Tile Tests\pmt_rig_250825\tile_21_overlay_csv_250903"  # <-- Change this to your folder path

# === INITIALIZE PLOT ===
plt.figure(figsize=(10, 6))

# === LOOP THROUGH ALL .DAT FILES ===
for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(".dat"):
        file_path = os.path.join(folder_path, file_name)

        # Read .dat file (tab-separated, skipping possible empty lines)
        df = pd.read_csv(file_path, sep="\t", engine="python").dropna()

        # Rename columns for safety
        df.columns = ["Volts", "Counts"]

        # Normalize counts to max = 1
        df["Counts"] = df["Counts"] / df["Counts"].max()

        # Plot
        plt.plot(df["Volts"], df["Counts"], label=file_name)

# === FINALIZE PLOT ===
plt.title("Normalized Overlay of .dat Files")
plt.xlabel("Volts")
plt.ylabel("Normalized Counts")
plt.legend(loc="best", fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()
