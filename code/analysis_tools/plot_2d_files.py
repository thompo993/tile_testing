import numpy as np 
import matplotlib.pyplot as plt
import os 
from pathlib import Path

def plot_2d_files(folder_path, save_path):
    # Create save directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Get all .2D files in the folder
    data_files = list(Path(folder_path).glob("*.2D"))
    
    if not data_files:
        print(f"No .2D files found in {folder_path}")
        return
    
    print(f"Found {len(data_files)} .2D files")
    # we look for LHS or RHS inside of the file name 
    # convention is as you look at the PMT rig, the LHS PMT is
    # on the LHS. this is also channel 2 on the HV power supply
    # we then signal this to the plot labels adding "lhs" or "rhs"
    # inside the string. 
    for data_file in data_files:
        file_name = data_file.name
        print(f"Processing {file_name}...")

        if "lhs" in file_name.lower():
            print("Identified as LHS file")
            b_hand = "LHS Stud"
            d_hand = "RHS Stud"
        elif "rhs" in file_name.lower():
            print("Identified as RHS file")
            b_hand = "RHS Stud"
            d_hand = "LHS Stud"
        else:
            b_hand = ""
            d_hand = ""
        
        # Load the data from the .2D file
        try:
            data = np.loadtxt(data_file)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue
        
        # Create a 2D heatmap plot
        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap='viridis', origin='lower', aspect='auto')   
        plt.colorbar(label='Intensity')
        plt.xlabel("{b_hand} (Ch_B) [A.U]".format(b_hand=b_hand))
        plt.ylabel("{d_hand} (Ch_D) [A.U]".format(d_hand=d_hand))
        plt.title(file_name)
        
        # Add y=x line to show perfect correlation
        max_val = data.shape[0]
        plt.plot([0, max_val], [0, max_val], 'w--', linewidth=2, label='Perfect correlation (y=x)')
        plt.legend(loc="best")
        plt.ylim(0, 255)
        plt.xlim(0, 255)
        # Save the plot as a PNG file
        save_file = Path(save_path) / f"{data_file.stem}.png"
        plt.savefig(save_file, dpi=100, bbox_inches='tight')
        print(f"Saved to {save_file}")  
        plt.close()
        
    print(f"All plots saved to {save_path}")

# Example usage
folder_path = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\scintillating_tiles\dual_pmt_rig_251112\260506_full_stave_tile_selection\210mm"
save_path = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\scintillating_tiles\log\260506_full_stave_tile_selection\210mm\2dplots"
plot_2d_files(folder_path, save_path)