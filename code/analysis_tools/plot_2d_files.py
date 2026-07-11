import numpy as np 
import matplotlib.pyplot as plt
import os 
from pathlib import Path

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "lines.linewidth":   1.0,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size":  4.0,
    "ytick.major.size":  4.0,
    "xtick.minor.size":  2.0,
    "ytick.minor.size":  2.0,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "axes.grid":         True,
    "grid.color":        "#CCCCCC",
    "grid.linewidth":    0.4,
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "axes.axisbelow":    True,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.dpi":       600,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "figure.dpi":        150,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#AAAAAA",
    "legend.frameon":    True,
})

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
        # plt.xlabel("{b_hand} (Ch_B) [A.U]".format(b_hand=b_hand))
        # plt.ylabel("{d_hand} (Ch_D) [A.U]".format(d_hand=d_hand))
        # plt.title(file_name)

        plt.xlabel("{b_hand} Light Output [A.U]".format(b_hand=b_hand))
        plt.ylabel("{d_hand} Light Output [A.U]".format(d_hand=d_hand))
        
        plt.title(f"{file_name[:5]} Tile ID {file_name[8:11]} | LHS v RHS Stud Light Output Correlation")
        # Add y=x line to show perfect correlation
        max_val = data.shape[0]
        plt.plot([0, max_val], [0, max_val], 'w--', linewidth=2, label='Perfect Correlation (y=x)')
        plt.legend(loc="best")
        plt.ylim(0, 255)
        plt.xlim(0, 255)
        # Save the plot as a PNG file
        save_file = Path(save_path) / f"{data_file.stem}.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_file}")  
        plt.close()
        
    print(f"All plots saved to {save_path}")

# Example usage
folder_path = r"C:\Users\thomp\OneDrive - University of Bristol\phys\y3\final_fml_rpt\data\210mm_260619\raw"
save_path = r"C:\Users\thomp\OneDrive - University of Bristol\phys\y3\final_fml_rpt\plots\2d_plots"
plot_2d_files(folder_path, save_path)