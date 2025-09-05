import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
import os
import glob
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# ------------------------
# Gaussian function
# ------------------------
def gaussian(x, A, mu, sigma):
    """Gaussian function"""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# ------------------------
# Read .set file for Runtime and StartDateTime
# ------------------------
def read_set_file(data_file_path):
    """
    Read the associated .set file and extract Runtime and StartDateTime
    """
    # Get the .set file path by changing the extension
    set_file_path = Path(data_file_path).with_suffix('.set')
    
    runtime = None
    start_datetime = None
    
    if set_file_path.exists():
        try:
            with open(set_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('RunTime='):
                        runtime = line.split('=')[1]
                    elif line.startswith('StartDateTime='):
                        start_datetime = line.split('=')[1]
        except Exception as e:
            print(f"Error reading .set file {set_file_path}: {e}")
    else:
        print(f"No .set file found for {Path(data_file_path).name}")
    
    return runtime, start_datetime

# ------------------------
# Load PHS data
# ------------------------
def load_phs_file(file_path):
    try:
        file_ext = Path(file_path).suffix.lower()

        if file_ext in ['.txt', '.dat']:
            try:
                data = pd.read_csv(file_path, sep='\t', header=0)
            except:
                data = pd.read_csv(file_path, sep=r'\s+', header=0)
        elif file_ext == '.csv':
            data = pd.read_csv(file_path, header=0)
        else:
            data = pd.read_csv(file_path, sep=None, engine='python', header=0)

        if data.shape[1] >= 2:
            x = data.iloc[:, 0].values
            y = data.iloc[:, 1].values
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            return x[valid_mask], y[valid_mask]
        else:
            print(f"Warning: File {file_path} doesn't have at least 2 columns")
            return None, None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None

# ------------------------
# Analyze highest X peak in one file
# ------------------------
def analyze_largest_peak(x, y, window=21, poly=3, prominence=0.05,
                         show_plot=True, save_plot=False, save_path=None, file_name=None,
                         runtime=None, start_datetime=None):
    """
    Smooths data, finds peak with highest x-value, fits Gaussian, and optionally plots/saves results.
    """
    # Smooth data
    y_smooth = savgol_filter(y, window_length=window, polyorder=poly)

    # Find peaks on smoothed data
    peaks, _ = find_peaks(
        y_smooth,
        height=np.max(y_smooth) * prominence,
        distance=len(y) // 20
    )

    if len(peaks) == 0:
        print("No peaks detected.")
        return None, None, None, None

    # Select peak with highest x-value (rightmost peak)
    highest_x_peak_idx = peaks[np.argmax(x[peaks])]
    peak_x = x[highest_x_peak_idx]
    peak_y = y_smooth[highest_x_peak_idx]

    # Fit Gaussian around highest x peak
    fit_range = (x > peak_x - (x[-1] - x[0]) * 0.05) & (x < peak_x + (x[-1] - x[0]) * 0.05)
    x_fit = x[fit_range]
    y_fit = y_smooth[fit_range]
    p0 = [peak_y, peak_x, (x_fit[-1] - x_fit[0]) / 6]

    try:
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
    except RuntimeError:
        popt = [peak_y, peak_x, (x_fit[-1] - x_fit[0]) / 6]

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, label="Raw Spectrum", color="lightgray", alpha=0.7)
    plt.plot(x, y_smooth, label="Smoothed Spectrum", color="blue", linewidth=2)
    plt.plot(x[peaks], y_smooth[peaks], "ro", markersize=8, label="Detected Peaks")
    plt.plot(x_fit, gaussian(x_fit, *popt), "g--", linewidth=3,
             label="Gaussian Fit (highest x peak)")
    
    # Create info text for the plot
    info_text = f'Runtime: {runtime}\n'
    info_text += f'Start DateTime: {start_datetime}\n'
    info_text += 'Integration Time (too be added)'
    
    
    plt.figtext(0.76, 0.71, info_text,
                fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.axvline(peak_x, color="purple", linestyle="--", linewidth=2, 
                label=f"Peak X = {peak_x:.4f}")
    plt.xlabel("Voltage Output", fontsize=12)
    plt.ylabel("Counts", fontsize=12)
    plt.title(f"Highest X Peak Detection: {file_name if file_name else 'Unknown File'}", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot if requested
    if save_plot and save_path and file_name:
        # Ensure the save path exists
        os.makedirs(save_path, exist_ok=True)
        
        # Generate timestamp in YYMMDD_HHMMSS format
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        
        # Create filename with timestamp
        base_name = os.path.splitext(file_name)[0]
        plot_filename = f"{base_name}_{timestamp}_peak_plot.png"
        full_plot_path = os.path.join(save_path, plot_filename)
        
        try:
            plt.savefig(full_plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {full_plot_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    # Show or close the plot
    if show_plot:
        plt.show()
    else:
        plt.close()

    return peaks, peak_x, peak_y, popt

# ------------------------
# Find all data files in folder
# ------------------------
def find_phs_files(folder_path):
    extensions = ['*.txt', '*.csv', '*.dat', '*.data']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    return sorted(files)

# ------------------------
# Process all files in folder
# ------------------------
def process_phs_folder(folder_path, save_results=True, save_plots=False, custom_save_path=None):
    files = find_phs_files(folder_path)
    if not files:
        print("No valid PHS data files found.")
        return

    results = []
    print(f"Found {len(files)} files to analyze.\n")

    # Use custom save path if provided, else default to folder_path
    save_path = custom_save_path if custom_save_path else folder_path
    
    # Ensure the save path exists
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        if save_plots:
            print(f"Plots will be saved to: {save_path}")

    for i, file in enumerate(files, 1):
        print(f"Processing {i}/{len(files)}: {Path(file).name}")
        x, y = load_phs_file(file)
        if x is None or y is None:
            print(f"Skipping file: {file}")
            continue

        # Read associated .set file
        runtime, start_datetime = read_set_file(file)

        peaks, peak_x, peak_y, popt = analyze_largest_peak(
            x, y,
            show_plot=True,  # Always show the fit interactively
            save_plot=save_plots,
            save_path=save_path,
            file_name=Path(file).name,
            runtime=runtime,
            start_datetime=start_datetime
        )

        if peak_x is None:
            print(f"No peaks found in {file}")
            continue

        results.append({
            "File": Path(file).name,
            "Highest_X_Peak_X": peak_x,
            "Highest_X_Peak_Y": peak_y,
            "Gaussian_A": popt[0],
            "Gaussian_Mu": popt[1],
            "Gaussian_Sigma": popt[2],
            "Runtime": runtime,
            "StartDateTime": start_datetime
        })
        print(f"Peak found at X = {peak_x:.5f}, Y = {peak_y:.2f}\n")

    # Save summary CSV in the save path with timestamp
    if save_results and results:
        # Generate timestamp for the CSV file as well
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        csv_filename = f"PHS_Highest_X_Peak_Summary_{timestamp}.csv"
        csv_path = os.path.join(save_path, csv_filename)
        
        try:
            pd.DataFrame(results).to_csv(csv_path, index=False)
            print(f"Results summary saved to: {csv_path}")
        except Exception as e:
            print(f"Error saving results CSV: {e}")

    # Print results table to console
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("SUMMARY OF HIGHEST X PEAKS:")
        print("="*80)
        display_columns = ["File", "Highest_X_Peak_X", "Highest_X_Peak_Y", "Runtime", "StartDateTime"]
        print(df[display_columns].to_string(index=False))
        print("="*80)
    else:
        print("No results to display.")

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    # Update these paths as needed
    folder_path = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\Scintillating Tile Tests\pmt_rig_250825\bulk_tile_testing\for_code_testing"
    custom_save_path = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\Scintillating Tile Tests\pmt_rig_250825\bulk_tile_testing\for_code_testing"
    
    # Process the folder
    process_phs_folder(folder_path, save_results=True, save_plots=True, custom_save_path=custom_save_path)