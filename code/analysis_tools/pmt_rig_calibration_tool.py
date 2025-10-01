import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog
import glob
import pandas as pd
from datetime import datetime
import re

def gaussian(x, amp, mean, std):
    """Gaussian function for curve fitting"""
    return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

def select_directory():
    """Open a dialog to select directory containing .dat files"""
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select directory containing .dat files")
    root.destroy()
    return directory

def get_output_directory():
    """Get the fixed output directory for saving plots"""
    output_dir = r"tile_testing\Figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def extract_voltage_from_filename(filename):
    """Extract the first numerical value from the filename as voltage"""
    numbers = re.findall(r'\d+\.?\d*', filename)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None

def determine_channel_from_filename(filename):
    """
    Determine if file belongs to ch0 or chB based on filename patterns.
    Returns 'ch0' or 'chB'
    """
    filename_lower = filename.lower()
    
    # Check for explicit channel indicators
    if 'ch0' in filename_lower or 'channel0' in filename_lower or '_0_' in filename_lower:
        return 'ch0'
    elif 'chb' in filename_lower or 'channelb' in filename_lower or '_b_' in filename_lower:
        return 'chB'
    
    # Default assignment if no clear indicator
    # You can modify this logic based on your naming convention
    return 'ch0'  # Default to ch0

def read_spectrum_file(filepath):
    """Read pulse height spectrum data from .dat file"""
    try:
        df = pd.read_csv(filepath, sep='\t', header=0)
        df.columns = df.columns.str.strip()
        channels_data = {}
        
        # Check for various channel naming patterns
        # Support both old format (Ch_A, Ch_B) and new format (Ch_B, Ch_D, Ch_B+D)
        channel_mappings = [
            ('Volts:Ch_A', 'Counts:Ch_A'),
            ('Volts:Ch_B', 'Counts:Ch_B'),
            ('Volts:Ch_D', 'Counts:Ch_D'),
            ('Volts:Ch_ B+D', 'Counts:Ch_B+D'),
        ]
        
        for volt_col, count_col in channel_mappings:
            if volt_col in df.columns and count_col in df.columns:
                volts = df[volt_col].values
                counts = df[count_col].values
                # Extract the actual channel name from the column
                channel_name = volt_col.replace('Volts:', '').replace('Counts:', '')
                channels_data[channel_name] = (volts, counts)
        
        return channels_data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def analyze_highest_peak(x, y, window=21, poly=3, prominence=0.05, 
                        show_plot=True, save_plot=True, save_path=None, 
                        file_name=None, channel_name=None, voltage=None,
                        assigned_channel=None):
    """
    Finds the peak with highest x-value (rightmost peak), fits Gaussian, and plots results.
    
    Parameters:
    -----------
    assigned_channel : str
        The assigned channel name ('ch0' or 'chB') based on filename
    
    Returns:
    --------
    peak_x : float
        X-position (voltage) of the highest peak
    peak_y : float
        Y-value (counts) of the highest peak
    popt : array
        Gaussian fit parameters [amplitude, mean, std]
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
        print(f"No peaks detected in {channel_name if channel_name else 'data'}.")
        return None, None, None

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
             label=f"Gaussian Fit (μ={popt[1]:.4f}V, σ={popt[2]:.4f}V)")
    
    plt.axvline(peak_x, color="purple", linestyle="--", linewidth=2, 
                label=f"Highest Peak X = {peak_x:.4f}V")
    
    # Add info text
    if voltage is not None:
        info_text = f'Applied Voltage: {voltage}V\n'
        info_text += f'Peak Position: {peak_x:.4f}V\n'
        info_text += f'Peak Height: {peak_y:.1f} counts\n'
        if assigned_channel:
            info_text += f'Assigned Channel: {assigned_channel}\n'
        if channel_name:
            info_text += f'Data Channel: {channel_name}'
        
        plt.figtext(0.76, 0.68, info_text,
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.5", 
                                          facecolor="lightgray", alpha=0.8))
    
    plt.xlabel("Voltage Output (V)", fontsize=12)
    plt.ylabel("Counts", fontsize=12)
    
    title_suffix = f" - {assigned_channel if assigned_channel else channel_name}" if (assigned_channel or channel_name) else ""
    plt.title(f"Highest Peak Detection: {file_name if file_name else 'Unknown File'}{title_suffix}", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot if requested
    if save_plot and save_path and file_name:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        base_name = os.path.splitext(file_name)[0]
        channel_suffix = f"_{assigned_channel}" if assigned_channel else (f"_{channel_name}" if channel_name else "")
        plot_filename = f"{base_name}_{timestamp}{channel_suffix}_highest_peak.png"
        full_plot_path = os.path.join(save_path, plot_filename)
        
        try:
            plt.savefig(full_plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {full_plot_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return peak_x, peak_y, popt

def plot_voltage_vs_peak_position(voltage_data, output_directory):
    """Plot applied voltage vs highest peak position with best fit lines"""
    
    # Separate data by assigned channel (ch0 and chB)
    ch0_voltages = [v for v, ch, _ in voltage_data if ch == 'ch0']
    ch0_peaks = [p for v, ch, p in voltage_data if ch == 'ch0']
    
    chB_voltages = [v for v, ch, _ in voltage_data if ch == 'chB']
    chB_peaks = [p for v, ch, p in voltage_data if ch == 'chB']
    
    plt.figure(figsize=(10, 6))
    
    # Plot ch0 data
    if ch0_voltages:
        ch0_voltages = np.array(ch0_voltages)
        ch0_peaks = np.array(ch0_peaks)
        
        sort_idx_0 = np.argsort(ch0_voltages)
        ch0_voltages = ch0_voltages[sort_idx_0]
        ch0_peaks = ch0_peaks[sort_idx_0]
        
        fit_0 = np.polyfit(ch0_voltages, ch0_peaks, 1)
        line_0 = np.poly1d(fit_0)
        
        plt.scatter(ch0_voltages, ch0_peaks, color='blue', label='ch0', s=50)
        plt.plot(ch0_voltages, line_0(ch0_voltages), color='blue', linestyle='--', 
                label=f'ch0 Best fit (slope: {fit_0[0]:.6f})')
        
        print(f"ch0 - Slope: {fit_0[0]:.6f}, Intercept: {fit_0[1]:.6f}")
    
    # Plot chB data
    if chB_voltages:
        chB_voltages = np.array(chB_voltages)
        chB_peaks = np.array(chB_peaks)
        
        sort_idx_B = np.argsort(chB_voltages)
        chB_voltages = chB_voltages[sort_idx_B]
        chB_peaks = chB_peaks[sort_idx_B]
        
        fit_B = np.polyfit(chB_voltages, chB_peaks, 1)
        line_B = np.poly1d(fit_B)
        
        plt.scatter(chB_voltages, chB_peaks, color='red', label='chB', s=50)
        plt.plot(chB_voltages, line_B(chB_voltages), color='red', linestyle='--', 
                label=f'chB Best fit (slope: {fit_B[0]:.6f})')
        
        print(f"chB - Slope: {fit_B[0]:.6f}, Intercept: {fit_B[1]:.6f}")
    
    plt.title('Applied Voltage vs Highest Peak Position')
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('Peak Position (V)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save summary plot
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"voltage_vs_peak_position_{current_datetime}.png"
    summary_path = os.path.join(output_directory, summary_filename)
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {summary_filename}")
    
    plt.show()
    plt.close()

def analyze_spectra_directory():
    """Main function to analyze all .dat files in selected directory"""
    
    input_directory = select_directory()
    if not input_directory:
        print("No input directory selected. Exiting.")
        return
    
    output_directory = get_output_directory()
    dat_files = glob.glob(os.path.join(input_directory, "*.dat"))
    
    if not dat_files:
        print(f"No .dat files found in {input_directory}")
        return
    
    print(f"Found {len(dat_files)} .dat files")
    print(f"Saving all plots to: {output_directory}")
    
    voltage_peak_data = []
    
    for filepath in dat_files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing {filename}...")
        
        # Determine channel assignment from filename
        assigned_channel = determine_channel_from_filename(filename)
        print(f"Assigned to: {assigned_channel}")
        
        voltage = extract_voltage_from_filename(filename)
        if voltage is None:
            print(f"Warning: Could not extract voltage from filename {filename}")
            continue
        
        print(f"Extracted voltage: {voltage}")
        channels_data = read_spectrum_file(filepath)
        
        if channels_data is None:
            continue
        
        # Process all data channels in the file, but assign them to ch0 or chB based on filename
        for channel_name, (volts, counts) in channels_data.items():
            print(f"\nAnalyzing {channel_name} (assigned to {assigned_channel})...")
            
            if np.sum(counts) == 0:
                print(f"No counts found in {channel_name}, skipping...")
                continue
            
            peak_x, peak_y, popt = analyze_highest_peak(
                volts, counts,
                window=21,
                poly=3,
                prominence=0.05,
                show_plot=True,
                save_plot=True,
                save_path=output_directory,
                file_name=filename,
                channel_name=channel_name,
                voltage=voltage,
                assigned_channel=assigned_channel
            )
            
            if peak_x is not None:
                print(f"Highest peak found at: {peak_x:.4f}V with height {peak_y:.1f} counts")
                # Store with assigned channel (ch0 or chB), not the data channel name
                voltage_peak_data.append((voltage, assigned_channel, peak_x))
            else:
                print(f"No peaks found in {channel_name}")
    
    # Create summary plot
    if voltage_peak_data:
        print(f"\nCreating summary plot with {len(voltage_peak_data)} data points...")
        plot_voltage_vs_peak_position(voltage_peak_data, output_directory)
        
        print("\nSummary of collected data:")
        for voltage, channel, peak_pos in voltage_peak_data:
            print(f"Voltage: {voltage}V, Channel: {channel}, Peak Position: {peak_pos:.4f}V")
    else:
        print("No data collected for summary plot")
    
    print(f"\nAnalysis complete! All plots saved to: {output_directory}")

def analyze_single_file(filepath, show_plots=True):
    """Analyze a single .dat file"""
    
    filename = os.path.basename(filepath)
    print(f"Processing {filename}...")
    
    # Determine channel assignment from filename
    assigned_channel = determine_channel_from_filename(filename)
    print(f"Assigned to: {assigned_channel}")
    
    voltage = extract_voltage_from_filename(filename)
    print(f"Extracted voltage: {voltage}")
    
    channels_data = read_spectrum_file(filepath)
    
    if channels_data is None:
        print("Failed to read file")
        return
    
    output_dir = get_output_directory()
    
    for channel_name, (volts, counts) in channels_data.items():
        print(f"\nAnalyzing {channel_name} (assigned to {assigned_channel})...")
        
        if np.sum(counts) == 0:
            print(f"No counts found in {channel_name}, skipping...")
            continue
        
        peak_x, peak_y, popt = analyze_highest_peak(
            volts, counts,
            window=21,
            poly=3,
            prominence=0.05,
            show_plot=show_plots,
            save_plot=True,
            save_path=output_dir,
            file_name=filename,
            channel_name=channel_name,
            voltage=voltage,
            assigned_channel=assigned_channel
        )
        
        if peak_x is not None:
            print(f"Highest peak found at: {peak_x:.4f}V with height {peak_y:.1f} counts")
            print(f"Gaussian fit parameters: amplitude={popt[0]:.2f}, mean={popt[1]:.4f}, std={popt[2]:.4f}")

if __name__ == "__main__":
    analyze_spectra_directory()
    
    # For single file analysis:
    # analyze_single_file("path/to/your/file.dat", show_plots=True)