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
    Determine if file belongs to ch0 or ch1 based on filename patterns.
    Returns 'ch0' or 'ch1'
    """
    filename_lower = filename.lower()
    name_without_ext = os.path.splitext(filename_lower)[0]
    
    if 'ch1' in filename_lower or 'channel1' in filename_lower or 'channel_1' in filename_lower:
        return 'ch1'
    elif 'ch0' in filename_lower or 'channel0' in filename_lower or 'channel_0' in filename_lower:
        return 'ch0'
    
    if re.search(r'_1[_\.]', name_without_ext) or name_without_ext.endswith('_1'):
        return 'ch1'
    elif re.search(r'_0[_\.]', name_without_ext) or name_without_ext.endswith('_0'):
        return 'ch0'
    
    match = re.search(r'(\d)$', name_without_ext)
    if match:
        digit = match.group(1)
        if digit == '1':
            return 'ch1'
        elif digit == '0':
            return 'ch0'
    
    print(f"  DEBUG: Could not determine channel for '{filename}'")
    return 'unknown'

def read_spectrum_file(filepath):
    """Read pulse height spectrum data from .dat file - supports Ch_B and Ch_D"""
    try:
        df = pd.read_csv(filepath, sep='\t', header=0)
        df.columns = df.columns.str.strip()
        
        print(f"  Available columns: {df.columns.tolist()}")
        
        # Dictionary to store channel data
        channel_data = {}
        
        # Check for Ch_B
        if 'Volts:Ch_B' in df.columns and 'Counts:Ch_B' in df.columns:
            channel_data['Ch_B'] = (df['Volts:Ch_B'].values, df['Counts:Ch_B'].values)
            print(f"  Found Ch_B data")
        
        # Check for Ch_D
        if 'Volts:Ch_D' in df.columns and 'Counts:Ch_D' in df.columns:
            channel_data['Ch_D'] = (df['Volts:Ch_D'].values, df['Counts:Ch_D'].values)
            print(f"  Found Ch_D data")
        
        # Check for Ch_B+D (summed channel)
        if 'Volts:Ch_B+D' in df.columns and 'Counts:Ch_B+D' in df.columns:
            channel_data['Ch_B+D'] = (df['Volts:Ch_B+D'].values, df['Counts:Ch_B+D'].values)
            print(f"  Found Ch_B+D data")
        
        return channel_data
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

def analyze_highest_peak(x, y, window=21, poly=3, prominence=0.05, 
                        show_plot=True, save_plot=False, save_path=None, 
                        file_name=None, voltage=None, assigned_channel=None,
                        physical_channel=None):
    """
    Finds the peak with highest y-value (tallest peak), fits Gaussian, and plots results.
    
    Parameters:
    -----------
    assigned_channel : str
        The assigned channel name ('ch0' or 'ch1') based on filename
    physical_channel : str
        The physical channel name ('Ch_B', 'Ch_D', etc.)
    save_plot : bool
        Whether to save the plot (default: False)
    save_path : str
        User-specified path for saving plots (must be provided if save_plot=True)
    
    Returns:
    --------
    peak_x : float
        X-position (voltage) of the highest peak
    peak_y : float
        Y-value (counts) of the highest peak
    popt : array
        Gaussian fit parameters [amplitude, mean, std]
    """
    
    # Check if there's any data
    if np.sum(y) == 0:
        print(f"No counts found in {physical_channel}, skipping...")
        return None, None, None
    
    # Smooth data
    y_smooth = savgol_filter(y, window_length=window, polyorder=poly)

    # Find peaks on smoothed data
    peaks, _ = find_peaks(
        y_smooth,
        height=np.max(y_smooth) * prominence,
        distance=len(y) // 20
    )

    if len(peaks) == 0:
        print(f"No peaks detected in {physical_channel}.")
        return None, None, None

    # Select peak with highest y-value (tallest peak)
    highest_y_peak_idx = peaks[np.argmax(y_smooth[peaks])]
    peak_x = x[highest_y_peak_idx]
    peak_y = y_smooth[highest_y_peak_idx]

    # Fit Gaussian around highest y peak
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
        info_text += f'Physical Channel: {physical_channel}\n'
        if assigned_channel:
            info_text += f'Assigned Channel: {assigned_channel}'
        
        plt.figtext(0.76, 0.68, info_text,
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.5", 
                                          facecolor="lightgray", alpha=0.8))
    
    plt.xlabel("Voltage Output (V)", fontsize=12)
    plt.ylabel("Counts", fontsize=12)
    
    title_suffix = f" - {assigned_channel}" if assigned_channel else ""
    plt.title(f"Highest Peak Detection ({physical_channel}): {file_name if file_name else 'Unknown File'}{title_suffix}", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot only if explicitly requested and path is provided
    if save_plot:
        if save_path is None:
            print("Warning: save_plot=True but no save_path provided. Plot not saved.")
        else:
            os.makedirs(save_path, exist_ok=True)
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            base_name = os.path.splitext(file_name)[0] if file_name else "spectrum"
            channel_suffix = f"_{assigned_channel}" if assigned_channel else ""
            plot_filename = f"{base_name}_{timestamp}{channel_suffix}_{physical_channel}_highest_peak.png"
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

def plot_voltage_vs_peak_position(voltage_data, output_directory, physical_channel='Ch_B'):
    """
    Plot applied voltage vs highest peak position - COMBINED plot for ch0 and ch1
    
    Parameters:
    -----------
    voltage_data : list of tuples
        List of (voltage, assigned_channel, physical_channel, peak_position)
    output_directory : str
        Directory to save plots
    physical_channel : str
        The physical channel to plot (e.g., 'Ch_B', 'Ch_D')
    """
    
    # Filter data for the specified physical channel
    filtered_data = [(v, ch, p) for v, ch, phys_ch, p in voltage_data if phys_ch == physical_channel]
    
    if not filtered_data:
        print(f"No data found for {physical_channel}")
        return
    
    # Separate data by assigned channel (ch0 and ch1)
    ch0_voltages = [v for v, ch, _ in filtered_data if ch == 'ch0']
    ch0_peaks = [p for v, ch, p in filtered_data if ch == 'ch0']
    
    ch1_voltages = [v for v, ch, _ in filtered_data if ch == 'ch1']
    ch1_peaks = [p for v, ch, p in filtered_data if ch == 'ch1']
    
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create COMBINED plot
    plt.figure(figsize=(12, 8))
    
    # Plot ch0 data if available
    if ch0_voltages:
        ch0_voltages = np.array(ch0_voltages)
        ch0_peaks = np.array(ch0_peaks)
        
        sort_idx_0 = np.argsort(ch0_voltages)
        ch0_voltages = ch0_voltages[sort_idx_0]
        ch0_peaks = ch0_peaks[sort_idx_0]
        
        fit_0 = np.polyfit(ch0_voltages, ch0_peaks, 1)
        line_0 = np.poly1d(fit_0)
        
        plt.scatter(ch0_voltages, ch0_peaks, color='blue', s=100, alpha=0.7, 
                   edgecolors='black', linewidth=1.5, label='ch0 data')
        plt.plot(ch0_voltages, line_0(ch0_voltages), color='darkblue', linestyle='--', linewidth=2,
                label=f'ch0 fit: y = {fit_0[0]:.6f}x + {fit_0[1]:.6f}')
        
        print(f"\n{physical_channel} - ch0 - Slope: {fit_0[0]:.6f}, Intercept: {fit_0[1]:.6f}")
    else:
        print(f"\n{physical_channel} - No ch0 data found.")
    
    # Plot ch1 data if available
    if ch1_voltages:
        ch1_voltages = np.array(ch1_voltages)
        ch1_peaks = np.array(ch1_peaks)
        
        sort_idx_1 = np.argsort(ch1_voltages)
        ch1_voltages = ch1_voltages[sort_idx_1]
        ch1_peaks = ch1_peaks[sort_idx_1]
        
        fit_1 = np.polyfit(ch1_voltages, ch1_peaks, 1)
        line_1 = np.poly1d(fit_1)
        
        plt.scatter(ch1_voltages, ch1_peaks, color='red', s=100, alpha=0.7, 
                   edgecolors='black', linewidth=1.5, label='ch1 data')
        plt.plot(ch1_voltages, line_1(ch1_voltages), color='darkred', linestyle='--', linewidth=2,
                label=f'ch1 fit: y = {fit_1[0]:.6f}x + {fit_1[1]:.6f}')
        
        print(f"\n{physical_channel} - ch1 - Slope: {fit_1[0]:.6f}, Intercept: {fit_1[1]:.6f}")
        
        # Calculate gain matching information if both channels present
        if ch0_voltages.size > 0:
            ratio = fit_1[0] / fit_0[0]
            print(f"\n{physical_channel} - Gain Ratio (ch1/ch0): {ratio:.4f}")
            print(f"{physical_channel} - Gain difference: {((ratio - 1) * 100):.2f}%")
    else:
        print(f"\n{physical_channel} - No ch1 data found.")
    
    # Finalize combined plot
    plt.title(f'Applied Voltage vs Highest Peak Position - {physical_channel} - Gain Matching', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Applied Voltage (V)', fontsize=12)
    plt.ylabel('Peak Position (V)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save combined plot
    summary_filename = f"voltage_vs_peak_position_{physical_channel}_combined_{current_datetime}.png"
    summary_path = os.path.join(output_directory, summary_filename)
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot: {summary_filename}")
    
    plt.show()
    plt.close()

def analyze_spectra_directory(save_individual_plots=False, individual_plot_path=None, 
                             target_channels=['Ch_B']):
    """
    Main function to analyze all .dat files in selected directory
    
    Parameters:
    -----------
    save_individual_plots : bool
        Whether to save individual peak fitting plots (default: False)
    individual_plot_path : str
        User-specified path for saving individual plots (required if save_individual_plots=True)
    target_channels : list of str
        List of physical channels to analyze (e.g., ['Ch_B', 'Ch_D'])
        Default: ['Ch_B']
    """
    
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
    print(f"Target channels for analysis: {target_channels}")
    print(f"Saving summary plots to: {output_directory}")
    
    # Debug: Print all filenames and their channel assignments
    print("\n=== CHANNEL ASSIGNMENT DEBUG ===")
    for filepath in dat_files:
        filename = os.path.basename(filepath)
        assigned_channel = determine_channel_from_filename(filename)
        print(f"{filename} -> {assigned_channel}")
    print("=== END DEBUG ===\n")
    
    if save_individual_plots:
        if individual_plot_path:
            print(f"Saving individual plots to: {individual_plot_path}")
        else:
            print("Warning: save_individual_plots=True but no path provided. Individual plots will not be saved.")
    
    # Store data as: (voltage, assigned_channel, physical_channel, peak_position)
    voltage_peak_data = []
    
    for filepath in dat_files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing {filename}...")
        
        # Determine channel assignment from filename
        assigned_channel = determine_channel_from_filename(filename)
        print(f"Assigned to: {assigned_channel}")
        
        if assigned_channel == 'unknown':
            print(f"Warning: Could not determine channel (ch0/ch1) from filename. Skipping.")
            continue
        
        voltage = extract_voltage_from_filename(filename)
        if voltage is None:
            print(f"Warning: Could not extract voltage from filename {filename}")
            continue
        
        print(f"Extracted voltage: {voltage}")
        channel_data = read_spectrum_file(filepath)
        
        if not channel_data:
            continue
        
        # Analyze each target channel
        for physical_channel in target_channels:
            if physical_channel not in channel_data:
                print(f"  {physical_channel} not found in file, skipping...")
                continue
            
            volts, counts = channel_data[physical_channel]
            print(f"  Analyzing {physical_channel} (assigned to {assigned_channel})...")
            
            peak_x, peak_y, popt = analyze_highest_peak(
                volts, counts,
                window=21,
                poly=3,
                prominence=0.05,
                show_plot=False,  # Don't show individual plots by default
                save_plot=save_individual_plots,
                save_path=individual_plot_path,
                file_name=filename,
                voltage=voltage,
                assigned_channel=assigned_channel,
                physical_channel=physical_channel
            )
            
            if peak_x is not None:
                print(f"  {physical_channel}: Highest peak found at {peak_x:.4f}V with height {peak_y:.1f} counts")
                # Store with both assigned channel and physical channel
                voltage_peak_data.append((voltage, assigned_channel, physical_channel, peak_x))
            else:
                print(f"  {physical_channel}: No peaks found")
    
    # Create summary plots for each physical channel
    if voltage_peak_data:
        print(f"\nCreating summary plots with {len(voltage_peak_data)} total data points...")
        
        # Get unique physical channels from the data
        unique_physical_channels = list(set([phys_ch for _, _, phys_ch, _ in voltage_peak_data]))
        
        for physical_channel in unique_physical_channels:
            print(f"\nCreating plot for {physical_channel}...")
            plot_voltage_vs_peak_position(voltage_peak_data, output_directory, physical_channel)
        
        print("\nSummary of collected data:")
        for voltage, assigned_ch, physical_ch, peak_pos in voltage_peak_data:
            print(f"Voltage: {voltage}V, Assigned: {assigned_ch}, Physical: {physical_ch}, Peak: {peak_pos:.4f}V")
    else:
        print("No data collected for summary plots")
    
    print(f"\nAnalysis complete! Summary plots saved to: {output_directory}")

def analyze_single_file(filepath, show_plots=True, save_plots=False, save_path=None,
                       target_channels=['Ch_B']):
    """
    Analyze a single .dat file
    
    Parameters:
    -----------
    filepath : str
        Path to the .dat file
    show_plots : bool
        Whether to display plots (default: True)
    save_plots : bool
        Whether to save plots (default: False)
    save_path : str
        User-specified path for saving plots (required if save_plots=True)
    target_channels : list of str
        List of physical channels to analyze (e.g., ['Ch_B', 'Ch_D'])
    """
    
    filename = os.path.basename(filepath)
    print(f"Processing {filename}...")
    
    # Determine channel assignment from filename
    assigned_channel = determine_channel_from_filename(filename)
    print(f"Assigned to: {assigned_channel}")
    
    voltage = extract_voltage_from_filename(filename)
    print(f"Extracted voltage: {voltage}")
    
    channel_data = read_spectrum_file(filepath)
    
    if not channel_data:
        print("Failed to read channels from file")
        return
    
    # Analyze each target channel
    for physical_channel in target_channels:
        if physical_channel not in channel_data:
            print(f"{physical_channel} not found in file, skipping...")
            continue
        
        volts, counts = channel_data[physical_channel]
        print(f"\nAnalyzing {physical_channel} (assigned to {assigned_channel})...")
        
        peak_x, peak_y, popt = analyze_highest_peak(
            volts, counts,
            window=21,
            poly=3,
            prominence=0.05,
            show_plot=show_plots,
            save_plot=save_plots,
            save_path=save_path,
            file_name=filename,
            voltage=voltage,
            assigned_channel=assigned_channel,
            physical_channel=physical_channel
        )
        
        if peak_x is not None:
            print(f"{physical_channel}: Highest peak found at {peak_x:.4f}V with height {peak_y:.1f} counts")
            print(f"Gaussian fit parameters: amplitude={popt[0]:.2f}, mean={popt[1]:.4f}, std={popt[2]:.4f}")

if __name__ == "__main__":
    # Analyze directory with Ch_B only (original behavior)
    # analyze_spectra_directory(target_channels=['Ch_B'])
    
    # Analyze directory with both Ch_B and Ch_D (creates separate plots for each)
    analyze_spectra_directory(
        save_individual_plots=False, 
        individual_plot_path=r"\\isis\shares\Detectors\Lisa Malliolio 2025\PMT_calibration_20251001\ch0_n_ch1_for_bens_code",
        target_channels=['Ch_B', 'Ch_D']  # Specify which physical channels to analyze
    )
    
    # For Ch_B+D summed channel:
    # analyze_spectra_directory(target_channels=['Ch_B+D'])
    
    # For all channels:
    # analyze_spectra_directory(target_channels=['Ch_B', 'Ch_D', 'Ch_B+D'])
    
    # For single file analysis:
    # analyze_single_file("path/to/your/file.dat", show_plots=True, target_channels=['Ch_B', 'Ch_D'])