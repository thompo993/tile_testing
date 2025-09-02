import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import tkinter as tk
from tkinter import filedialog
import glob
import pandas as pd
from datetime import datetime
import re

def select_directory():
    """Open a dialog to select directory containing .dat files"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    directory = filedialog.askdirectory(
        title="Select directory containing .dat files"
    )
    
    root.destroy()
    return directory

def get_output_directory():
    """Get the fixed output directory for saving plots"""
    output_dir = r"tile_testing\Figures"
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    return output_dir

def extract_voltage_from_filename(filename):
    """Extract the first numerical value from the filename as voltage"""
    # Find all numerical values in the filename
    numbers = re.findall(r'\d+\.?\d*', filename)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None

def smooth_data(counts, method='savgol', window_length=21, polyorder=3):
    """Apply smoothing to the counts data before peak finding
    
    Parameters:
    -----------
    counts : array
        The count data to smooth
    method : str
        Smoothing method ('savgol' or 'moving_average')
    window_length : int
        Length of the smoothing window (must be odd for savgol)
    polyorder : int
        Polynomial order for Savitzky-Golay filter
    
    Returns:
    --------
    smoothed_counts : array
        Smoothed count data
    """
    
    if method == 'savgol':
        # Ensure window_length is odd and not larger than data length
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, len(counts))
        if window_length < polyorder + 1:
            window_length = polyorder + 1
            if window_length % 2 == 0:
                window_length += 1
        
        smoothed = savgol_filter(counts, window_length, polyorder)
        
    elif method == 'moving_average':
        # Simple moving average
        smoothed = np.convolve(counts, np.ones(window_length)/window_length, mode='same')
    
    else:
        raise ValueError("Method must be 'savgol' or 'moving_average'")
    
    return smoothed

def read_spectrum_file(filepath):
    """Read pulse height spectrum data from .dat file
    
    Handles the specific format with columns:
    Volts:Ch_A, Counts:Ch_A, Volts:Ch_B, Counts:Ch_B
    """
    try:
        # Read the file with pandas to handle the tab-separated format
        # Skip the first row if it contains headers
        df = pd.read_csv(filepath, sep='\t', header=0)
        
        # Clean up column names (remove extra whitespace)
        df.columns = df.columns.str.strip()
        
        # Extract data for both channels
        channels_data = {}
        
        # Check if we have Channel A data
        if 'Volts:Ch_A' in df.columns and 'Counts:Ch_A' in df.columns:
            volts_a = df['Volts:Ch_A'].values
            counts_a = df['Counts:Ch_A'].values
            channels_data['Ch_A'] = (volts_a, counts_a)
        
        # Check if we have Channel B data
        if 'Volts:Ch_B' in df.columns and 'Counts:Ch_B' in df.columns:
            volts_b = df['Volts:Ch_B'].values
            counts_b = df['Counts:Ch_B'].values
            channels_data['Ch_B'] = (volts_b, counts_b)
        
        return channels_data
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def find_peak_distances(channels, counts, prominence=None, height=None, distance=None, 
                       apply_smoothing=False, smooth_method='savgol', window_length=21, polyorder=3):
    """Find peaks in the spectrum and calculate distances between them
    
    Parameters:
    -----------
    apply_smoothing : bool
        Whether to apply smoothing before peak finding
    smooth_method : str
        Smoothing method ('savgol' or 'moving_average')
    window_length : int
        Length of the smoothing window
    polyorder : int
        Polynomial order for Savitzky-Golay filter
    """
    
    # Apply smoothing if requested
    if apply_smoothing:
        smoothed_counts = smooth_data(counts, smooth_method, window_length, polyorder)
        peak_finding_data = smoothed_counts
    else:
        smoothed_counts = None
        peak_finding_data = counts
    
    # Find peaks using scipy
    peaks, properties = find_peaks(
        peak_finding_data, 
        prominence=prominence,
        height=height,
        distance=distance
    )
    
    # Get peak positions (in channel units)
    peak_channels = channels[peaks]
    peak_counts = counts[peaks]  # Always use original counts for display
    
    # Calculate distances between consecutive peaks
    peak_distances = []
    if len(peak_channels) > 1:
        peak_distances = np.diff(peak_channels)
    
    return peaks, peak_channels, peak_counts, peak_distances, properties, smoothed_counts

def mean_peak_distance(peak_distances):
    """Calculate the mean distance between peaks"""
    if len(peak_distances) == 0:
        return 0.0
    return np.mean(peak_distances)

def plot_spectrum_with_peaks(channels, counts, peaks, peak_channels, peak_counts, 
                           peak_distances, filename, channel_name, smoothed_counts=None):
    """Plot the spectrum with identified peaks and distance annotations
    
    Parameters:
    -----------
    smoothed_counts : array or None
        Smoothed data to overlay on plot (if smoothing was used)
    """
    
    plt.figure(figsize=(12, 8))
    
    # Plot original spectrum
    plt.plot(channels, counts, 'b-', linewidth=1, label=f'Original Spectrum {channel_name}', alpha=0.7)
    
    # Plot smoothed spectrum if available
    if smoothed_counts is not None:
        plt.plot(channels, smoothed_counts, 'g-', linewidth=2, label=f'Smoothed Spectrum {channel_name}')
    
    # Plot peaks
    plt.plot(peak_channels, peak_counts, 'ro', markersize=8, label='Peaks')
    
    average_separation = np.mean(peak_distances) if len(peak_distances) > 0 else 0
    plt.plot([], [], ' ', label=f'Mean ΔV = {average_separation:.6f}')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Counts')
    title = f'Pulse Height Spectrum: {filename} - {channel_name}'
    if smoothed_counts is not None:
        title += ' (with Smoothing)'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_voltage_vs_peak_separation(voltage_data, separation_data, output_directory):
    """Plot voltage vs peak separation with best fit lines for each channel"""
    
    # Separate data by channel
    ch_a_voltages = [v for v, ch, _ in voltage_data if ch == 'Ch_A']
    ch_a_separations = [s for v, ch, s in voltage_data if ch == 'Ch_A']
    
    ch_b_voltages = [v for v, ch, _ in voltage_data if ch == 'Ch_B']
    ch_b_separations = [s for v, ch, s in voltage_data if ch == 'Ch_B']
    
    plt.figure(figsize=(10, 6))
    
    # Plot Channel A data if available
    if ch_a_voltages:
        ch_a_voltages = np.array(ch_a_voltages)
        ch_a_separations = np.array(ch_a_separations)
        
        # Sort by voltage for cleaner lines
        sort_idx_a = np.argsort(ch_a_voltages)
        ch_a_voltages = ch_a_voltages[sort_idx_a]
        ch_a_separations = ch_a_separations[sort_idx_a]
        
        # Line of best fit
        fit_a = np.polyfit(ch_a_voltages, ch_a_separations, 1)
        line_a = np.poly1d(fit_a)
        
        plt.scatter(ch_a_voltages, ch_a_separations, color='blue', label='Channel A', s=50)
        plt.plot(ch_a_voltages, line_a(ch_a_voltages), color='blue', linestyle='--', 
                label=f'Ch A Best fit (slope: {fit_a[0]:.6f})')
        
        print(f"Channel A - Slope: {fit_a[0]:.6f}, Intercept: {fit_a[1]:.6f}")
    
    # Plot Channel B data if available
    if ch_b_voltages:
        ch_b_voltages = np.array(ch_b_voltages)
        ch_b_separations = np.array(ch_b_separations)
        
        # Sort by voltage for cleaner lines
        sort_idx_b = np.argsort(ch_b_voltages)
        ch_b_voltages = ch_b_voltages[sort_idx_b]
        ch_b_separations = ch_b_separations[sort_idx_b]
        
        # Line of best fit
        fit_b = np.polyfit(ch_b_voltages, ch_b_separations, 1)
        line_b = np.poly1d(fit_b)
        
        plt.scatter(ch_b_voltages, ch_b_separations, color='red', label='Channel B', s=50)
        plt.plot(ch_b_voltages, line_b(ch_b_voltages), color='red', linestyle='--', 
                label=f'Ch B Best fit (slope: {fit_b[0]:.6f})')
        
        print(f"Channel B - Slope: {fit_b[0]:.6f}, Intercept: {fit_b[1]:.6f}")
    
    plt.title('Voltage vs Average Peak Separation')
    plt.xlabel('Voltage')
    plt.ylabel('Average Peak Separation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the summary plot
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"voltage_vs_peak_separation_{current_datetime}.png"
    summary_path = os.path.join(output_directory, summary_filename)
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {summary_filename}")
    
    plt.show()
    plt.close()

def analyze_spectra_directory():
    """Main function to analyze all .dat files in selected directory"""
    
    # Select input directory
    input_directory = select_directory()
    if not input_directory:
        print("No input directory selected. Exiting.")
        return
    
    # Get fixed output directory
    output_directory = get_output_directory()
    
    # Find all .dat files
    dat_files = glob.glob(os.path.join(input_directory, "*.dat"))
    
    if not dat_files:
        print(f"No .dat files found in {input_directory}")
        return
    
    print(f"Found {len(dat_files)} .dat files")
    print(f"Saving all plots to: {output_directory}")
    
    # Store voltage and peak separation data
    voltage_separation_data = []
    
    # Process each file
    for filepath in dat_files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing {filename}...")
        
        # Extract voltage from filename
        voltage = extract_voltage_from_filename(filename)
        if voltage is None:
            print(f"Warning: Could not extract voltage from filename {filename}")
            continue
        
        print(f"Extracted voltage: {voltage}")
        
        # Read spectrum data
        channels_data = read_spectrum_file(filepath)
        
        if channels_data is None:
            continue
        
        # Process each channel in the file
        for channel_name, (volts, counts) in channels_data.items():
            print(f"\nAnalyzing {channel_name}...")
            
            # Skip if all counts are zero
            if np.sum(counts) == 0:
                print(f"No counts found in {channel_name}, skipping...")
                continue
            
            # Find peaks with Savitzky-Golay smoothing
            peaks, peak_channels, peak_counts, peak_distances, properties, smoothed_counts = find_peak_distances(
                volts, counts,
                prominence=np.max(counts) * 0.05,
                height=np.max(counts) * 0.02,
                distance=10,
                apply_smoothing=True,
                smooth_method='savgol',
                window_length=21,
                polyorder=3
            )
            
            print(f"Found {len(peaks)} peaks in {channel_name}")
            if len(peak_channels) > 0:
                print("Peak positions (V):", [f"{v:.6f}" for v in peak_channels])
            if len(peak_distances) > 0:
                print("Peak distances (V):", [f"{dist:.6f}" for dist in peak_distances])
                avg_separation = mean_peak_distance(peak_distances)
                print(f"Average peak separation: {avg_separation:.6f} V")
                
                # Store the data for summary plot
                voltage_separation_data.append((voltage, channel_name, avg_separation))
            else:
                print("No peak distances to calculate")
            
            # Create plot
            fig = plot_spectrum_with_peaks(
                volts, counts, peaks, peak_channels, peak_counts, 
                peak_distances, filename, channel_name, smoothed_counts
            )
            
            # Save plot directly to tile_testing\Figures
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = os.path.splitext(filename)[0]
            output_filename = f"{base_filename}_{channel_name}_peaks_{current_datetime}.png"
            output_path = os.path.join(output_directory, output_filename)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {output_filename}")
            
            # Show plot (comment out if you don't want to display)
            plt.show()
            
            # Close figure to free memory
            plt.close(fig)
    
    # Create summary plot of voltage vs peak separation
    if voltage_separation_data:
        print(f"\nCreating summary plot with {len(voltage_separation_data)} data points...")
        plot_voltage_vs_peak_separation(voltage_separation_data, None, output_directory)
        
        # Print summary data
        print("\nSummary of collected data:")
        for voltage, channel, separation in voltage_separation_data:
            print(f"Voltage: {voltage}, Channel: {channel}, Avg Peak Separation: {separation:.6f}")
    else:
        print("No data collected for summary plot")
    
    print(f"\nAnalysis complete! All plots saved to: {output_directory}")

def analyze_single_file(filepath, show_plots=True, apply_smoothing=False):
    """Analyze a single .dat file - useful for testing
    
    Parameters:
    -----------
    apply_smoothing : bool
        Whether to apply smoothing before peak finding
    """
    
    filename = os.path.basename(filepath)
    print(f"Processing {filename}...")
    
    # Extract voltage from filename
    voltage = extract_voltage_from_filename(filename)
    print(f"Extracted voltage: {voltage}")
    
    # Read spectrum data
    channels_data = read_spectrum_file(filepath)
    
    if channels_data is None:
        print("Failed to read file")
        return
    
    # Get output directory
    output_dir = get_output_directory()
    
    # Process each channel in the file
    for channel_name, (volts, counts) in channels_data.items():
        print(f"\nAnalyzing {channel_name}...")
        
        # Skip if all counts are zero
        if np.sum(counts) == 0:
            print(f"No counts found in {channel_name}, skipping...")
            continue
        
        # Find peaks
        peaks, peak_channels, peak_counts, peak_distances, properties, smoothed_counts = find_peak_distances(
            volts, counts,
            prominence=np.max(counts) * 0.05,
            height=np.max(counts) * 0.02,
            distance=10,
            apply_smoothing=apply_smoothing,
            smooth_method='savgol',
            window_length=21,
            polyorder=3
        )
        
        print(f"Found {len(peaks)} peaks in {channel_name}")
        if len(peak_channels) > 0:
            print("Peak positions (V):", [f"{v:.6f}" for v in peak_channels])
        if len(peak_distances) > 0:
            print("Peak distances (V):", [f"{dist:.6f}" for dist in peak_distances])
            print(f"Average peak separation: {mean_peak_distance(peak_distances):.6f} V")
        
        # Create plot
        fig = plot_spectrum_with_peaks(
            volts, counts, peaks, peak_channels, peak_counts, 
            peak_distances, filename, channel_name, smoothed_counts
        )
        
        # Save plot to tile_testing\Figures
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(filename)[0]
        output_filename = f"{base_filename}_{channel_name}_peaks_{current_datetime}.png"
        output_path = os.path.join(output_dir, output_filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
        
        # Show plot
        if show_plots:
            plt.show()
        
        # Close figure to free memory
        plt.close(fig)

if __name__ == "__main__":
    # Run the batch analysis
    analyze_spectra_directory()
    
    # Examples of single file analysis:
    """
    # For single file analysis without smoothing:
    analyze_single_file("path/to/your/file.dat", show_plots=True)
    
    # For single file analysis with smoothing:
    analyze_single_file("path/to/your/file.dat", show_plots=True, apply_smoothing=True)
    """