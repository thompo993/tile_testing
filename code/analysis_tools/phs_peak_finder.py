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
# gaussian function
# ------------------------
def gaussian(x, A, mu, sigma):
    """
    standard gaussian function
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# ------------------------
# read .set file for Runtime and StartDateTime
# ------------------------
def read_set_file(data_file_path):
    """
    Read the associated .set file and extract Runtime, StartDateTime, and Integration settings
    """
    # Get the .set file path by changing the extension
    set_file_path = Path(data_file_path).with_suffix('.set')
    
    runtime = None
    start_datetime = None
    integration_time = None
    is_integration_enabled = None
    division_1 = None
    division_3 = None
    
    if set_file_path.exists():
        try:
            with open(set_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('RunTime='):
                        runtime = line.split('=')[1]
                    elif line.startswith('StartDateTime='):
                        start_datetime = line.split('=')[1]
                    elif line.startswith('IntegrationTime='):
                        integration_time = line.split('=')[1]
                    elif line.startswith('ChanFullScaleRange[1]'):
                        division_1 = line.split('=')[1]
                    elif line.startswith('ChanFullScaleRange[3]'):
                        division_3 = line.split('=')[1]
                    elif line.startswith('IsIntegrationEnabled='):
                        is_integration_enabled = line.split('=')[1].lower() == 'true'
                    
        except Exception as e:
            print(f"Error reading .set file {set_file_path}: {e}")
    else:
        print(f"No .set file found for {Path(data_file_path).name}")
    
    # assert that divisions are the same: 
    assert division_1 == division_3, "Divisions are not equal"
    #making variable names easier to read, making it an float
    division = float(division_1)


    return runtime, start_datetime, integration_time, is_integration_enabled, division



# ------------------------
# Format integration info for display
# ------------------------
def format_integration_info(integration_time, is_integration_enabled):
    """
    Format integration information for display in the info text
    """
    if is_integration_enabled is None:
        return "Integration: Not specified"
    elif not is_integration_enabled:
        return "Integration: OFF"
    elif integration_time is not None:
        # Try to format scientific notation nicely
        try:
            time_value = float(integration_time)
            if time_value >= 1:
                return f"Integration: {time_value:.3f}s"
            else:
                return f"Integration: {time_value:.2e}s"
        except (ValueError, TypeError):
            return f"Integration: {integration_time}"
    else:
        return "Integration: ON (time not specified)"

# ------------------------
# Parse runtime to seconds
# ------------------------
def parse_runtime_to_seconds(runtime_str):
    """
    Parse runtime string to seconds for normalisation
    Supports formats like 'HH:MM:SS' or just seconds as string
    """
    if runtime_str is None:
        return None
    
    try:
        # If it contains colons, assume HH:MM:SS format
        if ':' in runtime_str:
            parts = runtime_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = map(float, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:
                minutes, seconds = map(float, parts)
                return minutes * 60 + seconds
        else:
            # Assume it's already in seconds
            return float(runtime_str)
    except (ValueError, TypeError):
        print(f"Could not parse runtime: {runtime_str}")
        return None

# ------------------------
# Extract channel names from header
# ------------------------
def extract_channel_names(header_line):
    """
    Extract channel names from the header line
    Expected format: "Volts:Ch_A	Counts:Ch_A		Volts:Ch_C	Counts:Ch_C		Volts:Ch_A+C	Counts:Ch_A+C"
    """
    channel_names = []
    parts = header_line.split('\t')
    
    for part in parts:
        part = part.strip()
        if part.startswith('Counts:'):
            # Extract channel name after "Counts:"
            channel_name = part.replace('Counts:', '')
            channel_names.append(channel_name)
    
    return channel_names

# ------------------------
# Load PHS data (modified for multi-channel)
# ------------------------
def load_phs_file(file_path, multi_channel=False):
    try:
        file_ext = Path(file_path).suffix.lower()

        if file_ext in ['.txt', '.dat']:

                data = pd.read_csv(file_path, sep="\t", header=0).dropna(axis=1, how="all")
                
        else:
            print(f"Unsupported file format: {file_ext}")

        if not multi_channel:
            # Original single channel behavior
            if data.shape[1] >= 2:
                x = data.iloc[:, 0].values
                y = data.iloc[:, 1].values
                valid_mask = ~(np.isnan(x) | np.isnan(y))
                return x[valid_mask], y[valid_mask]
            else:
                print(f"Warning: File {file_path} doesn't have at least 2 columns")
                return None, None
        else:
            # Multi-channel behavior
            # Extract channel names from header
            header_line = None
            try:
                with open(file_path, 'r') as f:
                    header_line = f.readline().strip()
            except:
                header_line = '\t'.join(data.columns)
            
            channel_names = extract_channel_names(header_line)
            
            if not channel_names:
                print(f"Warning: No channel names found in {file_path}")
                return None, None
            
            # Extract data for each channel (pairs of voltage/counts columns)
            channels_data = {}
            col_idx = 0
            
            for channel_name in channel_names:
                if col_idx + 1 < data.shape[1]:
                    x = data.iloc[:, col_idx].values      # Voltage column
                    y = data.iloc[:, col_idx + 1].values  # Counts column
                    
                    # Remove invalid data points
                    valid_mask = ~(np.isnan(x) | np.isnan(y))
                    channels_data[channel_name] = {
                        'x': x[valid_mask],
                        'y': y[valid_mask]
                    }
                    col_idx += 2  # Move to next channel pair
                else:
                    break
       
            return channels_data, channel_names
            
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        if multi_channel:
            return None, None
        else:
            return None, None

# ------------------------
# Analyze highest X peak in one file
# ------------------------
def analyze_largest_peak(x, y, window=10, poly=3, prominence=0.05,
                         show_plot=True, save_plot=False, save_path=None, file_name=None,
                         runtime=None, start_datetime=None, integration_time=None, 
                         is_integration_enabled=None, normalise=True, channel_name=None, division = 1):
    """
    Smooths data, finds peak with highest x-value, fits Gaussian, and optionally plots/saves results.
    """
    # Parse runtime for normalisation
    runtime_seconds = parse_runtime_to_seconds(runtime) if runtime else None
    
    # normalise data if requested and runtime is available
    y_original = y.copy()
    if normalise and runtime_seconds and runtime_seconds > 0:
        # Normalise counts by runtime
        y = y / runtime_seconds
        y_label = "Counts/second"
        normalisation_note_runtime = f"normalised by runtime ({runtime}s)"
        #Normalise by bins using division:
        
    else:
        y_label = "Counts"
        normalisation_note_runtime = "Raw counts (no normalisation)" if not normalise else "Raw counts (runtime unavailable)"
    
    # Smooth data (using potentially normalised y)
    y_smooth = savgol_filter(y, window_length=window, polyorder=poly)

    # Find peaks on smoothed data
    peaks, _ = find_peaks(
        y_smooth,
        height=np.max(y_smooth) * prominence,
        distance=len(y) // 20
    )

    if len(peaks) == 0:
        print(f"No peaks detected in {channel_name if channel_name else 'data'}.")
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
    
    # Plot original (potentially normalised) data
    if normalise and runtime_seconds:
        plt.plot(x, y, label="normalised Spectrum", color="lightblue", alpha=0.7, linewidth=1.5)
    else:
        plt.plot(x, y, label="Raw Spectrum", color="lightgray", alpha=0.7)
    
    plt.plot(x, y_smooth, label="Smoothed Spectrum", color="blue", linewidth=2)
    plt.plot(x[peaks], y_smooth[peaks], "ro", markersize=8, label="Detected Peaks")
    plt.plot(x_fit, gaussian(x_fit, *popt), "g--", linewidth=3,
             label="Gaussian Fit (highest x peak)")
    
    # Create info text for the plot with integration information
    integration_info = format_integration_info(integration_time, is_integration_enabled)
    info_text = f'Runtime: {runtime}\n'
    info_text += f'mV Per Division {division*100}\n'
    info_text += f'{normalisation_note_runtime}\n'
    info_text += f'Start DateTime: {start_datetime}\n'
    info_text += integration_info
   
    if channel_name:
        info_text += f'\nChannel: {channel_name}'
    
    plt.figtext(0.76, 0.71, info_text,
                fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.axvline(peak_x, color="purple", linestyle="--", linewidth=2, 
                label=f"Peak X = {peak_x:.4f}")
    plt.xlabel("Voltage Output", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    title_suffix = f" - {channel_name}" if channel_name else ""
    plt.title(f"Highest X Peak Detection: {file_name if file_name else 'Unknown File'}{title_suffix}", 
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
        norm_suffix = "_normalised" if normalise and runtime_seconds else "_raw"
        channel_suffix = f"_{channel_name}" if channel_name else ""
        plot_filename = f"{base_name}_{timestamp}{norm_suffix}{channel_suffix}_peak_plot.png"
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
# Create overlay plot of all spectra
# ------------------------
def create_phs_overlay(spectra_data, save_path=None, normalise=True):
    """
    Create an overlay plot of all PHS spectra
    
    Parameters:
    -----------
    spectra_data : list of dict
        List containing dictionaries with keys: 'x', 'y', 'filename', 'runtime'
    save_path : str
        Path to save the overlay plot
    normalise : bool
        Whether the data was normalised
    """
    if not spectra_data:
        print("No spectra data available for overlay plot.")
        return
    
    # Create the overlay plot
    plt.figure(figsize=(14, 10))
    
    # Define a color map for different files
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(spectra_data), 10)))
    if len(spectra_data) > 10:
        # If more than 10 files, use a continuous colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(spectra_data)))
    
    for i, spectrum in enumerate(spectra_data):
        x = spectrum['x']
        y = spectrum['y']
        filename = spectrum['filename']
        runtime = spectrum['runtime']
        channel = spectrum.get('channel', '')
        
        # Use different line styles for better distinction if many files
        linestyle = '-' if len(spectra_data) <= 10 else '-'
        alpha = 0.7 if len(spectra_data) <= 5 else 0.6
        linewidth = 1.5 if len(spectra_data) <= 10 else 1.0
        
        label = f"{filename}" + (f" - {channel}" if channel else "")
        plt.plot(x, y, color=colors[i], alpha=alpha, linewidth=linewidth,
                linestyle=linestyle, label=label)
    
    # Set labels and title
    y_label = "Counts/second" if normalise else "Counts"
    plt.xlabel("Voltage Output", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    title = "PHS Spectra Overlay - "
    title += "Normalised by Runtime" if normalise else "Raw Counts"
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add legend (handle many files gracefully)
    if len(spectra_data) <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        # For many files, add a simplified legend or note
        plt.figtext(0.02, 0.98, f"Showing {len(spectra_data)} spectra", 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                   verticalalignment='top')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save overlay plot if requested
    if save_path:
        # Ensure the save path exists
        os.makedirs(save_path, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        norm_suffix = "_normalised" if normalise else "_raw"
        overlay_filename = f"PHS_Spectra_Overlay_{timestamp}{norm_suffix}.png"
        full_overlay_path = os.path.join(save_path, overlay_filename)
        
        try:
            plt.savefig(full_overlay_path, dpi=300, bbox_inches='tight')
            print(f"Overlay plot saved to: {full_overlay_path}")
        except Exception as e:
            print(f"Error saving overlay plot: {e}")
    
    plt.show()

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
# Process all files in folder (modified for multi-channel)
# ------------------------
def process_phs_folder(folder_path, save_results=True, save_plots=False, 
                       custom_save_path=None, normalise=True, phs_overlay=False, multi_channel=False):
    """
    Process all PHS files in a folder.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing PHS data files
    save_results : bool
        Whether to save results CSV
    save_plots : bool  
        Whether to save individual plots
    custom_save_path : str
        Custom path for saving outputs
    normalise : bool
        Whether to normalise counts by runtime (default: True)
    phs_overlay : bool
        Whether to create an overlay plot of all spectra (default: False)
    multi_channel : bool
        Whether to process multiple channels (default: False)
    """
    files = find_phs_files(folder_path)
    if not files:
        print("No valid PHS data files found.")
        return

    results = []
    spectra_data = []  # Store data for overlay plot
    
    print(f"Found {len(files)} files to analyze.")
    print(f"Multi-channel: {'ON' if multi_channel else 'OFF'}")
    print(f"normalisation: {'ON' if normalise else 'OFF'}")
    print(f"PHS Overlay: {'ON' if phs_overlay else 'OFF'}")
    print("")

    # Use custom save path if provided, else default to folder_path
    save_path = custom_save_path if custom_save_path else folder_path
    
    # Ensure the save path exists
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        if save_plots:
            print(f"Plots will be saved to: {save_path}")

    for i, file in enumerate(files, 1):
        print(f"Processing {i}/{len(files)}: {Path(file).name}")
        
        if not multi_channel:
            # Original single channel processing
            x, y = load_phs_file(file, multi_channel=False)
            if x is None or y is None:
                print(f"Skipping file: {file}")
                continue

            # Read associated .set file
            runtime, start_datetime, integration_time, is_integration_enabled, divison = read_set_file(file)

            # Store original data for overlay (before any processing)
            y_original = y.copy()
            runtime_seconds = parse_runtime_to_seconds(runtime) if runtime else None
            
            # Apply normalisation for overlay data if requested
            if phs_overlay:
                if normalise and runtime_seconds and runtime_seconds > 0:
                    y_overlay = y_original / runtime_seconds
                else:
                    y_overlay = y_original.copy()
                
                spectra_data.append({
                    'x': x.copy(),
                    'y': y_overlay,
                    'filename': Path(file).name,
                    'runtime': runtime
                })

            peaks, peak_x, peak_y, popt = analyze_largest_peak(
                x, y,
                show_plot=True,
                save_plot=save_plots,
                save_path=save_path,
                file_name=Path(file).name,
                runtime=runtime,
                start_datetime=start_datetime,
                integration_time=integration_time,
                is_integration_enabled=is_integration_enabled,
                normalise=normalise,
                division=divison
            )

            if peak_x is None:
                print(f"No peaks found in {file}")
                continue

            # Store results
            result = {
                "File": Path(file).name,
                "Highest_X_Peak_X": peak_x,
                "Highest_X_Peak_Y": peak_y,
                "Gaussian_A": popt[0],
                "Gaussian_Mu": popt[1],
                "Gaussian_Sigma": popt[2],
                "Runtime": runtime,
                "StartDateTime": start_datetime,
                "IntegrationTime": integration_time,
                "IsIntegrationEnabled": is_integration_enabled,
                "normalised": normalise and parse_runtime_to_seconds(runtime) is not None
            }
            results.append(result)
            
            # Print status
            norm_status = " (normalised)" if result["normalised"] else " (raw)"
            integration_info = format_integration_info(integration_time, is_integration_enabled)
            print(f"Peak found at X = {peak_x:.5f}, Y = {peak_y:.2f}{norm_status}")
            print(f"{integration_info}\n")
        
        else:
            # Multi-channel processing
            channels_data, channel_names = load_phs_file(file, multi_channel=True)
            if channels_data is None or not channel_names:
                print(f"Skipping file: {file}")
                continue

            # Read associated .set file
            runtime, start_datetime, integration_time, is_integration_enabled, divison = read_set_file(file)
            runtime_seconds = parse_runtime_to_seconds(runtime) if runtime else None
            
            # Create base result for this file
            combined_result = {
                "File": Path(file).name,
                "Runtime": runtime,
                "StartDateTime": start_datetime,
                "IntegrationTime": integration_time,
                "IsIntegrationEnabled": is_integration_enabled,
                "normalised": normalise and runtime_seconds is not None
            }
            
            print(f"Processing {len(channel_names)} channels: {', '.join(channel_names)}")
            
            # Process each channel
            for channel_name in channel_names:
                if channel_name not in channels_data:
                    continue
                    
                x = channels_data[channel_name]['x']
                y = channels_data[channel_name]['y']
                
                # Store original data for overlay
                if phs_overlay:
                    y_overlay = y.copy()
                    if normalise and runtime_seconds and runtime_seconds > 0:
                        y_overlay = y_overlay / runtime_seconds
                    
                    spectra_data.append({
                        'x': x.copy(),
                        'y': y_overlay,
                        'filename': Path(file).name,
                        'runtime': runtime,
                        'channel': channel_name
                    })
                
                # Analyze peak for this channel
                peaks, peak_x, peak_y, popt = analyze_largest_peak(
                    x, y,
                    show_plot=True,
                    save_plot=save_plots,
                    save_path=save_path,
                    file_name=Path(file).name,
                    runtime=runtime,
                    start_datetime=start_datetime,
                    integration_time=integration_time,
                    is_integration_enabled=is_integration_enabled,
                    normalise=normalise,
                    channel_name=channel_name,
                    division=divison
                )
                
                if peak_x is None:
                    print(f"No peaks found in {channel_name}")
                    continue
                
                # Add channel-specific data to combined result
                combined_result[f"Peak_{channel_name}_X"] = peak_x
                combined_result[f"Peak_{channel_name}_Y"] = peak_y
                combined_result[f"Gaussian_{channel_name}_A"] = popt[0]
                combined_result[f"Gaussian_{channel_name}_Mu"] = popt[1]
                combined_result[f"Gaussian_{channel_name}_Sigma"] = popt[2]
                
                # Print status
                norm_status = " (normalised)" if combined_result["normalised"] else " (raw)"
                print(f"{channel_name}: Peak found at X = {peak_x:.5f}, Y = {peak_y:.2f}{norm_status}")
            
            # Add combined result if any peaks were found
            if any(f"Peak_{ch}_X" in combined_result for ch in channel_names):
                results.append(combined_result)
            
            integration_info = format_integration_info(integration_time, is_integration_enabled)
            print(f"{integration_info}\n")

    # Create overlay plot if requested
    if phs_overlay and spectra_data:
        print("\nCreating PHS spectra overlay plot...")
        create_phs_overlay(spectra_data, save_path=save_path, normalise=normalise)

    # Save summary CSV in the save path with timestamp
    if save_results and results:
        # Generate timestamp for the CSV file as well
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        norm_suffix = "_normalised" if normalise else "_raw"
        multi_suffix = "_multichannel" if multi_channel else ""
        csv_filename = f"PHS_Peak_Summary_{timestamp}{norm_suffix}{multi_suffix}.csv"
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
        if multi_channel:
            print("SUMMARY OF PEAKS (MULTI-CHANNEL):")
        else:
            print("SUMMARY OF HIGHEST X PEAKS:")
        if normalise:
            print("(normalised by runtime where available)")
        else:
            print("(Raw counts - no normalisation)")
        print("="*80)
        
        # Display appropriate columns based on mode
        if multi_channel:
            display_columns = ["File", "Runtime", "StartDateTime", "IntegrationTime", "IsIntegrationEnabled", "normalised"]
            # Add channel-specific columns
            for col in df.columns:
                if col.startswith("Peak_") and col.endswith("_X"):
                    display_columns.append(col)
                    display_columns.append(col.replace("_X", "_Y"))
        else:
            display_columns = ["File", "Highest_X_Peak_X", "Highest_X_Peak_Y", "Runtime", 
                              "StartDateTime", "IntegrationTime", "IsIntegrationEnabled", "normalised"]
        
        # Only show columns that exist in the dataframe
        existing_columns = [col for col in display_columns if col in df.columns]
        print(df[existing_columns].to_string(index=False))
        print("="*80)
    else:
        print("No results to display.")

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    # Update these paths as needed
    folder_path = r"\\isis\Shares\Detectors\Lisa Malliolio 2025\PMT_calibration_20251001\batch1_43mmtiles_251003\roughpass"
    custom_save_path = r"Save Path Here"
    
    # Process with multi-channel enabled
    process_phs_folder(folder_path, save_results=False, save_plots=False, 
                      custom_save_path=custom_save_path, normalise=True, 
                      phs_overlay=True, multi_channel=True)
    
    # To process single channel (original behavior):
    # process_phs_folder(folder_path, save_results=True, save_plots=True, 
    #                   custom_save_path=custom_save_path, normalise=True, 
    #                   phs_overlay=True, multi_channel=False)