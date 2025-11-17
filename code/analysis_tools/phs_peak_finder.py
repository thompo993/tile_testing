import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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
    trig_1 = None
    trig_3 = None 
    
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
                    elif line.startswith('TriggerLevel[1]'):
                        trig_1 = line.split('=')[1]
                    elif line.startswith('TriggerLevel[3]'):
                        trig_3 = line.split('=')[1]
                    
                    
        except Exception as e:
            print(f"Error reading .set file {set_file_path}: {e}")
    else:
        print(f"No .set file found for {Path(data_file_path).name}")
    
    # Check if divisions are present before asserting equality
    if division_1 is not None and division_3 is not None:
        division = float(division_1)
    elif division_1 is not None:
        print(f"Warning: Only division_1 found, using it as division value")
        division = float(division_1)
    elif division_3 is not None:
        print(f"Warning: Only division_3 found, using it as division value")
        division = float(division_3)
    else:
        print(f"Warning: No division values found in .set file, using default value of 1")
        division = 1.0

    return runtime, start_datetime, integration_time, is_integration_enabled, division, trig_1, trig_3

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
            if multi_channel:
                return None, None
            else:
                return None, None

        if not multi_channel:
            # Original single channel behavior
            if data.shape[1] >= 2:
                x = data.iloc[:, 0].values
                y = data.iloc[:, 1].values
                valid_mask = ~pd.isna(y)   # or y.notna() if y is a Series
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
                    valid_mask = ~pd.isna(y)
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
# Analyze ALL peaks in one file (FIXED)
# ------------------------
def analyze_all_peaks(x, y, window=10, poly=3, prominence=0.05,
                      show_plot=True, save_plot=False, save_path=None, file_name=None,
                      runtime=None, start_datetime=None, integration_time=None, 
                      is_integration_enabled=None, normalise=True, channel_name=None, 
                      division=1.0, trig_1=None, trig_3=None):
    """
    Smooths data, finds ALL peaks, fits Gaussian to each, and optionally plots/saves results.
    Returns a list of all peak information.
    """
    # Parse runtime for normalisation
    runtime_seconds = parse_runtime_to_seconds(runtime) if runtime else None
    
    # normalise data if requested and runtime is available
    y_original = y.copy()
    if normalise and runtime_seconds and runtime_seconds > 0:
        y = y / runtime_seconds
        y_label = "Counts/second"
        normalisation_note_runtime = f"normalised by runtime ({runtime}s)"
    else:
        y_label = "Counts"
        normalisation_note_runtime = "Raw counts (no normalisation)" if not normalise else "Raw counts (runtime unavailable)"
    
    # Smooth data
    y_smooth = savgol_filter(y, window_length=window, polyorder=poly)

    # Find ALL peaks on smoothed data
    peaks, _ = find_peaks(
        y_smooth,
        height=np.max(y_smooth) * prominence,
        distance=len(y) // 20
    )

    if len(peaks) == 0:
        print(f"No peaks detected in {channel_name if channel_name else 'data'}.")
        return []

    # Store information for all peaks
    all_peak_info = []
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot original data
    if normalise and runtime_seconds:
        plt.plot(x, y, label="normalised Spectrum", color="lightblue", alpha=0.7, linewidth=1.5)
    else:
        plt.plot(x, y_original, label="Raw Spectrum", color="lightgray", alpha=0.7)
    
    plt.plot(x, y_smooth, label="Smoothed Spectrum", color="blue", linewidth=2)
    plt.plot(x[peaks], y_smooth[peaks], "ro", markersize=8, label="Detected Peaks")
    
    # FIX: Changed from 'colour' to 'color' to match loop variable
    color = "green"
    
    for idx, peak_idx in enumerate(peaks):
        peak_x = x[peak_idx]
        peak_y = y_smooth[peak_idx]
        
        # Fit Gaussian around this peak
        fit_range = (x > peak_x - (x[-1] - x[0]) * 0.05) & (x < peak_x + (x[-1] - x[0]) * 0.05)
        x_fit = x[fit_range]
        y_fit = y_smooth[fit_range]
        p0 = [peak_y, peak_x, (x_fit[-1] - x_fit[0]) / 6]

        try:
            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
            
            # Plot Gaussian fit
            plt.plot(x_fit, gaussian(x_fit, *popt), "--", linewidth=2, color=color,
                    label=f"Peak {idx+1} Fit (X={peak_x:.4f})")
            
            # Store peak information
            peak_info = {
                'peak_number': idx + 1,
                'peak_x': peak_x,
                'peak_y': peak_y,
                'gaussian_A': popt[0],
                'gaussian_mu': popt[1],
                'gaussian_sigma': popt[2]
            }
            all_peak_info.append(peak_info)
            
        except RuntimeError:
            # If fit fails, still store the peak location
            peak_info = {
                'peak_number': idx + 1,
                'peak_x': peak_x,
                'peak_y': peak_y,
                'gaussian_A': None,
                'gaussian_mu': None,
                'gaussian_sigma': None
            }
            all_peak_info.append(peak_info)
            print(f"Warning: Gaussian fit failed for peak {idx+1} at X={peak_x:.4f}")
    
    # Create info text for the plot
    integration_info = format_integration_info(integration_time, is_integration_enabled)
    info_text = f'Start DateTime: {start_datetime}\n'
    info_text += f'Runtime: {runtime}\n'
    # FIX: Ensure division is valid before formatting
    if division is not None:
        info_text += f'mV Per Division: {division*100:.2f}\n'
    info_text += f'{normalisation_note_runtime}\n'
    info_text += f'Trigger Level Ch1: {trig_1} mV\n'
    info_text += f'Trigger Level Ch3: {trig_3} mV\n'
    info_text += integration_info
    info_text += f'\nTotal Peaks Detected: {len(peaks)}' 
    if channel_name:
        info_text += f'\nChannel: {channel_name}'
    
    plt.figtext(0.76, 0.63, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.xlabel("Voltage Output", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    title_suffix = f" - {channel_name}" if channel_name else ""
    plt.title(f"All Peaks Detection: {file_name if file_name else 'Unknown File'}{title_suffix}", 
                fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot if requested
    if save_plot and save_path and file_name:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        base_name = os.path.splitext(file_name)[0]
        norm_suffix = "_normalised" if normalise and runtime_seconds else "_raw"
        channel_suffix = f"_{channel_name}" if channel_name else ""
        plot_filename = f"{base_name}_{timestamp}{norm_suffix}{channel_suffix}_all_peaks_plot.png"
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

    return all_peak_info

# ------------------------
# Create overlay plot of all spectra
# ------------------------
def create_phs_overlay(spectra_data, save_path=None, normalise=True):
    """
    Create an overlay plot of all PHS spectra
    """
    if not spectra_data:
        print("No spectra data available for overlay plot.")
        return
    
    plt.figure(figsize=(14, 10))
    
    # choose tab10 for up to 10 spectra, otherwise viridis for more
    n = min(len(spectra_data), 10)
    colors = cm.get_cmap("tab10")(np.linspace(0, 1, n))
    if len(spectra_data) > 10:
        colors = cm.get_cmap("viridis")(np.linspace(0, 1, len(spectra_data)))
    
    for i, spectrum in enumerate(spectra_data):
        x = spectrum['x']
        y = spectrum['y']
        filename = spectrum['filename']
        runtime = spectrum['runtime']
        channel = spectrum.get('channel', '')
        
        linestyle = '-' if len(spectra_data) <= 10 else '-'
        alpha = 0.7 if len(spectra_data) <= 5 else 0.6
        linewidth = 1.5 if len(spectra_data) <= 10 else 1.0
        
        label = f"{filename}" + (f" - {channel}" if channel else "")
        plt.plot(x, y, color=colors[i], alpha=alpha, linewidth=linewidth,
                linestyle=linestyle, label=label)
    
    y_label = "Counts/second" if normalise else "Counts"
    plt.xlabel("Voltage Output", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    title = "PHS Spectra Overlay - "
    title += "Normalised by Runtime" if normalise else "Raw Counts"
    plt.title(title, fontsize=14, fontweight='bold')
    
    if len(spectra_data) <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        plt.figtext(0.02, 0.98, f"Showing {len(spectra_data)} spectra", 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                   verticalalignment='top')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
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
# Process all files in folder (FIXED)
# ------------------------
def process_phs_folder(folder_path, save_results=True, save_plots=False, 
                       custom_save_path=None, normalise=True, phs_overlay=False, multi_channel=False):
    """
    Process all PHS files in a folder and extract ALL peaks.
    """
    files = find_phs_files(folder_path)
    if not files:
        print("No valid PHS data files found.")
        return

    results = []
    spectra_data = []
    
    print(f"Found {len(files)} files to analyze.")
    print(f"Multi-channel: {'ON' if multi_channel else 'OFF'}")
    print(f"normalisation: {'ON' if normalise else 'OFF'}")
    print(f"PHS Overlay: {'ON' if phs_overlay else 'OFF'}")
    print("")

    save_path = custom_save_path if custom_save_path else folder_path
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        if save_plots:
            print(f"Plots will be saved to: {save_path}")

    for i, file in enumerate(files, 1):
        print(f"Processing {i}/{len(files)}: {Path(file).name}")
        
        if not multi_channel:
            # Single channel processing
            x, y = load_phs_file(file, multi_channel=False)
            if x is None or y is None:
                print(f"Skipping file: {file}")
                continue

            runtime, start_datetime, integration_time, is_integration_enabled, division, trig_1, trig_3 = read_set_file(file)

            # FIX: Ensure y_original is properly converted to numeric
            y_original = pd.to_numeric(pd.Series(y), errors='coerce').to_numpy(dtype=float)
            runtime_seconds = parse_runtime_to_seconds(runtime) if runtime else None
            
            if phs_overlay:
                if normalise and runtime_seconds and runtime_seconds > 0:
                    y_overlay = y_original / float(runtime_seconds)
                else:
                    y_overlay = y_original.copy()
                
                spectra_data.append({
                    'x': x.copy(),
                    'y': y_overlay,
                    'filename': Path(file).name,
                    'runtime': runtime
                })
            # Get ALL peaks
            all_peaks = analyze_all_peaks(
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
                division=division,
                trig_1=trig_1, 
                trig_3=trig_3    
            )

            if not all_peaks:
                print(f"No peaks found in {file}")
                continue

            # Store each peak as a separate row
            for peak_info in all_peaks:
                result = {
                    "File": Path(file).name,
                    "Peak_Number": peak_info['peak_number'],
                    "Peak_X": peak_info['peak_x'],
                    "Peak_Y": peak_info['peak_y'],
                    "Gaussian_A": peak_info['gaussian_A'],
                    "Gaussian_Mu": peak_info['gaussian_mu'],
                    "Gaussian_Sigma": peak_info['gaussian_sigma'],
                    "Runtime": runtime,
                    "StartDateTime": start_datetime,
                    "IntegrationTime": integration_time,
                    "IsIntegrationEnabled": is_integration_enabled,
                    "normalised": normalise and runtime_seconds is not None
                }
                results.append(result)
            
            norm_status = " (normalised)" if (normalise and runtime_seconds) else " (raw)"
            integration_info = format_integration_info(integration_time, is_integration_enabled)
            print(f"Found {len(all_peaks)} peaks{norm_status}")
            print(f"{integration_info}\n")
        
        else:
            # Multi-channel processing
            channels_data, channel_names = load_phs_file(file, multi_channel=True)
            if channels_data is None or not channel_names:
                print(f"Skipping file: {file}")
                continue

            runtime, start_datetime, integration_time, is_integration_enabled, division, trig_1, trig_3 = read_set_file(file)
            runtime_seconds = parse_runtime_to_seconds(runtime) if runtime else None
            
            print(f"Processing {len(channel_names)} channels: {', '.join(channel_names)}")
            
            for channel_name in channel_names:
                if channel_name not in channels_data:
                    continue
                    
                x = channels_data[channel_name]['x']
                y = channels_data[channel_name]['y']
                
                # FIX: Ensure y is numeric before overlay processing
                y = pd.to_numeric(pd.Series(y), errors='coerce').to_numpy(dtype=float)
                
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
                
                # Get ALL peaks for this channel
                all_peaks = analyze_all_peaks(
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
                    division=division,
                    trig_1=trig_1,
                    trig_3=trig_3
                )
                
                if not all_peaks:
                    print(f"No peaks found in {channel_name}")
                    continue
                
                # Store each peak as a separate row
                for peak_info in all_peaks:
                    result = {
                        "File": Path(file).name,
                        "Channel": channel_name,
                        "Peak_Number": peak_info['peak_number'],
                        "Peak_X": peak_info['peak_x'],
                        "Peak_Y": peak_info['peak_y'],
                        "Gaussian_A": peak_info['gaussian_A'],
                        "Gaussian_Mu": peak_info['gaussian_mu'],
                        "Gaussian_Sigma": peak_info['gaussian_sigma'],
                        "Runtime": runtime,
                        "StartDateTime": start_datetime,
                        "IntegrationTime": integration_time,
                        "IsIntegrationEnabled": is_integration_enabled,
                        "normalised": normalise and runtime_seconds is not None
                    }
                    results.append(result)
                
                norm_status = " (normalised)" if (normalise and runtime_seconds) else " (raw)"
                print(f"{channel_name}: Found {len(all_peaks)} peaks{norm_status}")
            
            integration_info = format_integration_info(integration_time, is_integration_enabled)
            print(f"{integration_info}\n")

    # Create overlay plot if requested
    if phs_overlay and spectra_data:
        print("\nCreating PHS spectra overlay plot...")
        create_phs_overlay(spectra_data, save_path=save_path, normalise=normalise)

    # Save summary CSV with ALL peaks
    if save_results and results:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        norm_suffix = "_normalised" if normalise else "_raw"
        multi_suffix = "_multichannel" if multi_channel else ""
        csv_filename = f"PHS_All_Peaks_Summary_{timestamp}{norm_suffix}{multi_suffix}.csv"
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
            print("SUMMARY OF ALL PEAKS (MULTI-CHANNEL):")
        else:
            print("SUMMARY OF ALL PEAKS:")
        if normalise:
            print("(normalised by runtime where available)")
        else:
            print("(Raw counts - no normalisation)")
        print("="*80)
        
        # Display appropriate columns
        if multi_channel:
            display_columns = ["File", "Channel", "Peak_Number", "Peak_X", "Peak_Y", 
                            "Runtime", "StartDateTime", "normalised"]
        else:
            display_columns = ["File", "Peak_Number", "Peak_X", "Peak_Y", 
                            "Runtime", "StartDateTime", "normalised"]
        
        existing_columns = [col for col in display_columns if col in df.columns]
        print(df[existing_columns].to_string(index=False))
        print("="*80)
        print(f"Total peaks found: {len(results)}")
    else:
        print("No results to display.")

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    # Update these paths as needed
    folder_path = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\Scintillating Tile Tests\dual_pmt_rig_251112\grease_tests_251113"
    custom_save_path = r"save_path_here"
    # Process with multi-channel enabled
    process_phs_folder(folder_path, save_results=False, save_plots=False, 
                        custom_save_path=custom_save_path, normalise=True, 
                        phs_overlay=True, multi_channel=True)
    
    # To process single channel (original behavior):
    # process_phs_folder(folder_path, save_results=True, save_plots=True, 
    #                   custom_save_path=custom_save_path, normalise=True, 
    #                   phs_overlay=True, multi_channel=False)