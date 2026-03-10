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
from matplotlib.offsetbox import AnchoredText
warnings.filterwarnings("ignore")

# ------------------------
# second-order polynomial function
# ------------------------
def polynomial_2nd_order(x, a, b, c):
    return a * x**2 + b * x + c

# ------------------------
# Gaussian function for fitting
# ------------------------
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))

# ------------------------
# Calculate peak position and error using Gaussian fit
# ------------------------
def calculate_peak_statistics(x_data, y_data):
    valid_mask = y_data > 0
    x_valid = x_data[valid_mask]
    y_valid = y_data[valid_mask]
    
    if len(x_valid) < 3:
        return None, None
    
    try:
        amplitude_guess = np.max(y_valid)
        mean_guess = x_valid[np.argmax(y_valid)]
        sigma_guess = (x_valid[-1] - x_valid[0]) / 4
        
        p0 = [amplitude_guess, mean_guess, sigma_guess]
        popt, pcov = curve_fit(gaussian, x_valid, y_valid, p0=p0)
        
        mean = popt[1]
        mean_error = np.sqrt(pcov[1, 1])
        
        return mean, mean_error
        
    except (RuntimeError, ValueError):
        return None, None

# ------------------------
# Extract ID from filename
# ------------------------
def extract_id_from_filename(filename):
    filename = str(Path(filename).name)
    id_pos = filename.lower().find('id')
    
    if id_pos == -1:
        return None
    
    start_pos = id_pos + 2
    underscore_pos = filename.find('_', start_pos)
    
    if underscore_pos == -1:
        dot_pos = filename.find('.', start_pos)
        if dot_pos == -1:
            return filename[start_pos:]
        else:
            return filename[start_pos:dot_pos]
    else:
        return filename[start_pos:underscore_pos]

# ------------------------
# read .set file for Runtime and StartDateTime
# ------------------------
def read_set_file(data_file_path):
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
    if is_integration_enabled is None:
        return "Integration: Not specified"
    elif not is_integration_enabled:
        return "Integration: OFF"
    elif integration_time is not None:
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
    if runtime_str is None:
        return None
    
    try:
        if ':' in runtime_str:
            parts = runtime_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = map(float, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:
                minutes, seconds = map(float, parts)
                return minutes * 60 + seconds
        else:
            return float(runtime_str)
    except (ValueError, TypeError):
        print(f"Could not parse runtime: {runtime_str}")
        return None

# ------------------------
# Extract channel names from header
# ------------------------
def extract_channel_names(header_line):
    channel_names = []
    parts = header_line.split('\t')
    
    for part in parts:
        part = part.strip()
        if part.startswith('Counts:'):
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
            return None, None

        if not multi_channel:
            if data.shape[1] >= 2:
                x = data.iloc[:, 0].values
                y = data.iloc[:, 1].values
                valid_mask = ~pd.isna(y)
                return x[valid_mask], y[valid_mask]
            else:
                print(f"Warning: File {file_path} doesn't have at least 2 columns")
                return None, None
        else:
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
            
            channels_data = {}
            col_idx = 0
            
            for channel_name in channel_names:
                if col_idx + 1 < data.shape[1]:
                    x = data.iloc[:, col_idx].values
                    y = data.iloc[:, col_idx + 1].values
                    valid_mask = ~pd.isna(y)
                    channels_data[channel_name] = {
                        'x': x[valid_mask],
                        'y': y[valid_mask]
                    }
                    col_idx += 2
                else:
                    break
       
            return channels_data, channel_names
            
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None

# ------------------------
# Save plot data to CSV
# ------------------------
def save_plot_data_to_csv(x, y, y_smooth, peaks, save_path, file_name, channel_name=None, 
                          normalise=True, runtime_seconds=None):
    if save_path is None or file_name is None:
        return
    
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    base_name = os.path.splitext(file_name)[0]
    norm_suffix = "_normalised" if normalise and runtime_seconds else "_raw"
    channel_suffix = f"_{channel_name}" if channel_name else ""
    csv_filename = f"{base_name}_{timestamp}{norm_suffix}{channel_suffix}_plot_data.csv"
    full_csv_path = os.path.join(save_path, csv_filename)
    
    try:
        df_data = {
            'Voltage': x,
            'Counts_Raw': y,
            'Counts_Smoothed': y_smooth,
            'Is_Peak': [1 if i in peaks else 0 for i in range(len(x))]
        }
        df = pd.DataFrame(df_data)
        df.to_csv(full_csv_path, index=False)
        print(f"Plot data CSV saved to: {full_csv_path}")
        
    except Exception as e:
        print(f"Error saving plot data CSV: {e}")

# ------------------------
# Determine best location for info box
# ------------------------
def best_infobox_location(ax, y_smooth, x):
    """
    Compare the average normalised y-value in the left vs right quarter of the
    x-range on the smoothed spectrum and return the AnchoredText loc string
    for whichever side has more headroom (lower average counts).
    Falls back to 'upper left' if the comparison cannot be made.
    """
    try:
        x_min, x_max = x.min(), x.max()
        quarter = (x_max - x_min) / 4.0

        left_mask  = x <= x_min + quarter
        right_mask = x >= x_max - quarter

        y_max = y_smooth.max() if y_smooth.max() != 0 else 1.0
        left_mean  = y_smooth[left_mask].mean()  / y_max if left_mask.any()  else 1.0
        right_mean = y_smooth[right_mask].mean() / y_max if right_mask.any() else 1.0

        return "upper left" if left_mean <= right_mean else "upper right"
    except Exception:
        return "upper left"

# ------------------------
# Analyze ALL peaks in one file
# ------------------------
def analyze_all_peaks(x, y, window=10, poly=3, prominence=0.05,
                      show_plot=True, save_plot=False, save_csv=False, save_path=None, file_name=None,
                      runtime=None, start_datetime=None, integration_time=None, 
                      is_integration_enabled=None, normalise=True, channel_name=None, 
                      division=1.0, trig_1=None, trig_3=None):
    """
    Smooths data, finds ALL peaks, fits second-order polynomial to each, and calculates
    statistics from data points. Returns a list of all peak information.
    """
    # Parse runtime for normalisation
    runtime_seconds = parse_runtime_to_seconds(runtime) if runtime else None
    
    # Normalise data if requested and runtime is available
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

    # Save plot data to CSV if requested
    if save_csv and save_path and file_name:
        save_plot_data_to_csv(x, y, y_smooth, peaks, save_path, file_name, 
                              channel_name, normalise, runtime_seconds)

    all_peak_info = []
    
    # ---------------------------------------------------------------
    # Create the plot — use fig, ax so ax is always available
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if normalise and runtime_seconds:
        ax.plot(x, y, label="normalised Spectrum", color="lightblue", alpha=0.7, linewidth=1.5)
    else:
        ax.plot(x, y_original, label="Raw Spectrum", color="lightgray", alpha=0.7)
    
    ax.plot(x, y_smooth, label="Smoothed Spectrum", color="blue", linewidth=2)
    
    color = "green"
    
    for idx, peak_idx in enumerate(peaks):
        peak_x = x[peak_idx]
        peak_y = y_smooth[peak_idx]
        
        if peak_x > 0.05:
            fit_range = (x > peak_x - (x[-1] - x[0]) * 0.05) & (x < peak_x + (x[-1] - x[0]) * 0.05)
            x_fit = x[fit_range]
            y_fit = y[fit_range]
            
            data_mean, data_mean_err = calculate_peak_statistics(x_fit, y_fit)
            
            p0 = [-peak_y / ((x_fit[-1] - x_fit[0]) / 2)**2, 0, peak_y]
            
            try:
                popt, pcov = curve_fit(polynomial_2nd_order, x_fit, y_fit, p0=p0)
                perr = np.sqrt(np.diag(pcov))
                
                ax.plot(x_fit, polynomial_2nd_order(x_fit, *popt), "--", linewidth=2, color=color,
                        label=f"Polynomial Fit")
                
                a, b, c = popt
                a_err, b_err, c_err = perr
                
                if a != 0:
                    x_max_poly = -b / (2 * a)
                    x_max_poly_err = np.sqrt((b/(2*a**2))**2 * a_err**2 + (1/(2*a))**2 * b_err**2)
                    
                    if x_fit.min() <= x_max_poly <= x_fit.max():
                        y_max = polynomial_2nd_order(x_max_poly, a, b, c)
                        ax.plot(x_max_poly, y_max, "ro", linewidth=2, 
                                color="red", markersize=6, markeredgewidth=2, 
                                label=f"Polynomial Peak fit X={x_max_poly:.5f}±{x_max_poly_err:.5f}")
                        
                        if data_mean is not None and data_mean_err is not None:
                            y_at_data_mean = polynomial_2nd_order(data_mean, a, b, c)
                            ax.plot(data_mean, y_at_data_mean, "s", 
                                    color="purple", markersize=8, markeredgewidth=2,
                                    label=f"Gaussian Fit X={data_mean:.5f}±{data_mean_err:.5f}")
                        
                        peak_info = {
                            'peak_number': idx + 1,
                            'peak_x_poly': x_max_poly,
                            'peak_x_poly_err': x_max_poly_err,
                            'peak_x_data_mean': data_mean,
                            'peak_x_data_mean_err': data_mean_err,
                            'peak_y': y_max,
                            'polynomial_a': popt[0], 'polynomial_b': popt[1], 'polynomial_c': popt[2],
                            'polynomial_a_err': a_err, 'polynomial_b_err': b_err, 'polynomial_c_err': c_err,
                            'num_data_points': len(x_fit)
                        }
                    else:
                        peak_info = {
                            'peak_number': idx + 1,
                            'peak_x_poly': peak_x,
                            'peak_x_poly_err': None,
                            'peak_x_data_mean': data_mean,
                            'peak_x_data_mean_err': data_mean_err,
                            'peak_y': peak_y,
                            'polynomial_a': popt[0], 'polynomial_b': popt[1], 'polynomial_c': popt[2],
                            'polynomial_a_err': a_err, 'polynomial_b_err': b_err, 'polynomial_c_err': c_err,
                            'num_data_points': len(x_fit)
                        }
                else:
                    peak_info = {
                        'peak_number': idx + 1,
                        'peak_x_poly': peak_x,
                        'peak_x_poly_err': None,
                        'peak_x_data_mean': data_mean,
                        'peak_x_data_mean_err': data_mean_err,
                        'peak_y': peak_y,
                        'polynomial_a': popt[0], 'polynomial_b': popt[1], 'polynomial_c': popt[2],
                        'polynomial_a_err': a_err, 'polynomial_b_err': b_err, 'polynomial_c_err': c_err,
                        'num_data_points': len(x_fit)
                    }
                all_peak_info.append(peak_info)
                
            except RuntimeError:
                peak_info = {
                    'peak_number': idx + 1,
                    'peak_x_poly': peak_x,
                    'peak_x_poly_err': None,
                    'peak_x_data_mean': data_mean,
                    'peak_x_data_mean_err': data_mean_err,
                    'peak_y': peak_y,
                    'polynomial_a': None, 'polynomial_b': None, 'polynomial_c': None,
                    'polynomial_a_err': None, 'polynomial_b_err': None, 'polynomial_c_err': None,
                    'num_data_points': len(x_fit) if len(x_fit) > 0 else 0
                }
                all_peak_info.append(peak_info)
                print(f"Warning: Polynomial fit failed for peak {idx+1} at X={peak_x:.4f}")
        else:
            # Skip fitting for low-voltage peaks
            peak_info = {
                'peak_number': idx + 1,
                'peak_x_poly': peak_x,
                'peak_x_poly_err': None,
                'peak_x_data_mean': None,
                'peak_x_data_mean_err': None,
                'peak_y': peak_y,
                'polynomial_a': None, 'polynomial_b': None, 'polynomial_c': None,
                'polynomial_a_err': None, 'polynomial_b_err': None, 'polynomial_c_err': None,
                'num_data_points': 0
            }
            all_peak_info.append(peak_info)

    # ---------------------------------------------------------------
    # Build info text box
    # ---------------------------------------------------------------
    integration_info = format_integration_info(integration_time, is_integration_enabled)

    info_text = f"Start DateTime: {start_datetime}\n"
    info_text += f"Runtime: {runtime}\n"
    if division is not None:
        info_text += f"mV Per Division: {division * 100:.2f}\n"
    info_text += f"{normalisation_note_runtime}\n"
    info_text += f"Trigger Level Ch1: {trig_1} mV\n"
    info_text += f"Trigger Level Ch3: {trig_3} mV\n"
    info_text += integration_info
    info_text += f"\nTotal Peaks Detected: {len(peaks)}"
    if channel_name:
        info_text += f"\nChannel: {channel_name}"

    # Choose the corner with the least spectral activity
    infobox_loc = best_infobox_location(ax, y_smooth, x)

    anchored = AnchoredText(
        info_text,
        loc=infobox_loc,
        prop={"size": 10},
        frameon=True
    )
    anchored.patch.set_boxstyle("round,pad=0.5")
    anchored.patch.set_facecolor("lightgray")
    anchored.patch.set_alpha(0.8)
    ax.add_artist(anchored)

    # Legend goes to whichever upper corner the info box is NOT in
    legend_loc = "upper right" if infobox_loc == "upper left" else "upper left"

    ax.set_xlabel("Voltage Output [V]", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    title_suffix = f" - {channel_name}" if channel_name else ""
    ax.set_title(
        f"All Peaks Detection: {file_name if file_name else 'Unknown File'}{title_suffix}",
        fontsize=14,
        fontweight="bold"
    )

    ax.legend(fontsize=9, loc=legend_loc)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # ---------------------------------------------------------------
    # Save plot if requested
    # ---------------------------------------------------------------
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
    if not spectra_data:
        print("No spectra data available for overlay plot.")
        return
    
    plt.figure(figsize=(14, 10))
    
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
        
        alpha = 0.7 if len(spectra_data) <= 5 else 0.6
        linewidth = 1.5 if len(spectra_data) <= 10 else 1.0
        
        label = f"{filename}" + (f" - {channel}" if channel else "")
        plt.plot(x, y, color=colors[i], alpha=alpha, linewidth=linewidth, label=label)
    
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
# Process all files in folder
# ------------------------
def process_phs_folder(folder_path, save_results=True, save_plots=False, save_csv=False,
                    custom_save_path=None, normalise=True, phs_overlay=False, multi_channel=False):
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
    print(f"Save CSV: {'ON' if save_csv else 'OFF'}")
    print("")

    save_path = custom_save_path if custom_save_path else folder_path
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        if save_plots:
            print(f"Plots will be saved to: {save_path}")
        if save_csv:
            print(f"Plot data CSVs will be saved to: {save_path}")

    for i, file in enumerate(files, 1):
        print(f"Processing {i}/{len(files)}: {Path(file).name}")
        
        file_id = extract_id_from_filename(Path(file).name)
        
        if not multi_channel:
            x, y = load_phs_file(file, multi_channel=False)
            if x is None or y is None:
                print(f"Skipping file: {file}")
                continue

            runtime, start_datetime, integration_time, is_integration_enabled, division, trig_1, trig_3 = read_set_file(file)

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

            all_peaks = analyze_all_peaks(
                x, y,
                show_plot=True,
                save_plot=save_plots,
                save_csv=save_csv,
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

            for peak_info in all_peaks:
                result = {
                    "ID": file_id,
                    "File": Path(file).name,
                    "Peak_Number": peak_info['peak_number'],
                    "Peak_X_Poly": peak_info['peak_x_poly'],
                    "Peak_X_Poly_Err": peak_info['peak_x_poly_err'],
                    "Peak_X_Gaussian": peak_info['peak_x_data_mean'],
                    "Peak_X_Gaussian_Err": peak_info['peak_x_data_mean_err'],
                    "Peak_Y": peak_info['peak_y'],
                    "Polynomial_a": peak_info['polynomial_a'],
                    "Polynomial_b": peak_info['polynomial_b'],
                    "Polynomial_c": peak_info['polynomial_c'],
                    "Polynomial_a_err": peak_info['polynomial_a_err'],
                    "Polynomial_b_err": peak_info['polynomial_b_err'],
                    "Polynomial_c_err": peak_info['polynomial_c_err'],
                    "Num_Data_Points": peak_info['num_data_points'],
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
                
                all_peaks = analyze_all_peaks(
                    x, y,
                    show_plot=True,
                    save_plot=save_plots,
                    save_csv=save_csv,
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
                
                for peak_info in all_peaks:
                    result = {
                        "ID": file_id,
                        "File": Path(file).name,
                        "Channel": channel_name,
                        "Peak_Number": peak_info['peak_number'],
                        "Peak_X_Poly": peak_info['peak_x_poly'],
                        "Peak_X_Poly_Err": peak_info['peak_x_poly_err'],
                        "Peak_X_Gaussian": peak_info['peak_x_data_mean'],
                        "Peak_X_Gaussian_Err": peak_info['peak_x_data_mean_err'],
                        "Peak_Y": peak_info['peak_y'],
                        "Polynomial_a": peak_info['polynomial_a'],
                        "Polynomial_b": peak_info['polynomial_b'],
                        "Polynomial_c": peak_info['polynomial_c'],
                        "Polynomial_a_err": peak_info['polynomial_a_err'],
                        "Polynomial_b_err": peak_info['polynomial_b_err'],
                        "Polynomial_c_err": peak_info['polynomial_c_err'],
                        "Num_Data_Points": peak_info['num_data_points'],
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

    if phs_overlay and spectra_data:
        print("\nCreating PHS spectra overlay plot...")
        create_phs_overlay(spectra_data, save_path=save_path, normalise=normalise)

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
        
        if multi_channel:
            display_columns = ["ID", "File", "Channel", "Peak_Number", 
                               "Peak_X_Gaussian", "Peak_X_Gaussian_Err",
                               "Peak_X_Poly", "Peak_X_Poly_Err", "Peak_Y", 
                               "Num_Data_Points", "Runtime", "StartDateTime", "normalised"]
        else:
            display_columns = ["ID", "File", "Peak_Number", 
                               "Peak_X_Gaussian", "Peak_X_Gaussian_Err",
                               "Peak_X_Poly", "Peak_X_Poly_Err", "Peak_Y", 
                               "Num_Data_Points", "Runtime", "StartDateTime", "normalised"]
        
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
    folder_path = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\scintillating_tiles\dual_pmt_rig_251112\calibration\260309\LHS"
    custom_save_path = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\scintillating_tiles\log\30mm\30mm_gainmatching_260310\LHS"
    
    process_phs_folder(folder_path, save_results=True, save_plots=True, 
                        save_csv=True, custom_save_path=custom_save_path, 
                        normalise=True, phs_overlay=True, multi_channel=False)