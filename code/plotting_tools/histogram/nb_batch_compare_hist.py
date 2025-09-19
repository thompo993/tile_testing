import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def load_and_plot_histograms(csv_path, bins, alpha=0.7):
    """
    Load CSV data and create layered histograms for peak locations.
    
    Parameters:
    csv_path (str): Path to the CSV file
    bins (int): Number of bins for the histograms
    alpha (float): Transparency level for overlapping histograms
    """
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Print column names for debugging
        print("Available columns:", list(df.columns))
        
        # Try to identify the correct column names
        og_col = None
        new_col = None
        nb_col = None
        
        for col in df.columns:
            if 'og' in col.lower() or 'original' in col.lower():
                og_col = col
            elif 'new' in col.lower() and 'nb' not in col.lower():
                new_col = col
            elif 'nb' in col.lower() and 'peaks' in col.lower():
                nb_col = col
        
        # If automatic detection fails, use exact column names
        if og_col is None:
            og_col = 'peak locations og'
        if new_col is None:
            new_col = 'peak locations new'
        if nb_col is None:
            nb_col = 'nb peaks'
            
        print(f"Using columns: '{og_col}', '{new_col}', and '{nb_col}'")
        
        # Extract the data columns
        og_data = df[og_col].dropna()  # Remove NaN values
        new_data = df[new_col].dropna()  # Remove NaN values
        nb_data = df[nb_col].dropna() if nb_col in df.columns else pd.Series()  # Handle missing column
        
        # Print basic statistics
        print(f"\nData Summary:")
        print(f"Original Batch (OG): {len(og_data)} samples")
        print(f"New Batch: {len(new_data)} samples")
        print(f"NB Peaks: {len(nb_data)} samples")
        
        if len(og_data) > 0:
            print(f"Original Batch range: {og_data.min():.2f} to {og_data.max():.2f}")
        if len(new_data) > 0:
            print(f"New Batch range: {new_data.min():.2f} to {new_data.max():.2f}")
        if len(nb_data) > 0:
            print(f"NB Peaks range: {nb_data.min():.2f} to {nb_data.max():.2f}")
        
        # Determine the range for all histograms to ensure they're comparable
        all_data = []
        if len(og_data) > 0:
            all_data.append(og_data)
        if len(new_data) > 0:
            all_data.append(new_data)
        if len(nb_data) > 0:
            all_data.append(nb_data)
        
        if not all_data:
            print("Error: No valid data found in any column")
            return None, None, None, None
        
        combined_min = min([data.min() for data in all_data])
        combined_max = max([data.max() for data in all_data])
        bin_edges = np.linspace(combined_min, combined_max, bins + 1)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot layered histograms
        if len(og_data) > 0:
            plt.hist(og_data, bins=bin_edges, alpha=alpha, label='Peak Locations (Original)', 
                    color='blue', edgecolor='black', linewidth=0.5)
        
        if len(new_data) > 0:
            plt.hist(new_data, bins=bin_edges, alpha=alpha, label='Peak Locations (New)', 
                    color='red', edgecolor='black', linewidth=0.5)
        
        if len(nb_data) > 0:
            plt.hist(nb_data, bins=bin_edges, alpha=alpha, label='NB Peaks (New Batch)', 
                    color='green', edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        plt.xlabel('Peak Location Values', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Layered Histograms of Peak Locations\n(Bins: {bins})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add statistics to the plot
        stats_text = ""
        if len(og_data) > 0:
            stats_text += f'Original: μ={og_data.mean():.5f}, σ={og_data.std():.5f}\n'
        if len(new_data) > 0:
            stats_text += f'New: μ={new_data.mean():.5f}, σ={new_data.std():.5f}\n'
        if len(nb_data) > 0:
            stats_text += f'NB Peaks: μ={nb_data.mean():.5f}, σ={nb_data.std():.5f}'
        
        if stats_text:
            plt.figtext(0.02, 0.02, stats_text,
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
        return df, og_data, new_data, nb_data
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_path}'")
        print("Please check the file path and make sure the file exists.")
    except KeyError as e:
        print(f"Error: Column not found - {e}")
        print("Available columns:", list(df.columns))
        print("Please check the column names in your CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def plot_with_custom_bins(csv_path, bins_list):
    """
    Create multiple plots with different bin numbers for comparison.
    
    Parameters:
    csv_path (str): Path to the CSV file
    bins_list (list): List of bin numbers to try
    """
    
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # Auto-detect columns
        og_col = None
        new_col = None
        nb_col = None
        
        for col in df.columns:
            if 'og' in col.lower() or 'original' in col.lower():
                og_col = col
            elif 'new' in col.lower() and 'nb' not in col.lower():
                new_col = col
            elif 'nb' in col.lower() and 'peaks' in col.lower():
                nb_col = col
        
        if og_col is None:
            og_col = 'peak locations og'
        if new_col is None:
            new_col = 'peak locations new'
        if nb_col is None:
            nb_col = 'nb peaks'
        
        og_data = df[og_col].dropna() if og_col in df.columns else pd.Series()
        new_data = df[new_col].dropna() if new_col in df.columns else pd.Series()
        nb_data = df[nb_col].dropna() if nb_col in df.columns else pd.Series()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Determine combined range
        all_data = []
        if len(og_data) > 0:
            all_data.append(og_data)
        if len(new_data) > 0:
            all_data.append(new_data)
        if len(nb_data) > 0:
            all_data.append(nb_data)
        
        if not all_data:
            print("Error: No valid data found for comparison plots")
            return
        
        combined_min = min([data.min() for data in all_data])
        combined_max = max([data.max() for data in all_data])
        
        for i, bins in enumerate(bins_list[:4]):  # Limit to 4 plots
            bin_edges = np.linspace(combined_min, combined_max, bins + 1)
            
            if len(og_data) > 0:
                axes[i].hist(og_data, bins=bin_edges, alpha=ALPHA, label='Original', color='blue')
            if len(new_data) > 0:
                axes[i].hist(new_data, bins=bin_edges, alpha=ALPHA, label='New', color='red')
            if len(nb_data) > 0:
                axes[i].hist(nb_data, bins=bin_edges, alpha=ALPHA, label='NB Peaks', color='green')
            
            axes[i].set_title(f'Bins: {bins}', fontweight='bold')
            axes[i].set_xlabel('Peak Location Values')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Peak Locations Histograms - Different Bin Numbers', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating multiple plots: {e}")

# Main execution
if __name__ == "__main__":

# Configuration - Edit these parameters as needed
    CSV_FILE_PATH = r"\\isis\Shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\Scintillating Tile Tests\pmt_rig_250825\spreadsheets\Histograms\comparison_histogram__between_new_og_nv_csv_250903.csv" 
    BIN_NUMBER = 14 # Change this to adjust the number of bins
    ALPHA = 0.7  # Transparency for overlapping histograms (0-1)

    print("Peak Locations Histogram Plotter (3 Column Version)")
    print("=" * 50)
    
    # Main plot with specified bin number
    print(f"\nGenerating histogram with {BIN_NUMBER} bins...")
    df, og_data, new_data, nb_data = load_and_plot_histograms(CSV_FILE_PATH, BIN_NUMBER, ALPHA)
    
    # Uncomment the lines below to create comparison plots with different bin numbers
    # print("\nGenerating comparison plots with different bin numbers...")
    # plot_with_custom_bins(CSV_FILE_PATH, [10, 20, 30, 50])
    
    print("\nScript completed!")