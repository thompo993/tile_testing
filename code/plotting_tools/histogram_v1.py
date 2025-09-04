import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration - Edit these parameters as needed
CSV_FILE_PATH = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\Scintillating Tile Tests\pmt_rig_250825\spreadsheets\histogram_csv_250903.csv" 
BIN_NUMBER = 10  # Change this to adjust the number of bins
ALPHA = 0.7  # Transparency for overlapping histograms (0-1)

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
        
        for col in df.columns:
            if 'og' in col.lower() or 'original' in col.lower():
                og_col = col
            elif 'new' in col.lower():
                new_col = col
        
        # If automatic detection fails, use exact column names
        if og_col is None:
            og_col = 'peak locations og'
        if new_col is None:
            new_col = 'peak locations new'
            
        print(f"Using columns: '{og_col}' and '{new_col}'")
        
        # Extract the data columns
        og_data = df[og_col].dropna()  # Remove NaN values
        new_data = df[new_col].dropna()  # Remove NaN values
        
        # Print basic statistics
        print(f"\nData Summary:")
        print(f"Original data points: {len(og_data)}")
        print(f"New data points: {len(new_data)}")
        print(f"Original range: {og_data.min():.2f} to {og_data.max():.2f}")
        print(f"New range: {new_data.min():.2f} to {new_data.max():.2f}")
        
        # Determine the range for both histograms to ensure they're comparable
        combined_min = min(og_data.min(), new_data.min())
        combined_max = max(og_data.max(), new_data.max())
        bin_edges = np.linspace(combined_min, combined_max, bins + 1)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot layered histograms
        plt.hist(og_data, bins=bin_edges, alpha=alpha, label='Peak Locations (Original)', 
                color='blue', edgecolor='black', linewidth=0.5)
        plt.hist(new_data, bins=bin_edges, alpha=alpha, label='Peak Locations (New)', 
                color='red', edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        plt.xlabel('Peak Location Values', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Layered Histograms of Peak Locations\n(Bins: {bins})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add some statistics to the plot
        plt.figtext(0.02, 0.02, f'Original: μ={og_data.mean():.5f}, σ={og_data.std():.5f}\n'
                                 f'New: μ={new_data.mean():.5f}, σ={new_data.std():.5f}',
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
        return df, og_data, new_data
        
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
        
        for col in df.columns:
            if 'og' in col.lower() or 'original' in col.lower():
                og_col = col
            elif 'new' in col.lower():
                new_col = col
        
        if og_col is None:
            og_col = 'peak locations og'
        if new_col is None:
            new_col = 'peak locations new'
        
        og_data = df[og_col].dropna()
        new_data = df[new_col].dropna()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        combined_min = min(og_data.min(), new_data.min())
        combined_max = max(og_data.max(), new_data.max())
        
        for i, bins in enumerate(bins_list[:4]):  # Limit to 4 plots
            bin_edges = np.linspace(combined_min, combined_max, bins + 1)
            
            axes[i].hist(og_data, bins=bin_edges, alpha=ALPHA, label='Original', color='blue')
            axes[i].hist(new_data, bins=bin_edges, alpha=ALPHA, label='New', color='red')
            
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
    print("Peak Locations Histogram Plotter")
    print("=" * 40)
    
    # Main plot with specified bin number
    print(f"\nGenerating histogram with {BIN_NUMBER} bins...")
    df, og_data, new_data = load_and_plot_histograms(CSV_FILE_PATH, BIN_NUMBER, ALPHA)
    
    # Uncomment the lines below to create comparison plots with different bin numbers
    # print("\nGenerating comparison plots with different bin numbers...")
    # plot_with_custom_bins(CSV_FILE_PATH, [10, 20, 30, 50])
    
    print("\nScript completed!")