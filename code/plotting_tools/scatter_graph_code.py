import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_peak_scatter(file_path):
    """
    Creates a scatter plot from a two-column CSV file.
    Colors points blue if name contains "og", otherwise red.
    
    Parameters:
        file_path (str): Path to the CSV file containing the data.
                        Expected format: First column = Name, Second column = Peak Value
    """
    
    # -----------------------------
    # Step 1: Load the data
    # -----------------------------
    file_ext = os.path.splitext(file_path)[-1].lower()
    
    if file_ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Please use a CSV file format.")
    
    # Get column names (assuming first two columns)
    if len(df.columns) < 2:
        raise ValueError("The CSV file must contain at least two columns.")
    
    name_col = df.columns[0]
    peak_col = df.columns[1]
    
    print(f"Using columns: '{name_col}' (names) and '{peak_col}' (peak values)")
    
    # -----------------------------
    # Step 2: Create color mapping
    # -----------------------------
    # Create a boolean mask for names containing "og"
    contains_og = df[name_col].astype(str).str.contains("og", case=False, na=False)
    
    # Assign colors: blue for "og", red for others
    colors = ["blue" if og else "red" for og in contains_og]
    
    # -----------------------------
    # Step 3: Create scatter plot
    # -----------------------------
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(range(len(df)), df[peak_col], c=colors, alpha=0.7, s=50)
    
    # Customize the plot
    plt.xlabel("Tile ID", fontsize=12)
    plt.ylabel('Peak Location', fontsize=12)
    plt.title("Scatter Plot: Peak Values by Sample", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Add legend
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                           markersize=8, label='Original Tiles')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                          markersize=8, label='New Tiles')
    plt.legend(handles=[blue_patch, red_patch], loc='best')
    
    # Add sample names as x-tick labels (rotated for readability)
    if len(df) <= 50:  # Only show names if not too many samples
        plt.xticks(range(len(df)), df[name_col], rotation=45, ha='right')
    else:
        plt.xticks(range(0, len(df), max(1, len(df)//20)), 
                  [df[name_col].iloc[i] for i in range(0, len(df), max(1, len(df)//20))], 
                  rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # -----------------------------
    # Step 4: Print summary statistics
    # -----------------------------
    og_count = sum(contains_og)
    total_count = len(df)
    
    print(f"\nSummary:")
    print(f"Total samples: {total_count}")
    print(f"Samples with 'og' in name: {og_count} (blue points)")
    print(f"Other samples: {total_count - og_count} (red points)")
    
    if og_count > 0:
        og_mean = df[contains_og][peak_col].mean()
        print(f"Average peak value for 'og' samples: {og_mean:.4f}")
    
    if total_count - og_count > 0:
        other_mean = df[~contains_og][peak_col].mean()
        print(f"Average peak value for other samples: {other_mean:.4f}")


if __name__ == "__main__":
    # Replace with your CSV file path
    file_path = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\Scintillating Tile Tests\pmt_rig_250825\spreadsheets\Scatter Plots\scatter_plot__cleaned_single_tile21_250903.csv"
    
    plot_peak_scatter(file_path)
    
    print("To use this script:")
    print("1. Update the file_path variable with your CSV file location")
    print("2. Uncomment the plot_peak_scatter(file_path) line")
    print("3. Run the script")
    print("\nExpected CSV format:")
    print("Column 1: Sample names")
    print("Column 2: Peak values")