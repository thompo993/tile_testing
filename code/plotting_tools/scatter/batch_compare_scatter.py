import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_peak_scatter(file_path):
    """
    Creates a scatter plot from a two-column CSV file.
    Colors points:
    - Blue if name contains "og" (original tiles)
    - Green if name contains "nb" (new batch)
    - Red for all others
    
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
    # Create boolean masks for different tile types
    contains_og = df[name_col].astype(str).str.contains("og", case=False, na=False)
    contains_nb = df[name_col].astype(str).str.contains("nb", case=False, na=False)
    
    # Assign colors with priority: nb (green) > og (blue) > others (red)
    colors = []
    for og, nb in zip(contains_og, contains_nb):
        if nb:
            colors.append("green")
        elif og:
            colors.append("blue")
        else:
            colors.append("red")
    
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
                          markersize=8, label='Standard Tiles')
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                           markersize=8, label='New Batch Tiles (04/09/25)')
    plt.legend(handles=[blue_patch, red_patch, green_patch], loc='best')
    
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
    og_count = sum(contains_og & ~contains_nb)  # Only og, not nb
    nb_count = sum(contains_nb)  # New batch tiles
    other_count = len(df) - og_count - nb_count
    total_count = len(df)
    
    print(f"\nSummary:")
    print(f"Total samples: {total_count}")
    print(f"Original tiles ('og' in name): {og_count} (blue points)")
    print(f"New batch tiles ('nb' in name): {nb_count} (green points)")
    print(f"Standard tiles: {other_count} (red points)")
    
    if og_count > 0:
        og_mean = df[contains_og & ~contains_nb][peak_col].mean()
        print(f"Average peak value for original tiles: {og_mean:.4f}")
    
    if nb_count > 0:
        nb_mean = df[contains_nb][peak_col].mean()
        print(f"Average peak value for new batch tiles: {nb_mean:.4f}")
    
    if other_count > 0:
        other_mean = df[~contains_og & ~contains_nb][peak_col].mean()
        print(f"Average peak value for standard tiles: {other_mean:.4f}")


if __name__ == "__main__":
    # Replace with your CSV file path
    file_path = r"File Path Here"
    
    plot_peak_scatter(file_path)
    
    print("To use this script:")
    print("1. Update the file_path variable with your CSV file location")
    print("2. Run the script")
    print("\nExpected CSV format:")
    print("Column 1: Sample names")
    print("Column 2: Peak values")
    print("\nColor coding:")
    print("- Blue: Original tiles (contains 'og')")
    print("- Green: New batch tiles (contains 'nb')")
    print("- Red: Standard tiles (all others)")