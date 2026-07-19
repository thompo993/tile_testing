import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "lines.linewidth":   1.0,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size":  4.0,
    "ytick.major.size":  4.0,
    "xtick.minor.size":  2.0,
    "ytick.minor.size":  2.0,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "axes.grid":         True,
    "grid.color":        "#CCCCCC",
    "grid.linewidth":    0.4,
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "axes.axisbelow":    True,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.dpi":       600,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "figure.dpi":        150,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#AAAAAA",
    "legend.frameon":    True,
})


def plot_2d_files(folder_path, save_path):
    # Create save directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Get all .2D files in the folder
    data_files = list(Path(folder_path).glob("*.2D"))

    if not data_files:
        print(f"No .2D files found in {folder_path}")
        return

    print(f"Found {len(data_files)} .2D files")
    # we look for LHS or RHS inside of the file name
    # convention is as you look at the PMT rig, the LHS PMT is
    # on the LHS. this is also channel 2 on the HV power supply
    # we then signal this to the plot labels adding "lhs" or "rhs"
    # inside the string.
    for data_file in data_files:
        file_name = data_file.name
        print(f"Processing {file_name}...")

        if "lhs" in file_name.lower():
            print("Identified as LHS file")
            b_hand = "LHS Stud"
            d_hand = "RHS Stud"
        elif "rhs" in file_name.lower():
            print("Identified as RHS file")
            b_hand = "RHS Stud"
            d_hand = "LHS Stud"
        else:
            b_hand = ""
            d_hand = ""

        # Load the data from the .2D file
        try:
            data = np.loadtxt(data_file)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue

        # Create a 2D heatmap plot
        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(label='Intensity')

        plt.xlabel("{b_hand} Light Output [A.U]".format(b_hand=b_hand))
        plt.ylabel("{d_hand} Light Output [A.U]".format(d_hand=d_hand))

        plt.title(f"{file_name[:5]} Tile ID {file_name[8:11]} | LHS v RHS Stud Light Output Correlation")
        # Add y=x line to show perfect correlation
        max_val = data.shape[0]
        plt.plot([0, max_val], [0, max_val], 'w--', linewidth=2, label='Perfect Correlation (y=x)')
        plt.legend(loc="best")
        plt.ylim(0, 255)
        plt.xlim(0, 255)
        # Save the plot as a PNG file
        save_file = Path(save_path) / f"{data_file.stem}.png"
        plt.savefig(save_file, dpi=600, bbox_inches='tight')
        print(f"Saved to {save_file}")
        plt.close()

    print(f"All plots saved to {save_path}")


# ---------------------------------------------------------------------------
# PMT-swap-corrected stud asymmetry quantification + averaged 2D heatmap
# ---------------------------------------------------------------------------
#
# Logic:
#   Each tile is measured twice, with the B/D channel-to-stud mapping swapped:
#     - "...LHS_run..." file: Channel B = LHS stud, Channel D = RHS stud
#     - "...RHS_run..." file: Channel B = RHS stud, Channel D = LHS stud
#
#   The same two PMTs/channels (B and D) are used both times, just looking at
#   different studs. That means any consistent difference between channel B
#   and channel D that comes from the PMTs themselves (gain, HV, etc.) shows
#   up as an offset that FLIPS which stud it favours between the two runs,
#   while a genuine tile/stud asymmetry stays attached to the same stud in
#   both runs. So:
#
#       LHS_stud_response = mean( B_from_LHSrun, D_from_RHSrun )
#       RHS_stud_response = mean( D_from_LHSrun, B_from_RHSrun )
#
#   This averaging cancels a symmetric PMT gain mismatch between channels B
#   and D exactly (to first order), and substantially reduces it even if the
#   mismatch isn't perfectly symmetric. This was verified against synthetic
#   data with a known stud asymmetry and a known 20% channel gain mismatch:
#   the averaged estimate recovered the true stud values to within noise.
#
#   Each .2D file is treated as a 2D count histogram: rows = D-channel bins
#   (y-axis), columns = B-channel bins (x-axis) -- matching how plot_2d_files
#   calls imshow(data) with xlabel=b_hand, ylabel=d_hand. The "response" for
#   a channel is the counts-weighted mean bin value (i.e. the centroid of the
#   distribution along that axis).
#
#   The same swap-correction idea can be applied to the full 2D histogram,
#   not just its centroid: in the LHS_run file, columns = LHS stud and rows
#   = RHS stud already (the desired orientation). In the RHS_run file,
#   columns = RHS stud and rows = LHS stud, i.e. the *transpose* of the
#   desired orientation. So:
#
#       averaged_2D = ( normalize(data_LHSrun) + normalize(data_RHSrun.T) ) / 2
#
#   with columns = LHS stud, rows = RHS stud. Each histogram is normalized
#   to sum to 1 (see `normalize_histogram`) BEFORE averaging, because the
#   two runs are not guaranteed to have the same total number of events --
#   without normalizing first, whichever run happened to collect more
#   statistics would dominate the combined distribution, which would
#   reintroduce a run-dependent bias into what's supposed to be an equally
#   weighted, PMT-swap-corrected combination. This gives a single averaged
#   2D correlation plot per tile/run, analogous to the 1D centroid
#   correction above.
#
#   Quantifying asymmetry on that averaged 2D plot:
#
#   `asymmetry_centroid` (the original scalar) compares only the marginal
#   *means* of the LHS and RHS axes -- a first-moment measure. It's cheap
#   but can miss asymmetries in the shape of the distribution (skew,
#   correlated outliers, etc.) that don't shift the mean much.
#
#   `asymmetry_diagonal` instead uses the full 2D shape: every bin (i, j)
#   in the averaged histogram represents events with RHS-bin = i,
#   LHS-bin = j. Bins with j > i sit below the y=x line (LHS read higher
#   for that event); bins with i > j sit above it (RHS read higher). We
#   sum the counts on each side of the line and form
#
#       asymmetry_diagonal = (N_LHS_brighter - N_RHS_brighter)
#                             / (N_LHS_brighter + N_RHS_brighter)
#
#   bounded in [-1, 1], positive => LHS stud tends to read brighter.
#   Counts sitting exactly on the diagonal (i == j) are tracked separately
#   and excluded from the ratio. A per-bin difference map
#   (averaged_2D - averaged_2D.T) is also plotted, showing *where* on the
#   plot the imbalance is concentrated rather than collapsing it to one
#   number.


def compute_weighted_means(data):
    """
    Given a 2D histogram `data` (rows = D-channel/y bins, columns = B-channel/x
    bins), return the counts-weighted mean bin value along each axis:
    (mean_b, mean_d).
    """
    total = data.sum()
    if total == 0:
        return np.nan, np.nan

    col_totals = data.sum(axis=0)  # summed over rows -> total counts per B bin
    row_totals = data.sum(axis=1)  # summed over cols -> total counts per D bin

    b_bins = np.arange(data.shape[1])
    d_bins = np.arange(data.shape[0])

    mean_b = np.sum(col_totals * b_bins) / total
    mean_d = np.sum(row_totals * d_bins) / total
    return mean_b, mean_d


def compute_diagonal_asymmetry(data):
    """
    Given a 2D histogram `data` (rows = RHS-stud bin index, columns =
    LHS-stud bin index -- i.e. the orientation of the PMT-swap-corrected
    averaged 2D plot), quantify how the joint distribution splits across
    the y = x (perfect correlation) line.

    Bins with column index > row index (LHS bin > RHS bin) lie below the
    diagonal, i.e. events where the LHS stud read brighter. Bins with
    row index > column index lie above the diagonal (RHS read brighter).
    Bins with row index == column index sit exactly on the line and are
    reported separately.

    Returns a dict with:
      counts_lhs_brighter, counts_rhs_brighter, counts_on_diagonal,
      asymmetry_diagonal  (bounded in [-1, 1]; positive => LHS stud tends
                            to read brighter than RHS stud)
    """
    row_idx, col_idx = np.indices(data.shape)

    lhs_brighter_mask = col_idx > row_idx
    rhs_brighter_mask = col_idx < row_idx
    on_diag_mask = col_idx == row_idx

    counts_lhs_brighter = data[lhs_brighter_mask].sum()
    counts_rhs_brighter = data[rhs_brighter_mask].sum()
    counts_on_diagonal = data[on_diag_mask].sum()

    denom = counts_lhs_brighter + counts_rhs_brighter
    if denom == 0:
        asymmetry_diagonal = np.nan
    else:
        asymmetry_diagonal = (counts_lhs_brighter - counts_rhs_brighter) / denom

    return {
        "counts_lhs_brighter": counts_lhs_brighter,
        "counts_rhs_brighter": counts_rhs_brighter,
        "counts_on_diagonal": counts_on_diagonal,
        "asymmetry_diagonal": asymmetry_diagonal,
    }


def normalize_histogram(data):
    """
    Normalize a 2D histogram so its bins sum to 1, turning raw event counts
    into a probability distribution.

    This matters before averaging the LHS-run and RHS-run histograms
    together: the two runs are not guaranteed to have collected the same
    total number of events (different run time, trigger rate, live time,
    etc.). Averaging raw counts would let whichever run happened to record
    more statistics dominate the combined, PMT-swap-corrected distribution
    -- silently reintroducing the exact per-channel imbalance the averaging
    step is meant to cancel out. Normalizing first ensures each run gets
    equal (50/50) weight regardless of its total counts.
    """
    total = data.sum()
    if total == 0:
        return np.zeros_like(data, dtype=float)
    return data / total


def parse_2d_filename(file_name):
    """
    Parses filenames of the form:
        {distance}_{tileID}_{LHS|RHS}_{run}_{BvsD}.2D
    e.g. 105mm_id005_LHS_run001_BvsD.2D

    Returns (distance, tile_id, side, run) with side normalised to upper case.
    Raises IndexError/ValueError if the name doesn't have enough '_'-separated
    parts, which the caller catches and skips.
    """
    stem = Path(file_name).stem  # strip .2D
    parts = stem.split("_")
    distance, tile_id, side, run = parts[0], parts[1], parts[2], parts[3]
    return distance, tile_id, side.upper(), run


def plot_averaged_heatmap(averaged_data, distance, tile_id, run, save_path):
    """
    Plots and saves the PMT-swap-corrected averaged 2D correlation heatmap
    (columns = LHS stud, rows = RHS stud) for a single (distance, tile_id,
    run) group.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(averaged_data, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label='Normalized Intensity (probability per bin)')

    plt.xlabel("LHS Stud Light Output [A.U]")
    plt.ylabel("RHS Stud Light Output [A.U]")
    plt.title(
        f"{distance} Tile ID {tile_id} ({run}) | Averaged, PMT-Swap-Corrected\n"
        f"LHS v RHS Stud Light Output Correlation"
    )

    max_val = averaged_data.shape[0]
    plt.plot([0, max_val], [0, max_val], 'w--', linewidth=2, label='Perfect Correlation (y=x)')
    plt.legend(loc="best")
    plt.ylim(0, 255)
    plt.xlim(0, 255)

    save_file = Path(save_path) / f"{distance}_{tile_id}_{run}_averaged.png"
    plt.savefig(save_file, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved averaged heatmap to {save_file}")


def plot_asymmetry_difference_map(averaged_data, distance, tile_id, run, save_path):
    """
    Plots (averaged_2D - averaged_2D.T): zero everywhere the distribution
    is mirror-symmetric about the y=x line, positive where LHS-brighter
    events outweigh their RHS-brighter mirror bin, negative where the
    reverse holds. This shows *where* in the plot any asymmetry sits,
    complementing the single `asymmetry_diagonal` scalar.
    """
    diff_map = averaged_data - averaged_data.T
    peak = np.max(np.abs(diff_map))
    vmax = peak if peak > 0 else 1.0

    plt.figure(figsize=(10, 8))
    plt.imshow(diff_map, cmap='RdBu_r', origin='lower', aspect='auto', vmin=-vmax, vmax=vmax)
    plt.colorbar(label='Normalized counts favouring LHS (+) vs RHS (-)')

    plt.xlabel("LHS Stud Light Output [A.U]")
    plt.ylabel("RHS Stud Light Output [A.U]")
    plt.title(
        f"{distance} Tile ID {tile_id} ({run}) | 2D Asymmetry Map\n"
        f"(averaged - averaged.T), deviation from mirror symmetry about y=x"
    )

    max_val = diff_map.shape[0]
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1)
    plt.ylim(0, 255)
    plt.xlim(0, 255)

    save_file = Path(save_path) / f"{distance}_{tile_id}_{run}_asymmetry_map.png"
    plt.savefig(save_file, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved asymmetry difference map to {save_file}")


def quantify_stud_asymmetry(folder_path, save_path):
    """
    Finds LHS/RHS file pairs for every (distance, tile_id, run) group and
    computes the PMT-swap-corrected LHS vs RHS stud light-output asymmetry.

    Outputs:
      - stud_asymmetry_summary.csv                    : one row per (distance,
                                                          tile_id, run), with
                                                          both the centroid-
                                                          based and diagonal-
                                                          based asymmetry
      - {distance}_{tile_id}_{run}_averaged.png        : PMT-swap-corrected
                                                          averaged 2D heatmap
      - {distance}_{tile_id}_{run}_asymmetry_map.png   : (averaged - averaged.T)
                                                          difference map, showing
                                                          where the asymmetry sits

    Returns the per-run DataFrame.
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    data_files = list(Path(folder_path).glob("*.2D"))

    groups = defaultdict(dict)
    for f in data_files:
        try:
            distance, tile_id, side, run = parse_2d_filename(f.name)
        except (IndexError, ValueError):
            print(f"Skipping {f.name}: doesn't match expected naming convention")
            continue
        if side not in ("LHS", "RHS"):
            print(f"Skipping {f.name}: side '{side}' is not LHS/RHS")
            continue
        key = (distance, tile_id, run)
        groups[key][side] = f

    results = []
    for (distance, tile_id, run), sides in groups.items():
        if "LHS" not in sides or "RHS" not in sides:
            print(f"Skipping {distance} {tile_id} {run}: missing pair (found {list(sides.keys())})")
            continue

        data_lhs = np.loadtxt(sides["LHS"])
        data_rhs = np.loadtxt(sides["RHS"])

        b_lhsrun, d_lhsrun = compute_weighted_means(data_lhs)
        b_rhsrun, d_rhsrun = compute_weighted_means(data_rhs)

        lhs_stud = np.mean([b_lhsrun, d_rhsrun])
        rhs_stud = np.mean([d_lhsrun, b_rhsrun])

        # Normalised centroid-based asymmetry, bounded in [-1, 1]:
        #   > 0 -> LHS stud brighter, < 0 -> RHS stud brighter
        # (first-moment only -- see compute_diagonal_asymmetry for a
        # full-distribution version)
        asymmetry_centroid = (lhs_stud - rhs_stud) / (lhs_stud + rhs_stud)

        # Diagnostic only: size of the raw B-vs-D channel offset, averaged
        # over both runs. If this is large relative to `asymmetry_centroid`
        # above, a sizeable chunk of any *uncorrected* single-run asymmetry
        # would have been coming from PMT gain, not the tile.
        pmt_gain_effect = ((b_lhsrun - d_lhsrun) + (b_rhsrun - d_rhsrun)) / 2

        result_row = {
            "distance": distance,
            "tile_id": tile_id,
            "run": run,
            "LHS_stud_response": lhs_stud,
            "RHS_stud_response": rhs_stud,
            "asymmetry_centroid": asymmetry_centroid,
            "pmt_gain_effect_estimate": pmt_gain_effect,
        }

        print(f"{distance} {tile_id} {run}: LHS={lhs_stud:.2f}, RHS={rhs_stud:.2f}, "
              f"asymmetry_centroid={asymmetry_centroid:+.4f}, pmt_gain_effect~{pmt_gain_effect:.2f}")

        # --- PMT-swap-corrected averaged 2D heatmap for this pair, plus
        #     the full-distribution diagonal-split asymmetry metric ---
        if data_lhs.shape != data_rhs.T.shape:
            print(
                f"Skipping averaged heatmap for {distance} {tile_id} {run}: "
                f"shape mismatch {data_lhs.shape} vs {data_rhs.T.shape}"
            )
        else:
            total_counts_lhsrun = data_lhs.sum()
            total_counts_rhsrun = data_rhs.sum()
            result_row["total_counts_LHSrun"] = total_counts_lhsrun
            result_row["total_counts_RHSrun"] = total_counts_rhsrun

            if total_counts_rhsrun > 0:
                counts_ratio = total_counts_lhsrun / total_counts_rhsrun
                print(
                    f"  -> total counts: LHS_run={total_counts_lhsrun:.0f}, "
                    f"RHS_run={total_counts_rhsrun:.0f} (ratio={counts_ratio:.3f}); "
                    f"normalizing both to equal weight before averaging"
                )

            # Normalize each run to a probability distribution (sums to 1)
            # BEFORE averaging, so a run with more raw statistics doesn't
            # dominate the combined PMT-swap-corrected distribution.
            norm_lhs = normalize_histogram(data_lhs)
            norm_rhs_t = normalize_histogram(data_rhs.T)
            averaged_data = (norm_lhs + norm_rhs_t) / 2

            plot_averaged_heatmap(averaged_data, distance, tile_id, run, save_path)
            plot_asymmetry_difference_map(averaged_data, distance, tile_id, run, save_path)

            diag_result = compute_diagonal_asymmetry(averaged_data)
            result_row.update(diag_result)

            print(
                f"  -> diagonal split: LHS_brighter={diag_result['counts_lhs_brighter']:.1f}, "
                f"RHS_brighter={diag_result['counts_rhs_brighter']:.1f}, "
                f"on_diagonal={diag_result['counts_on_diagonal']:.1f}, "
                f"asymmetry_diagonal={diag_result['asymmetry_diagonal']:+.4f}"
            )

        results.append(result_row)

    if not results:
        print("No complete LHS/RHS pairs found -- nothing to summarise.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    csv_path = Path(save_path) / "stud_asymmetry_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved per-run asymmetry summary to {csv_path}")

    return df


# Example usage
folder_path = r"C:\Users\thomp\OneDrive - University of Bristol\phys\y3\final_fml_rpt\data\105mm_260701\raw"
save_path = r"C:\Users\thomp\OneDrive - University of Bristol\phys\y3\final_fml_rpt\data\105mm_260701\2D"

plot_2d_files(folder_path, save_path)
quantify_stud_asymmetry(folder_path, save_path)