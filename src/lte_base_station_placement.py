""" LTE BS placement – Άσκηση 1 (βελτιωμένη έκδοση) Usage examples:
python lte_bs_placement_e2.py --csv file_12.csv --n_bs 3 --radius 0.7 --sir_db 0
python lte_bs_placement_e2.py --csv file_14.csv --n_bs 5 --restarts 10
python lte_bs_placement_e2.py --csv file_5.csv --k_sweep 2,3,4,5
 Όλα τα αποτελέσματα αποθηκεύονται στο: enhanced_results_2/ """
# ============================================================
# LTE BS placement – Άσκηση 1 (βελτιωμένη έκδοση)
# ============================================================

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d


# ============================================================
#  Βοηθητικές συναρτήσεις – assignments & metrics
# ============================================================

def assign_ues_to_bs(X: np.ndarray, bs_coords: np.ndarray):
    diff = X[:, None, :] - bs_coords[None, :, :]
    dists = np.linalg.norm(diff, axis=2) + 1e-9
    serve_idx = np.argmin(dists, axis=1)
    serve_dist = dists[np.arange(len(X)), serve_idx]
    return serve_idx, serve_dist, dists


def compute_metrics(
    X: np.ndarray,
    bs_coords: np.ndarray,
    radius: float = 0.3,
    sir_threshold_db: float = 0.0,
    pathloss_exp: float = 2.0,
):
    serve_idx, serve_dist, dists = assign_ues_to_bs(X, bs_coords)

    mean_dist = float(np.mean(serve_dist))
    median_dist = float(np.median(serve_dist))

    counts = np.bincount(serve_idx, minlength=len(bs_coords))
    load_std = float(np.std(counts))
    load_mean = float(np.mean(counts))
    load_cv = float(load_std / (load_mean + 1e-9))

    jain = (counts.sum() ** 2) / (len(counts) * np.sum(counts ** 2) + 1e-12)

    covered = serve_dist <= radius
    coverage_pct = float(100.0 * np.mean(covered))
    dead_pct = 100.0 - coverage_pct

    power = 1.0 / (dists ** pathloss_exp)
    signal = power[np.arange(len(X)), serve_idx]
    interference = np.sum(power, axis=1) - signal

    sir_lin = signal / (interference + 1e-12)
    sir_db = 10.0 * np.log10(sir_lin + 1e-12)

    sir_ok_pct = float(100.0 * np.mean(sir_db >= sir_threshold_db))
    interference_prob = float(100.0 * np.mean(sir_db < sir_threshold_db))

    inertia = float(np.sum((X - bs_coords[serve_idx]) ** 2))

    return {
        "mean_distance": mean_dist,
        "median_distance": median_dist,
        "load_counts": counts.tolist(),
        "load_std": load_std,
        "load_mean": load_mean,
        "load_cv": load_cv,
        "jain": float(jain),
        "coverage_pct": coverage_pct,
        "dead_pct": dead_pct,
        "sir_ok_pct": sir_ok_pct,
        "interference_prob": interference_prob,
        "sir_db": sir_db.tolist(),
        "serve_dist": serve_dist.tolist(),
        "serve_idx": serve_idx.tolist(),
        "inertia": inertia,
    }


# ============================================================
#  K-means (manual) with history
# ============================================================

def kmeans_record_iterations(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
    max_iter: int = 50,
    radius: float = 1.5,
    sir_threshold_db: float = 0.0,
    pathloss_exp: float = 2.0,
):
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=k, replace=True)
    centers = X[idx].astype(float)

    history = []
    assigns = np.zeros(len(X), dtype=int)

    for it in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        assigns = np.argmin(dists, axis=1)

        new_centers = np.zeros_like(centers)
        for j in range(k):
            members = X[assigns == j]
            if len(members) == 0:
                pts_d = np.min(dists, axis=1)
                far_idx = np.argmax(pts_d)
                new_centers[j] = X[far_idx]
            else:
                new_centers[j] = members.mean(axis=0)

        met = compute_metrics(
            X,
            new_centers,
            radius=radius,
            sir_threshold_db=sir_threshold_db,
            pathloss_exp=pathloss_exp,
        )

        history.append({
            "iteration": it,
            **{kk: vv for kk, vv in met.items()
               if kk not in ("sir_db", "serve_dist", "serve_idx")},
            "centers": new_centers.copy(),
            "assignments": assigns.copy(),
        })

        if np.max(np.linalg.norm(new_centers - centers, axis=1)) < 1e-4:
            centers = new_centers
            break
        centers = new_centers

    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    assigns = np.argmin(dists, axis=1)
    return centers, assigns, history


def choose_best(run_results):
    best = None
    best_key = None
    for seed, (centers, assigns, history, met) in run_results.items():
        key = (met["inertia"], met["mean_distance"])
        if best is None or key < best_key:
            best = (centers, assigns, history, met)
            best_key = key
    return best


# ============================================================
#  Voronoi neighbors
# ============================================================

def voronoi_neighbors(bs_coords: np.ndarray):
    try:
        vor = Voronoi(bs_coords)
    except Exception as e:
        print("Voronoi neighbor computation failed:", e)
        return {i: [] for i in range(len(bs_coords))}

    neighbors = {i: set() for i in range(len(bs_coords))}
    for a, b in vor.ridge_points:
        neighbors[a].add(b)
        neighbors[b].add(a)
    return {i: sorted(list(s)) for i, s in neighbors.items()}


# ============================================================
#  Save functions
# ============================================================

def save_history_csv(history, outpath: Path):
    if not history:
        return
    keys = [k for k in history[0].keys() if k not in ("centers", "assignments")]
    df = pd.DataFrame([{k: h.get(k) for k in keys} for h in history])
    df.to_csv(outpath, index=False)


# ============================================================
#  Plot results (includes dummy BS for Voronoi when k=2)
# ============================================================

def plot_results(X, bs_coords, assignments, outdir: Path):

    outdir.mkdir(parents=True, exist_ok=True)

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    dx = (x_max - x_min) * 0.05
    dy = (y_max - y_min) * 0.05

    x_lo, x_hi = x_min - dx, x_max + dx
    y_lo, y_hi = y_min - dy, y_max + dy

    # ================= Scatter ==================
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.scatter(X[:, 0], X[:, 1], c=assignments, s=12, alpha=0.8)
    ax1.scatter(bs_coords[:, 0], bs_coords[:, 1], marker="X", s=140)
    ax1.set_xlim(x_lo, x_hi)
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_title("UEs by assigned BS; BS locations")
    fig1.tight_layout()
    fig1.savefig(outdir / "scatter_ues_bs.png", dpi=200)
    plt.close(fig1)

    # ================= Voronoi ==================
       # ================= Voronoi ==================
    try:
        fig2, ax2 = plt.subplots(figsize=(6, 6))

        if len(bs_coords) == 2:
            # -------------------------------------------------
            #  SPECIAL CASE: EXACT VORONOI FOR 2 POINTS
            # -------------------------------------------------

            p1 = bs_coords[0]
            p2 = bs_coords[1]

            # Mid-point
            mid = (p1 + p2) / 2

            # Direction perpendicular to (p2 - p1)
            v = p2 - p1
            perp = np.array([-v[1], v[0]])  # rotate 90 degrees

            # Normalize so the line extends far
            perp = perp / (np.linalg.norm(perp) + 1e-12)

            # Two far points for drawing
            L = 1000  # large number to ensure line covers the plot
            pA = mid - perp * L
            pB = mid + perp * L

            # Plot UEs + BSs
            ax2.scatter(X[:, 0], X[:, 1], s=8, alpha=0.6)
            ax2.scatter(bs_coords[:, 0], bs_coords[:, 1], marker="X", s=140)

            # Draw Voronoi line
            ax2.plot([pA[0], pB[0]], [pA[1], pB[1]],
                     linestyle="--", color="black")

            ax2.set_xlim(x_lo, x_hi)
            ax2.set_ylim(y_lo, y_hi)
            ax2.set_title("Voronoi regions of BSs (2-BS exact)")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")

            fig2.tight_layout()
            fig2.savefig(outdir / "voronoi.png", dpi=200)
            plt.close(fig2)
            return  # IMPORTANT: exit early
        else:
            # Normal Voronoi for 3+ BS
            vor = Voronoi(bs_coords)
            voronoi_plot_2d(
                vor,
                ax=ax2,
                show_vertices=False,
                show_points=False,
            )

            ax2.scatter(X[:, 0], X[:, 1], s=8, alpha=0.6)
            ax2.scatter(bs_coords[:, 0], bs_coords[:, 1], marker="X", s=140)

            ax2.set_xlim(x_lo, x_hi)
            ax2.set_ylim(y_lo, y_hi)
            ax2.set_title("Voronoi regions of BSs")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")

            fig2.tight_layout()
            fig2.savefig(outdir / "voronoi.png", dpi=200)
            plt.close(fig2)

    except Exception as e:
        print("Voronoi plot skipped:", e)


    # ================= Barplot ==================
    counts = np.bincount(assignments, minlength=len(bs_coords))
    plt.figure()
    plt.bar(np.arange(len(bs_coords)), counts)
    plt.xlabel("BS index")
    plt.ylabel("#UEs assigned")
    plt.title("Load per BS")
    plt.tight_layout()
    plt.savefig(outdir / "bar_load.png", dpi=200)
    plt.close()


# ============================================================
#  Distribution Plots
# ============================================================

def plot_distributions(serve_dist, sir_db, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(serve_dist, bins=30)
    plt.title("Distribution of serving distances")
    plt.tight_layout()
    plt.savefig(outdir / "hist_distance.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(sir_db, bins=30)
    plt.title("Distribution of SIR (dB)")
    plt.tight_layout()
    plt.savefig(outdir / "hist_sir.png", dpi=200)
    plt.close()


# ============================================================
#  History Plot
# ============================================================

def plot_history(history, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    iters = [h["iteration"] for h in history]
    metrics = {
        "mean_distance": [h["mean_distance"] for h in history],
        "coverage_pct": [h["coverage_pct"] for h in history],
        "dead_pct": [h["dead_pct"] for h in history],
        "load_cv": [h["load_cv"] for h in history],
        "interference_prob": [h["interference_prob"] for h in history],
        "inertia": [h["inertia"] for h in history],
        "jain": [h["jain"] for h in history],
    }

    for name, series in metrics.items():
        plt.figure()
        plt.plot(iters, series, marker="o")
        plt.title(f"{name} per iteration")
        plt.tight_layout()
        plt.savefig(outdir / f"history_{name}.png", dpi=200)
        plt.close()


# ============================================================
#  Run for a given k (with subfolder)
# ============================================================

def run_for_k(
    X: np.ndarray,
    n_bs: int,
    radius: float,
    sir_db: float,
    pathloss_exp: float,
    max_iter: int,
    restarts: int,
    random_state: int,
    outdir: Path,
):

    outdir.mkdir(parents=True, exist_ok=True)

    run_results = {}
    base_seed = int(random_state)

    for i in range(restarts):
        seed = base_seed + i
        centers, assigns, history = kmeans_record_iterations(
            X, n_bs,
            random_state=seed,
            max_iter=max_iter,
            radius=radius,
            sir_threshold_db=sir_db,
            pathloss_exp=pathloss_exp,
        )
        met = compute_metrics(
            X, centers,
            radius=radius,
            sir_threshold_db=sir_db,
            pathloss_exp=pathloss_exp
        )
        run_results[seed] = (centers, assigns, history, met)

    centers, assigns, history, met = choose_best(run_results)

    summary = {
        "n_bs": n_bs,
        "radius": radius,
        "sir_threshold_db": sir_db,
        "mean_distance": met["mean_distance"],
        "median_distance": met["median_distance"],
        "coverage_pct": met["coverage_pct"],
        "dead_pct": met["dead_pct"],
        "interference_prob": met["interference_prob"],
        "sir_ok_pct": met["sir_ok_pct"],
        "load_std": met["load_std"],
        "load_cv": met["load_cv"],
        "jain": met["jain"],
        "inertia": met["inertia"],
        "load_counts": met["load_counts"],
    }

    pd.Series(summary).to_csv(outdir / "metrics_summary.csv", header=False)

    save_history_csv(history, outdir / "history_metrics.csv")
    plot_results(X, centers, assigns, outdir)
    plot_distributions(np.array(met["serve_dist"]), np.array(met["sir_db"]), outdir)
    plot_history(history, outdir)

    bs_df = pd.DataFrame(centers, columns=["x", "y"])
    bs_df.to_csv(outdir / "bs_coords.csv", index=False)

    neighbors = voronoi_neighbors(centers)
    with open(outdir / "bs_neighbors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bs_index", "neighbors"])
        for i, neigh in neighbors.items():
            w.writerow([i, " ".join(map(str, neigh))])

    print(f"\n=== RESULTS FOR k = {n_bs} ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--n_bs", type=int, default=5)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--sir_db", type=float, default=0.0)
    parser.add_argument("--outdir", type=str, default="enhanced_results_2")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--pathloss_exp", type=float, default=2.0)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--k_sweep", type=str, default="")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("Το CSV πρέπει να περιέχει στήλες 'x' και 'y'")
    X = df[["x", "y"]].to_numpy(dtype=float)

    if args.k_sweep.strip():
        ks = [int(x) for x in args.k_sweep.split(",") if x.strip()]

        rows = []
        for k_val in ks:
            sub_outdir = outdir / f"k_{k_val}"
            summary = run_for_k(
                X, k_val,
                radius=args.radius,
                sir_db=args.sir_db,
                pathloss_exp=args.pathloss_exp,
                max_iter=args.max_iter,
                restarts=args.restarts,
                random_state=args.random_state,
                outdir=sub_outdir
            )
            rows.append({
                "k": k_val,
                "mean_distance": summary["mean_distance"],
                "coverage_pct": summary["coverage_pct"],
                "dead_pct": summary["dead_pct"],
                "sir_ok_pct": summary["sir_ok_pct"],
                "interference_prob": summary["interference_prob"],
                "load_cv": summary["load_cv"],
                "jain": summary["jain"],
                "inertia": summary["inertia"],
            })

        pd.DataFrame(rows).to_csv(outdir / "k_sweep.csv", index=False)
        print(f"\nk_sweep summary saved to {outdir / 'k_sweep.csv'}")
        return

    run_for_k(
        X, args.n_bs,
        radius=args.radius,
        sir_db=args.sir_db,
        pathloss_exp=args.pathloss_exp,
        max_iter=args.max_iter,
        restarts=args.restarts,
        random_state=args.random_state,
        outdir=outdir
    )


if __name__ == "__main__":
    main()
