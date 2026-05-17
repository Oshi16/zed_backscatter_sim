"""
plot_results.py — Publication-Quality Figure Generation
=========================================================
This module generates all four figures for the first paper submission.
It is designed to be called from main.py after all three experiments
have been run, receiving their result dictionaries as inputs.

It can also be run standalone by loading previously saved results from
numpy .npz files, which saves hours of re-simulation when you only need
to tweak visual styling before camera-ready submission.

FIGURE DESCRIPTIONS:
    Figure 1 — Main Result (energy_outage_vs_distance)
        Energy outage probability of Node B vs. distance from the AP.
        Overlays all three systems (exp1, exp2, exp3) to show the
        cumulative benefit of each architectural innovation.

    Figure 2 — Trade-off Analysis (outage_tradeoff_vs_tau_A)
        Energy outage probabilities of Node A and Node B as the donor
        threshold τ_A varies. Reveals the optimal operating point where
        total network outage is minimized.

    Figure 3 — Transfer Efficiency Sensitivity (outage_vs_eta_tr)
        Node B energy outage probability vs. WPT transfer efficiency η_tr.
        Shows robustness of the cooperative scheme and motivates the
        choice of RF vs. optical WPT for inter-node energy transfer.

    Figure 4 — Network Scaling (outage_vs_n_poor_nodes)
        Average energy outage of poor nodes as the number of energy-
        constrained nodes sharing from one donor node increases from 1 to 5.
        Tests whether the cooperative benefit degrades with network size.

USAGE:
    # From main.py (preferred):
    from analysis.plot_results import generate_all_figures
    generate_all_figures(results_exp1, results_exp2, results_exp3,
                         results_fig2, results_fig3, results_fig4,
                         output_dir='figures/')

    # Standalone (after saving results with main.py --save-only):
    python plot_results.py --results-dir results/ --output-dir figures/
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse


# ─────────────────────────────────────────────────────────────────────────────
# Global Style Configuration
# ─────────────────────────────────────────────────────────────────────────────

def apply_ieee_style():
    """
    Apply a clean, IEEE Transactions-compatible matplotlib style.

    IEEE papers use 8.5" × 11" pages with single or double-column layouts.
    For a 4-page IEEE WCL submission, figures should be:
        - Single column: 3.5 inches wide
        - Double column (spanning both cols): 7.16 inches wide
    Font sizes in figures should be 8–10pt to remain legible after PDF scaling.

    This function sets global rcParams so all subsequent plt.figure() calls
    inherit these settings automatically.
    """
    matplotlib.rcParams.update({
        # Font family: use Times New Roman to match IEEE LaTeX output.
        # If Times is not installed, 'DejaVu Serif' is a fallback.
        'font.family':          'serif',
        'font.serif':           ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size':            9,
        'axes.titlesize':       9,
        'axes.labelsize':       9,
        'xtick.labelsize':      8,
        'ytick.labelsize':      8,
        'legend.fontsize':      8,
        'legend.framealpha':    0.9,
        'legend.edgecolor':     'gray',

        # Line widths: slightly thicker than matplotlib default so curves
        # remain visible after JPEG compression in conference proceedings.
        'lines.linewidth':      1.8,
        'lines.markersize':     5,

        # Axes appearance
        'axes.spines.top':      False,   # remove top spine for cleaner look
        'axes.spines.right':    False,   # remove right spine
        'axes.grid':            True,
        'grid.linestyle':       '--',
        'grid.alpha':           0.4,
        'grid.color':           '#aaaaaa',

        # Figure background: white for publication, transparent for slides
        'figure.facecolor':     'white',
        'axes.facecolor':       'white',

        # DPI: 300 for print quality, 150 for fast preview
        'figure.dpi':           150,
        'savefig.dpi':          300,
        'savefig.bbox':         'tight',
        'savefig.pad_inches':   0.05,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Main Comparison: Energy Outage Probability vs. Distance
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure1_outage_vs_distance(results_exp1, results_exp2, results_exp3,
                                     output_dir='figures/', show=False):
    """
    Figure 1: Energy outage probability of Node B vs. distance from AP.

    This is the paper's most important figure because it directly shows
    the cumulative benefit of each design layer:
        exp1 (RF-only, no coop)     → highest outage  [worst case]
        exp2 (hybrid, no coop)      → moderate outage [benefit of hybrid EH]
        exp3 (hybrid, cooperative)  → lowest outage   [full proposed system]

    The gap between exp1 and exp2 quantifies the hybrid harvesting gain.
    The gap between exp2 and exp3 quantifies the cooperative sharing gain.
    The gap between exp1 and exp3 quantifies the total system improvement.
    These three numbers belong in your paper's abstract.

    Parameters
    ----------
    results_exp1, results_exp2, results_exp3 : dict
        Output dictionaries from each experiment's run_sweep() function.
    output_dir : str
        Directory where the figure PDF/PNG will be saved.
    show : bool
        If True, display the figure interactively (useful during development).
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))   # single-column IEEE width

    for results in [results_exp1, results_exp2, results_exp3]:
        dist    = results['distance_m_sweep']
        outageB = results['energy_outage_B']
        std_B   = results.get('energy_outage_B_std', np.zeros_like(outageB))
        label   = results['system_label']
        color   = results['system_color']
        ls      = results.get('system_ls', '-')

        ax.plot(dist, outageB, linestyle=ls, color=color, label=label,
                marker='o', markersize=4, markerfacecolor='white',
                markeredgewidth=1.2)

        # Shaded confidence band (±1 std across Monte Carlo trials).
        # This is optional but demonstrates statistical rigor.
        ax.fill_between(dist,
                        np.clip(outageB - std_B, 0, 1),
                        np.clip(outageB + std_B, 0, 1),
                        alpha=0.12, color=color)

    # ── Axis formatting ───────────────────────────────────────────────────────
    ax.set_xlabel('Node B Distance from AP (m)')
    ax.set_ylabel('Energy Outage Probability')
    ax.set_title('Fig. 1: Energy Outage vs. Distance')
    ax.set_xlim(results_exp1['distance_m_sweep'][[0, -1]])
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    # ── Legend ────────────────────────────────────────────────────────────────
    # Place legend inside the plot to avoid wasted whitespace on the right side.
    # IEEE papers have strict page limits — every millimetre matters.
    ax.legend(loc='upper left', handlelength=2.0)

    # ── Annotation: highlight gain at a specific distance ────────────────────
    # Annotating at 10m makes the gain visible and numerically concrete.
    idx_10m = np.argmin(np.abs(results_exp1['distance_m_sweep'] - 10.0))
    y1 = results_exp1['energy_outage_B'][idx_10m]
    y3 = results_exp3['energy_outage_B'][idx_10m]
    gain_pct = (y1 - y3) / y1 * 100 if y1 > 0 else 0

    ax.annotate(f'  {gain_pct:.0f}% reduction\n  at 10m',
                xy=(10.0, y3), xytext=(10.5, y3 + 0.15),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                fontsize=7, color='black')

    plt.tight_layout()
    fpath = os.path.join(output_dir, 'fig1_outage_vs_distance.pdf')
    fig.savefig(fpath)
    fig.savefig(fpath.replace('.pdf', '.png'))
    print(f"  Saved Figure 1 → {fpath}")

    if show:
        plt.show()
    plt.close(fig)
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Trade-off: Donor Threshold τ_A vs. Outage of A and B
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure2_tradeoff_vs_tau_A(results_fig2, output_dir='figures/', show=False):
    """
    Figure 2: Energy outage probability of Node A and Node B as the donor
    threshold τ_A is swept from 0.3 (very generous donor) to 0.9 (conservative).

    PHYSICAL INTUITION:
        When τ_A is low (say 0.3), Node A shares energy whenever its buffer
        exceeds 30% of capacity — it shares very frequently, so Node B rarely
        runs out. But Node A itself is at risk of depletion because it keeps
        giving away energy it might need for itself. This shows as high outage
        for A at low τ_A.

        When τ_A is high (say 0.9), Node A hoards energy until it has 90% of
        capacity — a level it may rarely reach, so sharing happens rarely. Node
        B suffers frequent outages. Node A itself is fine.

        The optimal τ_A is at the crossover point or the minimum of total
        (A + B) outage. This is a key design parameter your paper contributes.

    WHAT results_fig2 SHOULD CONTAIN:
        tau_A_sweep      : array of τ_A values swept
        energy_outage_A  : outage probability of Node A at each τ_A
        energy_outage_B  : outage probability of Node B at each τ_A
        comm_outage_B    : communication outage probability of Node B
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    tau_vals = results_fig2['tau_A_sweep']
    outA     = results_fig2['energy_outage_A']
    outB     = results_fig2['energy_outage_B']
    commB    = results_fig2.get('comm_outage_B', np.zeros_like(outB))

    # Plot outage of A and B on the same axes
    ax.plot(tau_vals, outA, 'b-o', markersize=4, markerfacecolor='white',
            markeredgewidth=1.2, label='Node A (Donor)')
    ax.plot(tau_vals, outB, 'r-s', markersize=4, markerfacecolor='white',
            markeredgewidth=1.2, label='Node B (Receiver)')
    ax.plot(tau_vals, commB, 'g--^', markersize=4, markerfacecolor='white',
            markeredgewidth=1.2, label='Node B Comm. Outage')

    # ── Mark the optimal τ_A (minimum total outage) ───────────────────────────
    total_outage = outA + outB
    idx_opt = np.argmin(total_outage)
    tau_opt = tau_vals[idx_opt]
    ax.axvline(x=tau_opt, color='gray', linestyle=':', linewidth=1.0,
               label=f'Optimal τ_A = {tau_opt:.2f}')
    ax.text(tau_opt + 0.01, 0.85 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1,
            f'τ*_A = {tau_opt:.2f}', fontsize=7, color='gray', va='top')

    ax.set_xlabel('Donor Threshold τ_A (fraction of E_max)')
    ax.set_ylabel('Outage Probability')
    ax.set_title('Fig. 2: Energy-Comm. Trade-off vs. Donor Threshold')
    ax.set_xlim(tau_vals[0] - 0.02, tau_vals[-1] + 0.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='center right', handlelength=1.8)

    plt.tight_layout()
    fpath = os.path.join(output_dir, 'fig2_tradeoff_tau_A.pdf')
    fig.savefig(fpath)
    fig.savefig(fpath.replace('.pdf', '.png'))
    print(f"  Saved Figure 2 → {fpath}")

    if show:
        plt.show()
    plt.close(fig)
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Transfer Efficiency η_tr Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure3_efficiency_sensitivity(results_fig3, output_dir='figures/',
                                         show=False):
    """
    Figure 3: Node B's energy outage probability vs. WPT transfer efficiency η_tr.

    This figure serves two purposes. First, it shows how sensitive the
    cooperative scheme is to the quality of the wireless energy link
    (a lower η_tr means more energy is lost in transit — the RF path
    dissipates it as heat). Second, it motivates the choice between
    RF-based WPT (omnidirectional, η_tr ≈ 0.3–0.5 at short range) and
    optical WPT (directional, η_tr ≈ 0.5–0.8 with LOS). Your paper can
    show two curves — one for each modality — and argue that optical WPT
    is preferred when LOS exists and RF WPT is the fallback.

    WHAT results_fig3 SHOULD CONTAIN:
        eta_tr_sweep      : array of η_tr values (e.g., 0.1 to 0.9)
        outage_B_RF       : outage using RF-based cooperative WPT at each η_tr
        outage_B_optical  : outage using optical cooperative WPT at each η_tr
        outage_B_nocoop   : outage with no cooperation (flat reference line)
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    eta = results_fig3['eta_tr_sweep']

    # RF cooperative WPT
    if 'outage_B_RF' in results_fig3:
        ax.plot(eta, results_fig3['outage_B_RF'], 'b-o', markersize=4,
                markerfacecolor='white', markeredgewidth=1.2,
                label='Cooperative (RF WPT)')

    # Optical cooperative WPT — typically higher efficiency at same distance
    # because the optical beam is directional and does not spread spherically.
    if 'outage_B_optical' in results_fig3:
        ax.plot(eta, results_fig3['outage_B_optical'], 'g-s', markersize=4,
                markerfacecolor='white', markeredgewidth=1.2,
                label='Cooperative (Optical WPT)')

    # Non-cooperative reference (flat line — does not depend on η_tr)
    if 'outage_B_nocoop' in results_fig3:
        nocoop_val = results_fig3['outage_B_nocoop']
        ax.axhline(y=nocoop_val, color='#e74c3c', linestyle='--', linewidth=1.5,
                   label=f'Hybrid, No Coop (ref = {nocoop_val:.2f})')

    # ── Mark realistic operating ranges ──────────────────────────────────────
    # RF WPT at short range (1–3m): η_tr ≈ 0.3–0.5 (shaded region)
    ax.axvspan(0.3, 0.5, alpha=0.08, color='blue',
               label='RF WPT practical range')
    # Optical WPT with LOS: η_tr ≈ 0.5–0.75
    ax.axvspan(0.5, 0.75, alpha=0.08, color='green',
               label='Optical WPT practical range')

    ax.set_xlabel('WPT Transfer Efficiency η_tr')
    ax.set_ylabel('Node B Energy Outage Probability')
    ax.set_title('Fig. 3: Sensitivity to WPT Transfer Efficiency')
    ax.set_xlim(eta[0] - 0.02, eta[-1] + 0.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='upper right', handlelength=1.8, fontsize=7)

    plt.tight_layout()
    fpath = os.path.join(output_dir, 'fig3_efficiency_sensitivity.pdf')
    fig.savefig(fpath)
    fig.savefig(fpath.replace('.pdf', '.png'))
    print(f"  Saved Figure 3 → {fpath}")

    if show:
        plt.show()
    plt.close(fig)
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Network Scaling: Outage vs. Number of Energy-Poor Nodes
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure4_scaling(results_fig4, output_dir='figures/', show=False):
    """
    Figure 4: Average energy outage of poor nodes vs. number of poor nodes
    (1 rich donor node, 1 to 5 poor receiver nodes).

    This figure tests scalability — does the cooperative benefit degrade
    gracefully as more nodes compete for the donor's surplus energy? If the
    outage rises steeply from N=1 to N=5, your cooperative scheme is not
    scalable and reviewers will note this as a limitation. If it rises only
    modestly, you can claim the system is scalable, which supports the ANCHOR
    project's stated goal of large-scale deployment.

    WHAT results_fig4 SHOULD CONTAIN:
        n_poor_nodes       : array [1, 2, 3, 4, 5]
        outage_exp1        : baseline (RF-only) avg outage at each N
        outage_exp2        : hybrid-no-coop avg outage at each N
        outage_exp3        : proposed system avg outage at each N
        std_exp3           : standard deviation of exp3 outage (for error bars)
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    n_nodes = results_fig4['n_poor_nodes']

    systems = [
        ('outage_exp1', 'RF-Only, No Coop', '#e74c3c', '--', None),
        ('outage_exp2', 'Hybrid, No Coop',  '#f39c12', '-.', None),
        ('outage_exp3', 'Proposed System',  '#2ecc71', '-',  'std_exp3'),
    ]

    for key, label, color, ls, std_key in systems:
        if key not in results_fig4:
            continue
        y = results_fig4[key]
        ax.plot(n_nodes, y, linestyle=ls, color=color, label=label,
                marker='D', markersize=5, markerfacecolor='white',
                markeredgewidth=1.2)
        if std_key and std_key in results_fig4:
            std = results_fig4[std_key]
            ax.fill_between(n_nodes,
                            np.clip(y - std, 0, 1),
                            np.clip(y + std, 0, 1),
                            alpha=0.12, color=color)

    ax.set_xlabel('Number of Energy-Poor Nodes (N_B)')
    ax.set_ylabel('Avg. Energy Outage Probability')
    ax.set_title('Fig. 4: Scalability vs. Number of Poor Nodes')
    ax.set_xticks(n_nodes)
    ax.set_xticklabels([str(int(n)) for n in n_nodes])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='upper left', handlelength=1.8)

    plt.tight_layout()
    fpath = os.path.join(output_dir, 'fig4_scaling.pdf')
    fig.savefig(fpath)
    fig.savefig(fpath.replace('.pdf', '.png'))
    print(f"  Saved Figure 4 → {fpath}")

    if show:
        plt.show()
    plt.close(fig)
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
# Bonus Figure — Composite 2×2 Panel (for thesis or conference poster)
# ─────────────────────────────────────────────────────────────────────────────

def plot_composite_panel(results_exp1, results_exp2, results_exp3,
                          results_fig2, results_fig3, results_fig4,
                          output_dir='figures/', show=False):
    """
    Generate a single 2×2 panel combining all four figures.

    This is not suitable for a 4-page IEEE letter (too wide) but is
    very useful for:
        - A PhD thesis chapter figure
        - A conference poster
        - A presentation slide

    It is generated automatically alongside the individual figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(7.16, 5.5))   # full double-column IEEE width
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel (a): Distance sweep ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for res in [results_exp1, results_exp2, results_exp3]:
        ax1.plot(res['distance_m_sweep'], res['energy_outage_B'],
                 linestyle=res.get('system_ls', '-'),
                 color=res['system_color'],
                 label=res['system_label'].split(',')[0],  # shortened label
                 marker='o', markersize=3)
    ax1.set_xlabel('Distance (m)', fontsize=8)
    ax1.set_ylabel('Energy Outage Prob.', fontsize=8)
    ax1.set_title('(a) Outage vs. Distance', fontsize=8)
    ax1.legend(fontsize=6, loc='upper left')
    ax1.set_ylim(-0.02, 1.02)

    # ── Panel (b): Threshold trade-off ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if results_fig2:
        ax2.plot(results_fig2['tau_A_sweep'], results_fig2['energy_outage_A'],
                 'b-o', markersize=3, label='Node A')
        ax2.plot(results_fig2['tau_A_sweep'], results_fig2['energy_outage_B'],
                 'r-s', markersize=3, label='Node B')
    ax2.set_xlabel('Donor Threshold τ_A', fontsize=8)
    ax2.set_ylabel('Outage Probability', fontsize=8)
    ax2.set_title('(b) Trade-off vs. τ_A', fontsize=8)
    ax2.legend(fontsize=6)
    ax2.set_ylim(-0.02, 1.02)

    # ── Panel (c): Transfer efficiency ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if results_fig3:
        if 'outage_B_RF' in results_fig3:
            ax3.plot(results_fig3['eta_tr_sweep'], results_fig3['outage_B_RF'],
                     'b-o', markersize=3, label='RF WPT')
        if 'outage_B_optical' in results_fig3:
            ax3.plot(results_fig3['eta_tr_sweep'], results_fig3['outage_B_optical'],
                     'g-s', markersize=3, label='Optical WPT')
        if 'outage_B_nocoop' in results_fig3:
            ax3.axhline(y=results_fig3['outage_B_nocoop'],
                        color='#e74c3c', linestyle='--', linewidth=1.2,
                        label='No Coop')
    ax3.set_xlabel('Transfer Efficiency η_tr', fontsize=8)
    ax3.set_ylabel('Node B Outage Prob.', fontsize=8)
    ax3.set_title('(c) Sensitivity to η_tr', fontsize=8)
    ax3.legend(fontsize=6)
    ax3.set_ylim(-0.02, 1.02)

    # ── Panel (d): Scaling ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if results_fig4:
        n_nodes = results_fig4['n_poor_nodes']
        color_map = {'outage_exp1': '#e74c3c',
                     'outage_exp2': '#f39c12',
                     'outage_exp3': '#2ecc71'}
        label_map = {'outage_exp1': 'RF-Only',
                     'outage_exp2': 'Hybrid-NoCoop',
                     'outage_exp3': 'Proposed'}
        ls_map    = {'outage_exp1': '--', 'outage_exp2': '-.', 'outage_exp3': '-'}
        for key in ['outage_exp1', 'outage_exp2', 'outage_exp3']:
            if key in results_fig4:
                ax4.plot(n_nodes, results_fig4[key],
                         linestyle=ls_map[key],
                         color=color_map[key],
                         label=label_map[key],
                         marker='D', markersize=3)
    ax4.set_xlabel('Number of Poor Nodes N_B', fontsize=8)
    ax4.set_ylabel('Avg. Outage Prob.', fontsize=8)
    ax4.set_title('(d) Scalability', fontsize=8)
    ax4.legend(fontsize=6)
    ax4.set_ylim(-0.02, 1.02)

    plt.suptitle('Cooperative Hybrid RF–Optical Backscatter Framework: Key Results',
                 fontsize=9, y=1.01)

    fpath = os.path.join(output_dir, 'composite_all_figures.pdf')
    fig.savefig(fpath)
    fig.savefig(fpath.replace('.pdf', '.png'))
    print(f"  Saved Composite Panel → {fpath}")

    if show:
        plt.show()
    plt.close(fig)
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
# Master Function Called by main.py
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(results_exp1, results_exp2, results_exp3,
                          results_fig2, results_fig3, results_fig4,
                          output_dir='figures/', show=False):
    """
    Generate all four paper figures plus the composite panel.

    Call this from main.py after all experiments are complete.

    Parameters
    ----------
    results_exp1 : dict from exp1_baseline.run_sweep()
    results_exp2 : dict from exp2_hybrid_nocoop.run_sweep()
    results_exp3 : dict from exp3_hybrid_coop.run_sweep()
    results_fig2 : dict from main.run_tau_A_sweep()
    results_fig3 : dict from main.run_efficiency_sweep()
    results_fig4 : dict from main.run_scaling_sweep()
    output_dir   : where to save PDF and PNG files
    show         : if True, display each figure interactively after saving
    """
    apply_ieee_style()
    os.makedirs(output_dir, exist_ok=True)

    print("\n── Generating paper figures ──────────────────────────────────────")

    plot_figure1_outage_vs_distance(results_exp1, results_exp2, results_exp3,
                                     output_dir=output_dir, show=show)
    if results_fig2:
        plot_figure2_tradeoff_vs_tau_A(results_fig2, output_dir=output_dir,
                                        show=show)
    if results_fig3:
        plot_figure3_efficiency_sensitivity(results_fig3, output_dir=output_dir,
                                             show=show)
    if results_fig4:
        plot_figure4_scaling(results_fig4, output_dir=output_dir, show=show)

    plot_composite_panel(results_exp1, results_exp2, results_exp3,
                          results_fig2, results_fig3, results_fig4,
                          output_dir=output_dir, show=show)

    print(f"── All figures saved to: {os.path.abspath(output_dir)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone execution (load saved results and regenerate figures)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Regenerate paper figures from saved .npz result files.')
    parser.add_argument('--results-dir', default='results/',
                        help='Directory containing saved .npz result files.')
    parser.add_argument('--output-dir', default='figures/',
                        help='Directory to save generated figures.')
    parser.add_argument('--show', action='store_true',
                        help='Display figures interactively after saving.')
    args = parser.parse_args()

    def load_npz(path):
        """Load an .npz file and return as a plain dict."""
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            return None
        data = np.load(path, allow_pickle=True)
        # np.load with allow_pickle gives NpzFile; convert to dict.
        result = {}
        for key in data.files:
            val = data[key]
            # Scalar arrays (0-d) are converted to Python scalars for metadata keys.
            if val.ndim == 0:
                result[key] = val.item()
            else:
                result[key] = val
        return result

    print("Loading saved results...")
    r1 = load_npz(os.path.join(args.results_dir, 'exp1_distance_sweep.npz'))
    r2 = load_npz(os.path.join(args.results_dir, 'exp2_distance_sweep.npz'))
    r3 = load_npz(os.path.join(args.results_dir, 'exp3_distance_sweep.npz'))
    r_fig2 = load_npz(os.path.join(args.results_dir, 'tau_A_sweep.npz'))
    r_fig3 = load_npz(os.path.join(args.results_dir, 'efficiency_sweep.npz'))
    r_fig4 = load_npz(os.path.join(args.results_dir, 'scaling_sweep.npz'))

    if r1 is None or r2 is None or r3 is None:
        print("ERROR: Core experiment results (exp1/exp2/exp3) are required. "
              "Run main.py first to generate results.")
    else:
        generate_all_figures(r1, r2, r3, r_fig2, r_fig3, r_fig4,
                              output_dir=args.output_dir, show=args.show)
