#!/usr/bin/env python3
"""
Q3 Spectral Gap Visualizer

Creates publication-ready 2-panel figure:
- Panel A: Neural kernel (learned structure)
- Panel B: Spectral gap proof (symbol vs floor)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
from rich.console import Console

console = Console()

# --- CONSTANTS FROM Q3 ---
B_MIN = 3.0
T_SYM = 0.06  # 3/50
C_STAR = 1.1  # 11/10 (Archimedean floor)

# --- NEURAL FORMULA (from PySR) ---
def neural_kernel(d):
    """f(d) = 1.20 * cos(0.357*d - 2.05) * exp(-0.0024*d)"""
    return 1.20 * np.cos(0.357 * d - 2.05) * np.exp(-0.0024 * d)

# --- Q3 THEORETICAL FUNCTIONS ---
def a_xi(xi):
    """Archimedean density: log(π) - Re(ψ(1/4 + iπξ))"""
    z = 0.25 + 1j * np.pi * xi
    return np.log(np.pi) - np.real(digamma(z))

def w_window(xi, B=B_MIN, t=T_SYM):
    """Fejér × Heat window"""
    fejer = np.maximum(0, 1 - np.abs(xi) / B)
    heat = np.exp(-4 * (np.pi**2) * t * (xi**2))
    return fejer * heat

def P_A_symbol(theta, num_terms=50):
    """Toeplitz symbol P_A(θ) on period-1 torus"""
    total = 0
    for m in range(-num_terms, num_terms + 1):
        arg = theta + m
        total += a_xi(arg) * w_window(arg)
    return 2 * np.pi * total

# --- EXECUTION ---
def run_gap_visualization():
    console.print("[bold magenta]═══ SPECTRAL GAP VISUALIZER ═══[/]\n")

    # 1. Data Prep
    console.print("[cyan]Computing neural kernel...[/]")
    max_d = 100
    q3_d = np.arange(0, max_d, 0.1)
    neural_k = neural_kernel(q3_d)

    console.print("[cyan]Computing Q3 symbol (high resolution)...[/]")
    N_THETA = 2000
    thetas = np.linspace(-0.5, 0.5, N_THETA)
    symbol_values = np.array([P_A_symbol(th) for th in thetas])

    # Find minimum and the gap
    min_symbol = np.min(symbol_values)
    min_idx = np.argmin(symbol_values)
    theta_min = thetas[min_idx]
    actual_gap = min_symbol - C_STAR

    console.print(f"\n[bold]Q3 Floor Analysis:[/]")
    console.print(f"  min P_A(θ) = {min_symbol:.6f}")
    console.print(f"  c* floor   = {C_STAR}")
    console.print(f"  [green]MARGIN (GAP) = +{actual_gap:.4f}[/]")

    # 2. PLOTTING (2-Panel Publication Ready) - HORIZONTAL LAYOUT
    console.print("\n[cyan]Generating publication figure (horizontal layout)...[/]")

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'figure.dpi': 300,
    })

    # HORIZONTAL: 1 row, 2 columns
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # --- PANEL A: The Learned Structure (Neural Kernel) ---
    ax[0].plot(q3_d, neural_k, 'r-', linewidth=2.5)
    ax[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax[0].fill_between(q3_d, 0, neural_k, where=(neural_k > 0), alpha=0.2, color='red')
    ax[0].fill_between(q3_d, 0, neural_k, where=(neural_k < 0), alpha=0.2, color='blue')

    ax[0].set_title('(A) Learned Correlation Kernel from Attention', fontweight='bold', fontsize=13)
    ax[0].set_ylabel(r'$\mu(d)$', fontsize=12)
    ax[0].set_xlabel('Token Distance d', fontsize=11)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_xlim(0, 80)
    ax[0].set_ylim(-1.3, 1.4)

    # Formula annotation (bottom left, no overlap)
    ax[0].text(0.03, 0.08, r'$\mu(d) = 1.20\cos(0.357d - 2.05)e^{-0.0024d}$',
               transform=ax[0].transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))

    # Period annotation (top left)
    ax[0].text(0.03, 0.92, 'Period ≈ 17.6', transform=ax[0].transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # --- PANEL B: The Spectral Gap (Q3 Symbol) ---
    ax[1].plot(thetas, symbol_values, 'b-', linewidth=2.5)

    # RED LINE (The Floor)
    ax[1].axhline(C_STAR, color='red', linewidth=3, linestyle='-')

    # Safe zone - fill between floor and curve
    ax[1].fill_between(thetas, C_STAR, symbol_values,
                       where=(symbol_values >= C_STAR),
                       color='green', alpha=0.25)

    # Mark the minimum point
    ax[1].plot(theta_min, min_symbol, 'ko', markersize=8, zorder=5)

    # Min annotation - simple, to the right
    ax[1].annotate(f'min = {min_symbol:.2f}',
                   xy=(theta_min, min_symbol),
                   xytext=(theta_min + 0.12, min_symbol + 1.5),
                   fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # Set ylim to show FULL curve
    ax[1].set_ylim(0, max(symbol_values) * 1.1)
    ax[1].set_xlim(-0.5, 0.5)

    ax[1].set_title('(B) Spectral Gap: Symbol Above Floor', fontweight='bold', fontsize=13)
    ax[1].set_ylabel(r'$P_A(\theta)$', fontsize=12)
    ax[1].set_xlabel(r'$\theta$ (period-1 torus)', fontsize=11)
    ax[1].grid(True, alpha=0.3)

    # Clean legend (bottom right, inside plot area)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2.5, label=r'$P_A(\theta)$'),
        Line2D([0], [0], color='red', lw=3, label=f'Floor $c^* = {C_STAR}$'),
        Patch(facecolor='green', alpha=0.25, label='Safe Region')
    ]
    ax[1].legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=10)

    # GAP box - outside plot, below
    gap_text = f"GAP = min $P_A$ − $c^*$ = {min_symbol:.2f} − {C_STAR} = +{actual_gap:.2f}"
    fig.text(0.75, 0.02, gap_text, fontsize=12, fontweight='bold',
             ha='center', color='darkgreen',
             bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='darkgreen', alpha=0.9))

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space at bottom for GAP box
    plt.savefig('Q3_Spectral_Gap_Proof.png', dpi=300, bbox_inches='tight')
    console.print("[green]✓ Saved Q3_Spectral_Gap_Proof.png[/]")

    console.print("\n[bold green]═══ VISUALIZATION COMPLETE ═══[/]")


if __name__ == "__main__":
    run_gap_visualization()
