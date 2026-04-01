# ── Global matplotlib style ────────────────────────────────────────────────────
# Adapted from Economist / IRENA report conventions: clean axis, muted grid,
# emphasis on data not chrome.  Serif titles give a "printed report" feel
# without being heavy.  All charts inherit this automatically.

import matplotlib.pyplot as plt


def apply_style():
    """Apply the project-wide matplotlib rcParams theme."""
    plt.rcParams.update({
        # Typography
        'font.family':        'serif',
        'font.serif':         ['Georgia', 'DejaVu Serif', 'Times New Roman'],
        'font.size':          10.5,
        'axes.titlesize':     13,
        'axes.titleweight':   600,
        'axes.titlepad':      12,
        'axes.labelsize':     10,
        'axes.labelpad':      6,
        'legend.fontsize':    9,
        'legend.title_fontsize': 9.5,
        'xtick.labelsize':    9,
        'ytick.labelsize':    9,

        # Spines
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.spines.left':   True,
        'axes.spines.bottom': True,
        'axes.linewidth':     0.6,

        # Grid — very light, horizontal only by default
        'axes.grid':           True,
        'axes.grid.axis':      'y',
        'grid.color':          '#dfe3e8',
        'grid.linewidth':      0.5,
        'grid.linestyle':      '-',

        # Backgrounds
        'axes.facecolor':     '#fafbfc',
        'figure.facecolor':   'white',

        # Lines
        'lines.linewidth':    1.6,
        'lines.markersize':   4,

        # Ticks
        'xtick.direction':    'out',
        'ytick.direction':    'out',
        'xtick.major.size':   3.5,
        'ytick.major.size':   0,
        'xtick.major.pad':    4,
        'ytick.major.pad':    4,
        'xtick.color':        '#555555',
        'ytick.color':        '#555555',

        # Misc
        'figure.dpi':         110,
        'savefig.dpi':        180,
        'savefig.bbox':       'tight',
        'axes.axisbelow':     True,
    })
