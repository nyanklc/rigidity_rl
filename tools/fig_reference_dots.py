"""The reference comparison (Table 6.1) as one row of panels, for slides.

Same run and same aggregation as tools/results_chapter.py, drawn in the idiom of
its chapter_grid figures, so every point equals the table.

    PYTHONPATH=. uv run tools/fig_reference_dots.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import results_chapter
from results_chapter import (COLORS, MARKERS, ROOT, SHORT, SIZE_PREFIX, find_run, load_run,
                             rows_by_method, save_pair, style, summarize)

METHODS = ['greedy', 'spectral', 'degree', 'constructive', 'anneal', 's0', 's1', 's2']
PANELS = [('attainment', 'Rigid outputs at the bound', '% of all instances'),
          ('edges', 'Edge count', r'$m$, rigid outputs'),
          ('shape', 'Error coefficient', r'$\varepsilon$, rigid outputs')]


def main():
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
                         'pdf.fonttype': 42})
    path, meta, rows = load_run(find_run(SIZE_PREFIX[10]))
    by = rows_by_method(path, meta, rows)
    stats = {(m, metric): summarize(by[m], metric, seed=123 + j)
             for j, m in enumerate(METHODS) for metric, _, _ in PANELS}
    for m in METHODS:
        print(f"{SHORT[m]:14s}" + ''.join(f"{stats[m, k][0]:9.2f}" for k, _, _ in PANELS))

    def draw(plain):
        fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.9))
        for ax, (metric, title, unit) in zip(axs, PANELS):
            for j, m in enumerate(METHODS):
                y, lo, hi, _ = stats[m, metric]
                ax.errorbar(j, y, yerr=[[max(0, y - lo)], [max(0, hi - y)]], color=COLORS[m],
                            marker=MARKERS[m], linestyle='none', lw=1.2, markersize=4,
                            capsize=2, label=SHORT[m])
            ax.set_title(title)
            ax.set_ylabel(unit)
            ax.set_xticks([])
            ax.set_xlim(-.7, len(METHODS) - .3)
            if metric == 'attainment':
                ax.set_ylim(-3, 103)
            if metric == 'shape':
                ax.set_yscale('log')
            style(ax)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
                   bbox_to_anchor=(.5, 1.01))
        if not plain:
            fig.text(.5, .006, '200 shared $n=10$ formations with random domains, $\\kappa=10$. '
                     'Bound attainment over all instances with Wilson intervals, edge count and '
                     'geometric-mean error over rigid outputs with bootstrap intervals.',
                     ha='center', va='bottom', fontsize=8)
        fig.tight_layout(rect=(0, .05 if not plain else 0, 1, .87), w_pad=1.7)
        return fig

    results_chapter.FIGS = ROOT / 'runs_outputs/presentation'
    save_pair('slide_reference', draw)
    print('written to', results_chapter.FIGS.relative_to(ROOT), 'from', path.relative_to(ROOT))


if __name__ == '__main__':
    main()
