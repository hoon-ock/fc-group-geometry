#!/usr/bin/env python3
"""Manuscript Figure 4: probe accuracy against depth, and the embeddings at two of those depths.

    python visuals/figure/fig4_depth_structure.py [--out-dir DIR]

  f4_depth_structure.pdf    (a) probe accuracy vs layer, with clustermap insets;
                            (b) t-SNE at layers 8 and 24, each coloured two ways

Panel (a) is one plot, five checkpoints, carrying its own evidence: two insets show the base model's
between-class matrix at layer 8 and at layer 24, each joined by a leader to the point on the
base curve it is taken from. Layer 8 sits on the low plateau and layer 24 after the rise, and
the insets say what that rise is -- the groups stop being speckle and fall into blocks. The
claim is that the jump is a change in the geometry, not only in what a classifier scores off it.

Both insets carry ONE leaf order, the base model's at layer 31 -- the same order Figure 2 uses,
so a cell is the same group pair in both insets and in Figure 2. Ordering each layer by its own
dendrogram would be wrong here rather than merely different: clustering always returns some
order, so layer 8 would be drawn with blocks it does not have.

The insets carry no labels of any kind. Each of the 19 cells is about 3pt, so group names are
not an option, and the depth each one is taken from is said by its leader landing on that layer
rather than by a caption over the image. Figure 2 is the labelled reference.

Panel (b) is the same two depths as whole embeddings, and it came from Figure 5, where it was
panels (b) and (c). It belongs beside (a): (a) measures the mid-stack transition as a curve and
marks layers 8 and 24 with the insets, and (b) shows what the points themselves are doing at
exactly those two depths. Models are rows, and the two layers sit side by side as column pairs:
layer 8 in the two left columns, layer 24 in the two right. Within each pair the columns are
the two colourings, template and group, so the crossover is read left-to-right along a single
model's row and a gutter separates the pairs. It is ONE panel, not two -- the pair of layers is
the comparison, so lettering them apart would split the thing being compared.

The ambiguity mass panel that used to be (b) is now Supplementary Figure S4. It was a separate
argument about a different measurement, and it was what forced this page's type down: its row
pitch, not anything here, set the 7pt model names.

PDF, not PNG, like the other scripts here: the destination is a LaTeX manuscript whose figures
are vector. It is not one of the 70 release figures and `run_local.sh` does not build it.

Composed from the existing generators rather than reimplemented. `probe_panel` was lifted out
of `make_visuals_probe.probe_figure` for this figure, `clustermap_panel` is shared with
`fig2_clustermaps.py` so the two manuscript figures cannot disagree on the colour limit, and
`tsne.panel` already took an axes.

Reads `analysis/data/layer_curves.csv`, `surface_baseline.csv`, `between_class_matrix.csv` and
`tsne_coords.csv` through those generators -- never `results/`, which is gitignored.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, HERE)

from _common import BASE, BEHAVIOURAL, FIGURE_LAYER, VISUALS  # noqa: E402
from make_visuals import leaf_order, read_matrices  # noqa: E402
import make_visuals_probe as probe  # noqa: E402
import make_visuals_tsne as tsne  # noqa: E402
import fig2_clustermaps as f2  # noqa: E402
from make_visuals_probe import DISPLAY5  # noqa: E402

DEFAULT_OUT = os.path.join(VISUALS, 'figure')

# 8.8, not 9.0. savefig's tight bbox trims this to ~6.42in wide, so including it at
# \textwidth MAGNIFIES it by 6.5/6.42 -- at HEIGHT 9.0 the included height came out 9.10in
# against the 9.0in printable height of letter at 1in margins, i.e. it did not fit the page it
# was built for. 8.8 lands at 8.90in after that scaling.
WIDTH, HEIGHT = 6.5, 8.8       # included at \textwidth; a dedicated-page figure

# One gridspec, where this figure used to place two `add_axes` rects. The rects existed
# because the old (b) annotated its group names OUTSIDE its axes and so needed ~1.3in of left
# margin that (a) did not; the embedding grid that replaced it needs only room for a row
# label, so the two blocks can share margins and a gridspec is the simpler statement.
#
# The vertical budget, solved from MEASURED furniture rather than guessed, against a 9.0in
# page -- the printable height of letter at 1in margins, which is what makes this a
# dedicated-page figure rather than an oversized one. Fixed costs are 3.07in:
#   top    0.50  (a)'s legend and panel label, which belong to the FIGURE, not to its axes
#   band   0.84  (a)'s x axis and label, then (b)'s layer headers and column titles
#   bottom 1.56  the two stacked colour keys, 20 groups over 10 templates
# That leaves 5.93in of axes, split 1 : 1.82 -- about 2.1in for (a) and 3.83in for (b)'s five
# rows, or 0.77in per embedding cell against the 0.88in they had as their own figure. That
# ~12% is what combining the two figures costs, and it is paid out of the cells rather than
# out of any type size, because `tsne.panel` ends in `set_aspect('equal', adjustable='datalim')`
# -- so a shorter cell shrinks the cluster's footprint rather than distorting it.
TOP, BOTTOM = 0.933, 0.173
LEFT, RIGHT = 0.098, 0.985
HEIGHT_RATIOS = [1.0, 1.92]
HSPACE = 0.28

# (b)'s four scatter columns, left to right: both colourings at layer 8, then both at layer 24.
# `key` is the CSV column `tsne.panel` filters on; the label is the colouring alone, because
# the layer is carried by the spanning header over each pair. The layers are CUTS below -- the
# same two the insets in (a) are taken from, which is the whole reason the two panels sit
# together.
COLS = [
    (8,  'template_index',   'Template'),
    (8,  'functional_group', 'Group'),
    (24, 'template_index',   'Template'),
    (24, 'functional_group', 'Group'),
]

# Width of the empty spacer column between the two layer pairs, relative to one scatter cell.
#
# It is paid for out of cell width that was doing nothing. `tsne.panel` ends in
# `set_aspect('equal', adjustable='datalim')`, so on a cell wider than it is tall matplotlib
# widens the DATA range instead of stretching the points: the cluster is sized by cell HEIGHT
# and the surplus width is padding. Spending some of that padding on a gutter is what makes
# "two left, two right" read as two groups rather than four columns.
GUTTER = 0.5

# Legend geometry. The released grid wraps the 20 functional groups at 7, but that key is the
# widest artist on this page and 7 columns caps the whole figure's type: measured, it is 6.03in
# at 8pt and 6.86in at 9pt against a 6.5in canvas. Six columns is 5.63in at 9pt, which is what
# lets everything else come up too. The 10 templates still fit one row.
FINE_NCOL, TEMPLATE_NCOL = 6, 10

ROWLAB_FS = 9.0
COLTITLE_FS = 9.0
HEADER_FS = 11.0
LEGEND_FS = 9.0
LEGEND_TITLE_FS = 9.5

# The two depths drawn as insets. Layer 8 sits on the probe's low plateau and layer 24 after
# its rise, which is what makes the pair worth drawing.
CUTS = (8, 24)

# Inset geometry, in axes fractions.
#
# Both insets sit LOW, and that is forced by the data rather than chosen. At layer 8 the base
# model is the LOWEST of the five curves and at layer 24 the HIGHEST, so a leader reaching
# layer 8 from above would cross all four other checkpoints; from below it crosses nothing but
# the n-gram baseline rule. The band between the two reference rules -- random guess at 0.20,
# baseline at 0.475 -- is empty across the full width, and these two corners hold no curve
# point at all.
# Height in axes fractions; the WIDTH is derived, not set. `clustermap_panel` holds the matrix
# square, so a box wider than it is tall draws the map in the middle and leaves empty box on
# both sides -- which is where the leaders used to start, detached from the image by ~0.3in.
# Deriving the width from the axes aspect makes the box match its content, so a corner of the
# box IS a corner of the clustermap.
#
# The band between the two reference rules is 0.056 (random guess) to 0.365 (n-gram baseline)
# in axes fractions. With the 'Layer N' captions gone, the whole of it is available to the
# images, which is where the extra size comes from: 0.240 -> 0.285 takes them 0.53in -> 0.63in.
INSET_H = 0.285
INSET_Y = 0.068
# Horizontal CENTRES, so the maps stay put when the derived width changes. The right one sits
# right of its anchor (layer 24 is at 0.749) only so its leader reads as a line rather than a
# vertical tick; with the captions gone it no longer has a title to lean away from, and it is
# pulled back far enough to keep clear of the colourbar at CB_X.
INSET_CX = (0.1225, 0.805)

# Vertical colourbar immediately right of the layer-24 inset. This is what sets that inset's
# x position: the bar, its tick labels and its rotated label together need about 0.09 of the
# axes width between the inset's right edge and the frame.
CB_X, CB_W = 0.900, 0.020

LEADER = dict(arrowstyle='-', color='0.45', linewidth=0.8, shrinkA=1.5, shrinkB=2.5)

# 12, between this file's old 14 and the 10 the embedding grid used in Figure 5. 14 was set
# when (b) was a bar chart whose own type ran to 21pt group names; beside an 11pt 'Layer 8'
# header it now reads as oversized.
PANEL_LAB = dict(fontsize=12, fontweight='bold', va='top', ha='left')

CB_LABEL_FS = 9.5
CB_TICK_FS = 9.0


def retune():
    """Rebind the generators' sizes for composite scale.

    Module globals read when the drawing functions run, so this changes only this figure --
    `make_visuals_probe`'s own released output is unaffected.
    """
    # Down from the 14/13 this figure used when (a) shared the page with a bar chart whose own
    # type ran to 21pt. It now sits above an embedding grid lettered at 9pt, and type two
    # sizes apart on one page reads as two figures pasted together rather than one. These are
    # still at or above the released probe figure's own 13/12/11/10 scaled to this panel: it
    # is ~5.8in wide, the same as there, but 2.1in tall rather than 4.
    #
    # Coming down also buys a one-row legend. The old comment recorded 13pt as the ceiling
    # because a 5-column key measures 6.86in at that size against a 6.5in canvas, forcing
    # ncol=3 and two rows; at 10.5pt the same key is 5.54in and fits on one.
    probe.PROBE_LABEL = 11.0
    probe.PROBE_TICK = 10.0
    probe.PROBE_LEGEND = 10.5
    probe.PROBE_REF = 10.0
    probe.PROBE_MARK = 3.5
    probe.PROBE_LW = 1.3
    # Panel (b). Each embedding is ~0.77in tall here, against the released grid's 1.28in cell
    # and the 0.88in it had as Figure 5's own panel, so the marks come down with the cell.
    tsne.POINT_SIZE = 2.6
    tsne.POINT_ALPHA = 0.75
    # Left at its default 8: the floor labels run rightward from there, which lands them in the
    # gap between the two insets. The earlier guide-line draft had to move them off layer 8.
    probe.PROBE_REF_X = 8


def inset_block(ax, curves, z_by, order):
    """The two clustermap insets, each joined to its point on the base curve."""
    im = None
    pos = ax.get_position()
    # A square box in inches is this wide in axes fractions.
    inset_w = INSET_H * (pos.height * HEIGHT) / (pos.width * WIDTH)
    for layer, cx in zip(CUTS, INSET_CX):
        x0 = cx - inset_w / 2
        cell = ax.inset_axes([x0, INSET_Y, inset_w, INSET_H])
        im = f2.clustermap_panel(cell, z_by[(BASE, layer)], order)
        cell.set_xticks([])
        cell.set_yticks([])

        value = curves[BASE][layer]
        # Leader from the top CORNER nearest the anchor, not the top centre: the title sits at
        # the centre, and a leader leaving from there runs straight through it -- which is
        # exactly what the first render did to 'Layer 24'. Which corner is derived from where
        # the anchor actually falls, so it stays right if the insets move.
        anchor_x = ax.transLimits.transform((layer, value))[0]
        x_start = x0 + inset_w if anchor_x > cx else x0
        # zorder BELOW the curves: at layer 24 base is the top of a converged bundle with no
        # room above it, so the leader has to cross the others and should pass under them.
        ax.annotate('', xy=(layer, value), xycoords='data',
                    xytext=(x_start, INSET_Y + INSET_H), textcoords='axes fraction',
                    arrowprops=LEADER, zorder=1.5)
        # A ring on the anchor, so the leader lands on something the eye can find among five
        # overlapping curves.
        ax.plot([layer], [value], linestyle='none', marker='o', markersize=8.5,
                markerfacecolor='none', markeredgecolor='0.25', markeredgewidth=1.0,
                zorder=8)
    return im


def scatter_block(fig, gs_cell, coords, models, styles):
    """(b) -- five model rows by four columns, split into two layer pairs.

    Moved here from `fig5_crossover.py` rather than reimplemented. Grid column 2 is left empty
    as the gutter, so the four drawn columns are at grid indices 0, 1, 3, 4. Returns
    `axes[row][col]` indexed by the LOGICAL column 0-3, which is what the header and panel-label
    placement measures against.
    """
    sub = gs_cell.subgridspec(len(models), 5, wspace=0.08, hspace=0.06,
                              width_ratios=[1, 1, GUTTER, 1, 1])
    grid_col = (0, 1, 3, 4)
    out = []
    for i, m in enumerate(models):
        row = []
        for j, (layer, key, collab) in enumerate(COLS):
            ax = fig.add_subplot(sub[i, grid_col[j]])
            tsne.panel(ax, coords[(m, layer)], key, styles[key])
            if i == 0:
                ax.set_title(collab, pad=3, fontsize=COLTITLE_FS)
            if j == 0:
                ax.set_ylabel(DISPLAY5[m], labelpad=4, fontsize=ROWLAB_FS)
            row.append(ax)
        out.append(row)
    return out


def key_handles(styles, labels):
    """Legend proxies for one colouring, as `grid_figure` builds them."""
    return [plt.Line2D([], [], linestyle='none', marker=tsne.as_pair(styles[v])[1],
                       markersize=3.0, markeredgewidth=0,
                       color=tsne.as_pair(styles[v])[0], label=tsne.display(v))
            for v in labels]


def _fig_bbox(fig, artist):
    """An artist's drawn extent in figure coordinates."""
    bb = artist.get_window_extent(fig.canvas.get_renderer())
    return bb.transformed(fig.transFigure.inverted())


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--order-layer', type=int, default=FIGURE_LAYER,
                   help=f'layer whose base-model dendrogram orders both insets. Default '
                        f'{FIGURE_LAYER}, matching Figure 2.')
    args = p.parse_args()

    # Figure 2's style, not the probe module's: this figure draws clustermaps and has to match
    # Figure 2's vector output settings exactly. It is also what keeps the embedding grid on
    # `pdf.fonttype: 42` now that it is drawn here -- in Figure 5 it was drawn under the probe
    # module's PNG style, which embedded Type-3 fonts.
    f2.style()
    retune()

    curves, surface = probe.read_curves()
    groups, z_by = read_matrices()
    missing = [L for L in CUTS if (BASE, L) not in z_by]
    if missing or (BASE, args.order_layer) not in z_by:
        raise SystemExit(
            f'no base-model matrix at layer(s) {missing or [args.order_layer]} in '
            'analysis/data/between_class_matrix.csv -- re-run analysis/analyze_taxonomy.py')

    coords = tsne.read_coords()
    models = [m for m in BEHAVIOURAL
              if all((m, layer) in coords for layer, _, _ in COLS)]
    if not models:
        raise SystemExit('no embeddings -- run analysis/compute_tsne_coords.py')

    flat = [r for v in coords.values() for r in v]
    fine_styles, fine_labels = tsne.fine_style(flat)
    tmpl_styles, tmpl_labels = tsne.template_style(flat)
    styles = {'functional_group': fine_styles, 'template_index': tmpl_styles}

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    gs = fig.add_gridspec(2, 1, height_ratios=HEIGHT_RATIOS, left=LEFT, right=RIGHT,
                          top=TOP, bottom=BOTTOM, hspace=HSPACE)

    # (a).
    ax = fig.add_subplot(gs[0, 0])
    probe.probe_panel(ax, curves, surface)
    order = leaf_order(z_by[(BASE, args.order_layer)])
    im = inset_block(ax, curves, z_by, order)

    cax = ax.inset_axes([CB_X, INSET_Y, CB_W, INSET_H])
    cb = fig.colorbar(im, cax=cax, orientation='vertical', extend='both')
    # No label: the tick numbers already carry the scale, and the rotated caption ran the whole
    # height of the inset band for a quantity the figure caption names anyway.
    cb.ax.tick_params(labelsize=CB_TICK_FS, length=2, pad=1)
    cb.outline.set_visible(False)

    # Measured, not a figure-fraction constant: on Figures 3 and 5 those were the single
    # largest source of retune churn, silently misplacing themselves whenever the canvas moved.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, ax.get_position().y1 + 0.006), ncol=len(labels),
               fontsize=probe.PROBE_LEGEND, handlelength=2.4, columnspacing=1.2,
               frameon=False)

    # (b).
    sc = scatter_block(fig, gs[1, 0], coords, models, styles)

    # Everything below is placed from MEASURED positions rather than figure-fraction constants,
    # which is the practice this file's own (a) label used to violate: it was pinned to
    # `A_RECT` and there is no A_RECT any more.
    fig.canvas.draw()
    pos = [[a.get_position() for a in row] for row in sc]
    # Above the column titles, not just above the axes.
    title_top = max(_fig_bbox(fig, sc[0][j].title).y1 for j in range(4))

    # One spanning header per layer pair, centred over the two columns it covers. The panel
    # label sits on the same baseline as the left one, so '(b)' and 'Layer 8' read as one line.
    # There is no '(c)': the two pairs are a single comparison and a single panel.
    head_y = title_top + 0.005
    for j0, layer in ((0, COLS[0][0]), (2, COLS[2][0])):
        mid = (pos[0][j0].x0 + pos[0][j0 + 1].x1) / 2
        fig.text(mid, head_y, f'Layer {layer}', ha='center', va='bottom', fontsize=HEADER_FS)

    lab = dict(PANEL_LAB, va='bottom')
    fig.text(0.004, ax.get_position().y1 + 0.052, '(a)', **lab)
    fig.text(0.004, head_y, '(b)', **lab)

    # The 20-group key is the tall one, so it is placed at the canvas floor first and the
    # 10-template key is stacked on whatever it actually occupied. They describe unrelated
    # encodings -- ten templates against twenty groups -- so merging them would make one
    # oversized box in which neither reads.
    fine = fig.legend(handles=key_handles(fine_styles, fine_labels), loc='lower center',
                      bbox_to_anchor=(0.5, 0.0), ncol=FINE_NCOL, fontsize=LEGEND_FS,
                      handletextpad=0.35, columnspacing=1.0, frameon=False,
                      title='Functional group')
    fine.get_title().set_fontsize(LEGEND_TITLE_FS)
    fig.canvas.draw()
    tmpl = fig.legend(handles=key_handles(tmpl_styles, tmpl_labels), loc='lower center',
                      bbox_to_anchor=(0.5, _fig_bbox(fig, fine).y1 + 0.006),
                      ncol=TEMPLATE_NCOL, fontsize=LEGEND_FS,
                      handletextpad=0.35, columnspacing=1.0, frameon=False,
                      title='Prompt template index')
    tmpl.get_title().set_fontsize(LEGEND_TITLE_FS)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'f4_depth_structure.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
