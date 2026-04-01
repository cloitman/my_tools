# plotting.py
# Reusable visualization utilities
# Originally used in: Indpendent trading research ML (order book ML), Independent study research for Counterfeit Detection in the CASLAB at Yale (RF scanner heatmaps),
#                     MCubed Materials Science Research at WashU (feature importance), Yale Fluids Lab MENG 363L (error bar plots),
#                     Yale MENG 487L and 488L: Senior Design I and II (grouped comparisons)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb


def correlation_heatmap(df, figsize=(12, 10), annot=True, cmap="YlGnBu"):
    """Plot a correlation heatmap for a DataFrame.

    Args:
        df: pandas DataFrame
        figsize: figure size tuple
        annot: whether to annotate cells with correlation values
        cmap: colormap name

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    sb.heatmap(df.corr(), cmap=cmap, annot=annot, ax=ax)
    plt.tight_layout()
    return fig


def annotated_heatmap(data, row_labels, col_labels, ax=None,
                      cbar_kw={}, cbarlabel="", **kwargs):
    """Create a heatmap from a numpy array with labels and colorbar.

    Includes white gridlines, rotated column labels on top, and a
    properly sized colorbar.

    Args:
        data: 2D numpy array of shape (N, M)
        row_labels: list of length N for row labels
        col_labels: list of length M for column labels
        ax: matplotlib Axes instance (optional, uses current if None)
        cbar_kw: dict of arguments for colorbar
        cbarlabel: label for the colorbar
        **kwargs: forwarded to ax.imshow()

    Returns:
        im: AxesImage object
        cbar: colorbar object
    """
    fontaxes = {
        'family': 'Arial',
        'color': 'black',
        'weight': 'bold',
        'size': 8,
    }

    if not ax:
        ax = plt.gca()

    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.035)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontdict=fontaxes)
    cbar.ax.tick_params(axis='both', which='major', labelsize=7)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis='both', which='major', labelsize=7)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap_cells(im, data=None, valfmt="{x:d}",
                           textcolors=["black", "white"],
                           threshold=None, fontsize=6, **textkw):
    """Add text annotations to each cell of a heatmap.

    Automatically switches text color based on the background value
    to maintain readability.

    Args:
        im: AxesImage from annotated_heatmap()
        data: annotation values (uses image data if None)
        valfmt: format string for values
        textcolors: [below_threshold_color, above_threshold_color]
        threshold: value to switch text color (default: midpoint)
        fontsize: annotation font size
        **textkw: forwarded to ax.text()

    Returns:
        list of Text objects
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None),
                                fontsize=fontsize, **kw)
            texts.append(text)

    return texts


def plot_comparison(actual, predicted, labels=('Actual', 'Predicted'),
                    title=None):
    """Overlay plot comparing actual vs predicted time series.

    Args:
        actual: array-like of actual values
        predicted: array-like of predicted values
        labels: tuple of (actual_label, predicted_label)
        title: optional plot title

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots()
    ax.plot(actual, label=labels[0])
    ax.plot(predicted, label=labels[1])
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_with_errorbars(groups, means, errors, xlabel='', ylabel='',
                        title='', figsize=(8, 5)):
    """Grouped bar chart with error bars.

    Used for comparing experimental groups with uncertainty.

    Args:
        groups: list of group labels
        means: array-like of mean values per group
        errors: array-like of error values per group
        xlabel: x-axis label
        ylabel: y-axis label
        title: plot title
        figsize: figure size tuple

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(groups))
    ax.bar(x, means, yerr=errors, capsize=4, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names, importances, top_n=20,
                            title='Feature Importance', figsize=(10, 8)):
    """Horizontal bar chart of feature importances.

    Sorts features by importance and shows the top N.

    Args:
        feature_names: list of feature name strings
        importances: array-like of importance values (e.g. coefficients)
        top_n: number of top features to display
        title: plot title
        figsize: figure size tuple

    Returns:
        matplotlib figure
    """
    import pandas as pd

    df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    df = df.sort_values('importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(df)), df['importance'].values, align='center')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    plt.tight_layout()
    return fig
