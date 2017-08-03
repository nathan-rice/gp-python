import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, LinearColorMapper, ColorBar, FuncTickFormatter, FixedTicker, AdaptiveTicker
from itertools import combinations, product, izip_longest
from scipy.stats import skew, kurtosis, gaussian_kde
from collections import Counter
import plotly.graph_objs as go

bar_color = "cornflowerblue"


def plot_histogram(*data):
    title = None
    columns = 3

    def plot_data(d, a):
        if d is None:
            a.axis("off")
            return
        a.hist(d, normed=True, color=bar_color, label=None)
        de = gaussian_kde(d)
        edge = 1
        x = pd.Series(np.linspace(edge * d.min(), d.max() / edge, 100))
        interpolated_y = de(x)
        cumulative = x.apply(lambda v: de.integrate_box_1d(d.min(), v)) * interpolated_y.max()
        a.plot(x, interpolated_y, linestyle='--', color="rebeccapurple", label="PDF")
        a.plot(x, cumulative, linestyle='--', color="dimgray", label="CDF")
        a.fill_between(x, interpolated_y, interpolate=True, color="rebeccapurple", alpha=0.35, zorder=10)
        a.fill_between(x, cumulative, interpolate=True, color="dimgray", alpha=0.125, zorder=15)
        a.set_xlim([x.min(), x.max()])

        a.yaxis.set_ticks_position('none')
        a.yaxis.set_ticklabels([])

    if columns > len(data):
        columns = len(data)
    rows = int(np.ceil(len(data) / columns))

    fig, axes = plt.subplots(rows, columns)

    if columns == 1:
        plot_data(data[0], axes)
        if title:
            axes.set_title(title)
        axes.set_ylabel("Density")
        axes.legend()
    else:
        flat_axes = axes.flatten()
        for d, a in izip_longest(data, flat_axes):
            plot_data(d, a)
        if title:
            for t, a in zip(title, flat_axes):
                a.set_title(t)

    fig.tight_layout()
    return fig


def paired_histogram(x, y, xtitle="", ytitle=""):
    p_color = "cornflowerblue"
    colors = reversed(["#FFFFFF", "#D6EBF2", "#C1E1EC", "#ADD8E6", "#9AC7E7", "#88B6E9",
                       "#76A5EB", "#6495ED", "#647CD8", "#6564C3", "#654BAE", "#663399"])

    colormap = list(zip(np.linspace(0, 1, 12), colors))

    trace1 = go.Scatter(
        x=x, y=y, mode='markers', name='points',
        marker=dict(color="rgb(0.25, 0.25, 0.25)", size=2, opacity=0.5)
    )
    trace2 = go.Histogram2dcontour(
        x=x, y=y, name='density', ncontours=20,
        colorscale=colormap, reversescale=True, showscale=False
    )
    trace3 = go.Histogram(
        x=x, name='x density',
        marker=dict(color=p_color),
        yaxis='y2'
    )
    trace4 = go.Histogram(
        y=y, name='y density', marker=dict(color=p_color),
        xaxis='x2'
    )
    data = [trace1, trace2, trace3, trace4]

    layout = go.Layout(
        showlegend=False,
        autosize=False,
        width=600,
        height=550,
        xaxis=dict(
            domain=[0, 0.85],
            showgrid=False,
            zeroline=False,
            title=xtitle
        ),
        yaxis=dict(
            domain=[0, 0.85],
            showgrid=False,
            zeroline=False,
            title=ytitle
        ),
        margin=dict(
            t=50
        ),
        hovermode='closest',
        bargap=0,
        xaxis2=dict(
            domain=[0.85, 1],
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            domain=[0.85, 1],
            showgrid=False,
            zeroline=False
        )
    )

    return go.Figure(data=data, layout=layout)


def multi_scatter(df):
    data = []
    fill_colors = iter(['lightblue', 'rebeccapurple', 'cornflowerblue', 'indigo', 'slategrey'])

    for (idx, group) in df.groupby(["nodelist"]):
        length = df.shape[0]
        node = [idx] * length
        count = group.groupby("day").count()
        zeros = [0] * length

        data.append(dict(
            type='scatter3d',
            mode='lines',
            x=count.index,  # year loop: in incr. order then in decr. order then years[0]
            y=node,
            z=count["job_db_inx"],
            name=idx,
            line=dict(
                color=next(fill_colors),
                width=4
            ),
        ))

    layout = dict(
        title='',
        showlegend=False,
        scene=dict(
            xaxis=dict(title=''),
            yaxis=dict(title=''),
            zaxis=dict(title=''),
            camera=dict(
                eye=dict(x=-1.7, y=-1.7, z=0.5)
            )
        )
    )

    fig = dict(data=data, layout=layout)
    return fig


def counter_histogram(labels):
    counts = Counter(labels)
    fig, ax = plt.subplots()
    sorted_keys = list(sorted(counts.keys()))
    keymap = {k: i for (i, k) in enumerate(sorted_keys)}
    ax.bar([keymap[k] for k in counts], list(counts.values()), color=bar_color)
    ticklabels = [str(k) for k in sorted_keys]
    ax.set_xticks(range(len(ticklabels)))
    ax.set_xticklabels(ticklabels)

    max_v = max(counts.values())

    def offset(k, v):
        return (k, v + max_v * 0.01)

    for (k, v) in counts.items():
        ax.annotate(str(v), offset(keymap[k], v))

    return fig
