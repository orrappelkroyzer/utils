import sys
from pathlib import Path
local_python_path = str(Path(__file__).parents[1])
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json") 
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd

from openpyxl import load_workbook

import plotly.express as px
IMAGE = 'image'
HTML = 'html'
DEFAULT_FONT_SIZE = config.get('font_size', 28)


def _apply_layout(fig, layout_params, font_size):
    """Apply layout parameters including title, base font size, legend and colorbar."""
    if layout_params is None:
        layout_params = {}

    # Center title and apply font size if no explicit title supplied
    if 'title' not in layout_params:
        layout_params['title'] = {
            'x': 0.5,
            'font': {'size': font_size},
            'xanchor': 'center'
        }

    # Ensure the default layout font uses this size and set legend/colorbar defaults
    base_font = layout_params.get('font', {})
    layout_params['font'] = {
        'size': base_font.get('size', font_size),
        **{k: v for k, v in base_font.items() if k != 'size'}
    }

    # Merge or create legend settings
    legend_cfg = layout_params.get('legend', {})
    legend_font = legend_cfg.get('font', {})
    legend_cfg['font'] = {
        'size': legend_font.get('size', font_size),
        **{k: v for k, v in legend_font.items() if k != 'size'}
    }
    layout_params['legend'] = legend_cfg

    # Merge or create coloraxis_colorbar settings
    cab_cfg = layout_params.get('coloraxis_colorbar', {})
    cab_title = cab_cfg.get('title', {})
    cab_title_font = cab_title.get('font', {})
    cab_title['font'] = {
        'size': cab_title_font.get('size', font_size),
        **{k: v for k, v in cab_title_font.items() if k != 'size'}
    }
    cab_cfg['title'] = cab_title

    cab_tickfont = cab_cfg.get('tickfont', {})
    cab_cfg['tickfont'] = {
        'size': cab_tickfont.get('size', font_size),
        **{k: v for k, v in cab_tickfont.items() if k != 'size'}
    }
    layout_params['coloraxis_colorbar'] = cab_cfg

    fig.update_layout(**layout_params)
    return layout_params


def _apply_axes(fig, xaxes, yaxes, font_size):
    """Apply x/y axis parameters including tick and title fonts."""
    if xaxes is None:
        xaxes = {}
    if yaxes is None:
        yaxes = {}

    t = dict(tickfont={'size': font_size}, title_font={'size': font_size})
    t.update(xaxes)
    fig.update_xaxes(**t)

    t = dict(tickfont={'size': font_size}, title_font={'size': font_size})
    t.update(yaxes)
    fig.update_yaxes(**t)


def _apply_annotations(fig, anotations, font_size):
    """Apply annotations / subplot titles font settings."""
    if anotations is None:
        anotations = {}
    ann_kwargs = dict(font=dict(size=font_size))
    ann_kwargs.update(anotations)
    fig.update_annotations(**ann_kwargs)


def _write_output(fig, filename, output_dir, output_type, width, height):
    """Write figure to disk as image or HTML."""
    if output_dir is None:
        output_dir = config['output_dir']

    if output_type == IMAGE:
        fn = output_dir / "{}.png".format(filename)
        func = fig.write_image
        kw_args = dict(height=height, width=width, engine="orca")
    elif output_type == HTML:
        fn = output_dir / "{}.html".format(filename)
        func = fig.write_html
        kw_args = dict(include_plotlyjs=True)
    else:
        raise AssertionError("received illegal output_type {}".format(output_type))

    logger.info("Writing image to {}".format(fn))
    fn.unlink(missing_ok=True)
    fn.parents[0].mkdir(parents=True, exist_ok=True)
    func(fn, **kw_args)

def write_csv(df, filename, output_dir=None, index=False):
    if output_dir is None:
        output_dir = config['output_dir']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fn = output_dir / "{}.csv".format(filename)
    logger.info("Writing csv to {}".format(fn))
    df.to_csv(fn, index=index)

def write_excel(df, filename, output_dir=None, sheet_name='Sheet1', index=False):
    if output_dir is None:
        output_dir = config['output_dir']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fn = output_dir / "{}.xlsx".format(filename)
    logger.info(f"Writing excel to sheet {sheet_name} in file {fn}")
    try:
        # Try to load the existing Excel file
        with pd.ExcelWriter(fn, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)
    except FileNotFoundError:
        # If the file does not exist, create a new one
        with pd.ExcelWriter(fn, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)
    


def fix_and_write(fig,
                  filename,
                  traces=None,
                  layout_params=None,
                  output_dir=None,
                  width_factor=1,
                  height_factor=1,
                  xaxes=None,
                  yaxes=None,
                  anotations=None,
                  output_type=IMAGE,
                  font_size=None):
    """
    Fix common layout aspects of a Plotly figure and write it to disk.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to modify and save.
    filename : str
        Base filename (without extension).
    traces : dict, optional
        Passed to fig.update_traces(**traces).
    layout_params : dict, optional
        Passed to fig.update_layout(**layout_params).
    output_dir : Path or str, optional
        Directory to write the file into. Defaults to config['output_dir'].
    width_factor, height_factor : float, optional
        Multipliers for base width/height from config.
    xaxes, yaxes : dict, optional
        Extra parameters for fig.update_xaxes / fig.update_yaxes.
    anotations : dict, optional
        Extra parameters for fig.update_annotations.
    output_type : {'image', 'html'}
        Output format.
    font_size : int, optional
        Font size to apply to all text in the figure (axes, titles,
        annotations, legend, colorbar, and layout default font). If None,
        falls back to DEFAULT_FONT_SIZE from config.
    """
    if font_size is None:
        font_size = DEFAULT_FONT_SIZE
    width = config.get('width', 1920) * width_factor
    height = config.get('height', 1280) * height_factor
    if traces is not None:
        fig.update_traces(**traces)

    # Layout and text handling
    layout_params = _apply_layout(fig, layout_params, font_size)
    _apply_axes(fig, xaxes, yaxes, font_size)
    _apply_annotations(fig, anotations, font_size)

    # Finally, write to disk
    _write_output(fig, filename, output_dir, output_type, width, height)

def combine_figures(figs_list):


    # Extract titles from fig1 and fig2
    titles = [fig.layout.title.text if fig.layout.title.text else f"Figure {i+1}" for i, fig in enumerate(figs_list)]

    # Create a subplot figure with the extracted titles
    combined_fig = make_subplots(
        cols=1, rows=len(figs_list),  # 1 row and 2 columns
        subplot_titles=(titles)  # Use extracted titles
    )

    # Add traces from fig1 to the first subplot
    for i, fig in enumerate(figs_list):
        for trace in fig.data:
            combined_fig.add_trace(trace, col=1, row=i+1)
    
    # Update the layout
    combined_fig.update_layout(
        showlegend=False  # Set to True if you want a combined legend
    )

    # Show the combined figure
    return combined_fig


tab10 = [[31, 119, 180],
 [31, 119, 180],
 [255, 127, 14],
 [255, 127, 14],
 [44, 160, 44],
 [44, 160, 44],
 [214, 39, 40],
 [214, 39, 40],
 [148, 103, 189],
 [148, 103, 189],
 [140, 86, 75],
 [140, 86, 75],
 [227, 119, 194],
 [227, 119, 194],
 [127, 127, 127],
 [127, 127, 127],
 [188, 189, 34],
 [188, 189, 34],
 [23, 190, 207],
 [23, 190, 207],
 [23, 190, 207]]
    
tab20 = [[31, 119, 180],
 [174, 199, 232],
 [255, 127, 14],
 [255, 187, 120],
 [44, 160, 44],
 [152, 223, 138],
 [214, 39, 40],
 [255, 152, 150],
 [148, 103, 189],
 [197, 176, 213],
 [140, 86, 75],
 [196, 156, 148],
 [227, 119, 194],
 [247, 182, 210],
 [127, 127, 127],
 [199, 199, 199],
 [188, 189, 34],
 [219, 219, 141],
 [23, 190, 207],
 [158, 218, 229],
 [158, 218, 229]]

#def get_colors(N, cmap_name=None, with_faded=False):
    # if cmap_name is not None:
    #     cmap = cm.get_cmap(plt.get_cmap(cmap_name))
    #     colors = cmap(np.linspace(0, 1, N))
    #     colors = [[int(x) for x in y] for y in (colors[:, :3]*255).tolist()]
    # elif N <= 9:
    #     colors = [[int(y) for y in x[4:-1].split(",")] for x in plotly.colors.qualitative.Set1]
    #     # [[int(x[1:][i:i+2], 16) for i in [0,2,4]] for x in plotly.colors.qualitative.Set1]
    # elif N<= 24:
    #     colors = [[int(x[1:][i:i+2], 16) for i in [0,2,4]] for x in plotly.colors.qualitative.Light24]
    # elif N < 100:
    #     colors = distinctipy.get_colors(N)
    # else:
    #     raise AssertionError("get_colors got too large N")
    # if with_faded:
    #     return ['rgb({})'.format(",".join([str(x) for x in y])) for y in colors], \
    #         ['rgba({},0.2)'.format(",".join([str(x) for x in y])) for y in colors]
    # return ['rgb({})'.format(",".join([str(x) for x in y])) for y in colors]
