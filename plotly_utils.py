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

import plotly.express as px
import plotly.colors
import distinctipy
IMAGE = 'image'
HTML = 'html'
font_size = config.get('font_size', 28)

def fix_and_write(fig,
                  filename,
                  traces = None,
                  layout_params = {},
                  output_dir = None,
                  width_factor = 1,
                  height_factor = 1,
                  xaxes={},
                  yaxes={},
                  anotations={},
                  output_type = IMAGE):
    width = config.get('width', 1920) * width_factor
    height = config.get('height', 1280) * height_factor
    if traces is not None:
        fig.update_traces(**traces)
    if 'title' not in layout_params:
         layout_params['title']={'x':0.5,
                                  'font_size' : font_size,
                                 'xanchor': 'center'}
    if 'legend' not in layout_params:
        layout_params['legend'] = dict(
            title_font_family='Courier New',
            font=dict(
                size=28
            )
        )
    fig.update_layout(**layout_params)

    t = dict(tickfont_size=font_size, titlefont_size=font_size)
    t.update(xaxes)
    fig.update_xaxes(**t)
    t =  dict(tickfont_size=font_size, titlefont_size=font_size)
    t.update(yaxes)
    fig.update_yaxes(**t)
    t = dict(font_size=font_size)
    t.update(anotations)
    #fig.update_annotations(**anotations)

    if output_dir is None:
        output_dir = config['output_dir']
    if output_type == IMAGE:
        fn = output_dir / "{}.png".format(filename)
        func = fig.write_image
        kw_args = dict(height=height, width=width)#dict(scale = width_in_mm * 17780.0)
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

def get_colors(N, cmap_name=None, with_faded=False):
    if cmap_name is not None:
        cmap = cm.get_cmap(plt.get_cmap(cmap_name))
        colors = cmap(np.linspace(0, 1, N))
        colors = [[int(x) for x in y] for y in (colors[:, :3]*255).tolist()]
    elif N <= 9:
        colors = [[int(y) for y in x[4:-1].split(",")] for x in plotly.colors.qualitative.Set1]
        # [[int(x[1:][i:i+2], 16) for i in [0,2,4]] for x in plotly.colors.qualitative.Set1]
    elif N<= 24:
        colors = [[int(x[1:][i:i+2], 16) for i in [0,2,4]] for x in plotly.colors.qualitative.Light24]
    elif N < 100:
        colors = distinctipy.get_colors(N)
    else:
        raise AssertionError("get_colors got too large N")
    if with_faded:
        return ['rgb({})'.format(",".join([str(x) for x in y])) for y in colors], \
            ['rgba({},0.2)'.format(",".join([str(x) for x in y])) for y in colors]
    return ['rgb({})'.format(",".join([str(x) for x in y])) for y in colors]