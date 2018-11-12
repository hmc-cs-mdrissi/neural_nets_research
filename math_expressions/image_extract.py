import os
import argparse
# PArse xml
import xml.etree.ElementTree as ET
import numpy as np
import cv2
# Load / dump data
import pickle

# Reference: The code is based on https://github.com/ThomasLech/CROHME_extractor/blob/master/extract.py
# path = '/Users/yvenica/Desktop/ICFHR_package/CROHME2011_data/CROHME_testGT/CROHME_testGT/' + 'formulaire050-equation070.inkml'

def extract_trace_grps(inkml_file_abs_path):
    trace_grps = []

    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"

    # Find traceGroup wrapper - traceGroup wrapping important traceGroups
    traceGrpWrapper = root.findall(doc_namespace + 'traceGroup')[0]
    traceGroups = traceGrpWrapper.findall(doc_namespace + 'traceGroup')
    for traceGrp in traceGroups:
        traceViews = traceGrp.findall(doc_namespace + 'traceView')
        # Get traceid of traces that refer to latex_class extracted above
        id_traces = [traceView.get('traceDataRef') for traceView in traceViews]
        # Construct pattern object
        trace_grp = []

        # Find traces with referenced by latex_class
        traces = [trace for trace in root.findall(doc_namespace + 'trace') if trace.get('id') in id_traces]
        # Extract trace coords
        for idx, trace in enumerate(traces):
            coords = []
            for coord in trace.text.replace('\n', '').split(','):
                # Remove empty strings from coord list (e.g. ['', '-238', '-91'] -> [-238', '-91'])
                coord = list(filter(None, coord.split(' ')))
                # Unpack coordinates
                x, y = coord[:2]
                # print('{}, {}'.format(x, y))
                if not float(x).is_integer():
                    # Count decimal places of x coordinate
                    d_places = len(x.split('.')[-1])
                    # ! Get rid of decimal places (e.g. '13.5662' -> '135662')
                    # x = float(x) * (10 ** len(x.split('.')[-1]) + 1)
                    x = float(x) * 10000
                else:
                    x = float(x)
                if not float(y).is_integer():
                    # Count decimal places of y coordinate
                    d_places = len(y.split('.')[-1])
                    # ! Get rid of decimal places (e.g. '13.5662' -> '135662')
                    # y = float(y) * (10 ** len(y.split('.')[-1]) + 1)
                    y = float(y) * 10000
                else:
                    y = float(y)

                # Cast x & y coords to integer
                x, y = round(x), round(y)
                coords.append([x, y])
            trace_grp.append(coords)
        trace_grps.append(trace_grp)

        # print('Pattern: {};'.format(pattern))
    return trace_grps
def get_tracegrp_properties(trace_groups):
    x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
    for trace_grp in trace_groups:
      for trace in trace_grp:
        x_min, y_min = np.amin(trace, axis=0)
        x_max, y_max = np.amax(trace, axis=0)
        x_mins.append(x_min)
        x_maxs.append(x_max)
        y_mins.append(y_min)
        y_maxs.append(y_max)
    return min(x_mins), min(y_mins), max(x_maxs) - min(x_mins), max(y_maxs) - min(y_mins)
def get_scale(width, height):
    ratio = width / height
    return ratio
def shift_trace_group(trace_groups, x_min, y_min):
    new_trace_grps = []
    for trace_grp in trace_groups:
      shifted_traces = []
      for trace in trace_grp:
      	shifted_traces.append(np.subtract(trace, [x_min, y_min]))
      new_trace_grps.append(shifted_traces)
    return new_trace_grps
def transform_coord(trace_groups):
	# Convert traces to np.array
	for t_idx in range(len(trace_groups)):
		trace_groups[t_idx] = np.asarray(trace_groups[t_idx])
	# Get properies of a trace group
	x, y, width, height = get_tracegrp_properties(trace_groups)
	# 1. Shift trace_group
	trace_groups = shift_trace_group(trace_groups, x_min=x, y_min=y)
	x, y, width, height = get_tracegrp_properties(trace_groups)
	#print (x,y,width, height)
	return trace_groups
def extract_traces_image(inkml_file_abs_path):
    trace_grps = extract_trace_grps(inkml_file_abs_path)
    trace_grps = transform_coord(trace_grps)
    x, y, width, height = get_tracegrp_properties(trace_grps)
    height_calib = 300
    width_calib = int(get_scale(width, height) * height_calib)
    rescale_ratio = height_calib / height
    canvas = np.zeros((height_calib+5, width_calib+5, 3), dtype = "uint8")
    canvas.fill(255)
    for trace_grp in trace_grps:
        for trace in trace_grp:
            for coord_idx in range(1, len(trace)):
                (x1,y1) = tuple(trace[coord_idx - 1])
                (x2,y2) = tuple(trace[coord_idx])
                cv2.line(canvas, (int(x1 * rescale_ratio)+2, int(y1* rescale_ratio)+2), (int(x2 * rescale_ratio)+2, int(y2* rescale_ratio)+2), color=(0), thickness=1, lineType=cv2.LINE_AA)
    return canvas
#extract_traces_image(path)
