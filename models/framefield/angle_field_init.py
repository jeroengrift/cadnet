import numpy as np
import shapely.geometry
import shapely.affinity
from PIL import Image, ImageDraw


class AngleFieldInit(object):
    def __init__(self, line_width=2):
        self.line_width = line_width

    def __call__(self, size, polylines):
        return init_angle_field(polylines, (size, size), self.line_width)
    

def init_angle_field(polylines, shape, line_width=2):
    assert type(polylines) == list
    if len(polylines):
        assert type(polylines[0]) == shapely.geometry.LineString

    im = Image.new("L", (shape[1], shape[0]))
    draw = ImageDraw.Draw(im)

    for polyline in polylines:
        coords = np.array(polyline) # pair of two coords        
        edge_vect_array = np.diff(coords, axis=0) # distance between x and y
        edge_angle_array = np.angle(edge_vect_array[:, 1] + 1j * edge_vect_array[:, 0])  # ij coord sys, angle of edge
        neg_indices = np.where(edge_angle_array < 0)
        edge_angle_array[neg_indices] += np.pi
        
        first_uint8_angle = None
        edge = (coords[0], coords[1])
        angle = edge_angle_array[0]
        uint8_angle = int((255 * angle / np.pi).round())
        if first_uint8_angle is None:
            first_uint8_angle = uint8_angle
            
        line = [(edge[0][0], edge[0][1]), (edge[1][0], edge[1][1])]
        draw.line(line, fill=uint8_angle, width=line_width)
        radius = line_width / 2
        draw.ellipse([line[0][0] - radius,
                line[0][1] - radius,
                line[0][0] + radius,
                line[0][1] + radius], fill=uint8_angle, outline=None)

    array = np.array(im)
    return array
