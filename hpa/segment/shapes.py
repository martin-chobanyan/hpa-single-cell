# from imantics import Mask
# from shapely.geometry import Polygon
#
#
# def mask_to_polygons(mask):
#     """Extract geometries from a binary mask
#
#     Parameters
#     ----------
#     mask: numpy.ndarray
#
#     Returns
#     -------
#     list[shapely.geometry.Polygon]
#     """
#     polygons = []
#     for points in Mask(mask).polygons().points:
#         geom = Polygon(points)
#         polygons.append(geom)
#     return polygons
