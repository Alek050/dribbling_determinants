import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import MultiPoint, Point, Polygon


def get_voronoi_area(row: pd.Series, player: str, field_dim: tuple = (106.0, 68.0)) -> float:
    """
    Calculates the Voronoi area of a player at a given timestamp.

    Args:
        row: A pandas Series representing the tracking data for a single timestamp.
        player: A string representing the column name of the player of interest.
        field_dim: A tuple of two floats representing the dimensions of the soccer field. Default is (106.0, 68.0).

    Returns:
        A float representing the Voronoi area of the player.
    """
    pos_player = Point((row[f"{player}_x"], row[f"{player}_y"]))

    posx_columns = [x for x in row.index if "_x" in x]
    posy_columns = [x for x in row.index if "_y" in x]

    x_positions = np.array(row[posx_columns])
    x_positions = x_positions[~pd.isnull(x_positions)]
    y_positions = np.array(row[posy_columns])
    y_positions = y_positions[~pd.isnull(y_positions)]
    points = np.array([x_positions, y_positions]).T
    x_field, y_field = field_dim[0] / 2.0, field_dim[1] / 2.0
    field_multipoints = MultiPoint(
        [
            Point(i)
            for i in [
                [-x_field, y_field],
                [x_field, y_field],
                [-x_field, -y_field],
                [x_field, -y_field],
            ]
        ]
    )
    field_hull = field_multipoints.convex_hull
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    for region in regions:
        polygon_points = vertices[region]
        shape = list(polygon_points.shape)
        shape[0] += 1
        polygon = Polygon(
            np.append(polygon_points, polygon_points[0]).reshape(*shape)
        ).intersection(field_hull)

        if polygon.contains(pos_player):
            return polygon.area

    # player is outside of the field
    return 0.0


def voronoi_finite_polygons_2d(vor: Voronoi, radius: float = None) -> list:
    """

    Reconstructs a 2D Voronoi diagram with finite-sized polygons.

    Args:
        vor: A Voronoi object representing the 2D Voronoi diagram.
        radius: A float representing the radius of the bounding circle. Default is None.

    Returns:
        A list of 2D polygons representing the finite-sized Voronoi diagram.

    Source:
    https://stackoverflow.com/questions/57385472/how-to-set-a-fixed-outer-boundary-to-voronoi-tessellations
    """

    assert vor.points.shape[1] == 2, "Requires 2D input"

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 4

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)
