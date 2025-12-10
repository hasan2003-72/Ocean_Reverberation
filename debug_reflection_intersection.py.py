import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from geometry import definiere_waende, is_point_in_polygon_3d
from config import ALLE_RAEUME
from ism_core import reflect_point_over_plane, _get_line_plane_intersection_general


def main():
    # -------------------------------------------------
    # 1) Raum wählen
    # -------------------------------------------------
    raum = "becken_5x5x5_layered"
    params = ALLE_RAEUME[raum]

    # -------------------------------------------------
    # 2) Wände erzeugen
    # -------------------------------------------------
    walls = definiere_waende(params)
    wall = walls[0]      # Wir nehmen einfach die ERSTE Wand

    verts = wall.vertices
    normal = wall.normal
    p0 = wall.point

    # -------------------------------------------------
    # 3) Beispielpunkte definieren
    # -------------------------------------------------
    # Beispiel: Punkt 30 cm vor der Wand
    p_in = p0 + normal * 0.3

    # Spiegelpunkt berechnen
    p_ref = reflect_point_over_plane(p_in, wall)

    # Segment für Schnittpunkt-Test
    p1 = p_in
    p2 = p_in + np.array([1.0, -0.5, 0.8])   # irgendein Zielpunkt

    # Schnittpunkt berechnen
    intersection = _get_line_plane_intersection_general(p1, p2, wall)

    # -------------------------------------------------
    # 4) Plot
    # -------------------------------------------------
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Wand zeichnen
    poly = Poly3DCollection([verts], alpha=0.4, facecolor='cyan', edgecolor='k')
    ax.add_collection3d(poly)

    # Wand-Eckpunkte
    ax.scatter(verts[:,0], verts[:,1], verts[:,2], s=40, color='blue')

    # Normalenvektor
    ax.quiver(
        p0[0], p0[1], p0[2],
        normal[0], normal[1], normal[2],
        length=0.3, color='red', linewidth=2
    )

    # Originalpunkt
    ax.scatter(p_in[0], p_in[1], p_in[2], s=80, color='yellow', label='p_in')

    # Spiegelpunkt
    ax.scatter(p_ref[0], p_ref[1], p_ref[2], s=80, color='green', label='reflektiert')

    # Linie p1 -> p2
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color='orange', linewidth=2, label='Linie p1->p2')

    # Schnittpunkt
    if intersection is not None:
        ax.scatter(intersection[0], intersection[1], intersection[2],
                   s=120, color='white', edgecolors='black', label='Schnittpunkt')

    # Achsen beschriften
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Reflexion & Schnittpunkt Debug-Plot")

    # Achsen skalieren
    all_pts = np.vstack([verts, p_in, p_ref, p1, p2])
    if intersection is not None:
        all_pts = np.vstack([all_pts, intersection])
    minvals = all_pts.min(axis=0)
    maxvals = all_pts.max(axis=0)
    ax.set_xlim(minvals[0]-0.2, maxvals[0]+0.2)
    ax.set_ylim(minvals[1]-0.2, maxvals[1]+0.2)
    ax.set_zlim(minvals[2]-0.2, maxvals[2]+0.2)

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    