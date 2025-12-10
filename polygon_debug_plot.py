import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  

from config import ALLE_RAEUME
from geometry import definiere_waende, is_point_in_polygon_3d


def main():
    # -------------------------------------------------
    # 1) Szenario auswählen
    # -------------------------------------------------
    raum_name = "becken_5x5x5_layered"   # oder "finnraum", etc.
    params = ALLE_RAEUME[raum_name]

    # -------------------------------------------------
    # 2) Wände (Polygone) erzeugen
    # -------------------------------------------------
    polygone = definiere_waende(params)
    print(f"[INFO] Es wurden {len(polygone)} Wände erzeugt.")

    # Eine Wand auswählen (z.B. Boden)
    wand = polygone[1]
    verts = wand.vertices
    normal = wand.normal
    p0 = wand.point

    print(f"[INFO] Wand: {wand.name}")
    print(f"       Material: {wand.material}")
    print(f"       Normalenvektor: {normal}")

    # -------------------------------------------------
    # 3) Testpunkt definieren (leicht nach innen verschoben)
    # -------------------------------------------------
    center = np.mean(verts, axis=0)
    p_test = center + normal * 0.01   # 1 cm in Normalenrichtung

    print("[INFO] Testpunkt:", p_test)

    inside = is_point_in_polygon_3d(p_test, verts, normal)
    print(f"→ Liegt der Punkt IN der Wand? {inside}")

    # -------------------------------------------------
    # 4) 3D-Plot
    # -------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Wand als 3D-Fläche
    poly = Poly3DCollection([verts], alpha=0.4, facecolor='cyan', edgecolor='k')
    ax.add_collection3d(poly)

    # Eckpunkte plotten
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color='blue', s=40)

    # Normalenvektor plotten
    ax.quiver(
        p0[0], p0[1], p0[2],
        normal[0], normal[1], normal[2],
        color='red', length=0.5, linewidth=2
    )

    # Testpunkt plotten
    color_point = "green" if inside else "red"
    ax.scatter(p_test[0], p_test[1], p_test[2],
               color=color_point, s=100)

    # Achsenlimits anpassen
    margin = 0.5
    ax.set_xlim(verts[:, 0].min() - margin, verts[:, 0].max() + margin)
    ax.set_ylim(verts[:, 1].min() - margin, verts[:, 1].max() + margin)
    ax.set_zlim(verts[:, 2].min() - margin, verts[:, 2].max() + margin)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Wand: {wand.name} – Punkt {'INNEN' if inside else 'AUSSEN'}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()