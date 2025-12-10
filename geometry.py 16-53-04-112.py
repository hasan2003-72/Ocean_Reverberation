# geometry.py
# geometry.py – Wand-Erzeugung & Polygon-Tests
# verwendet freq-abhängige Materialien aus config.py
# benutzt Dispatcher: room_walls_from_scenario()

import os
import numpy as np                                                      #vektorrechnung
from typing import List, Dict, Any, Tuple                                     #Type Hints: List, Dict, Any helfen beim Verstehen der Datentypen (Editor/Checker), ändern aber die Laufzeit nicht
from numba import jit
from config import (
    Polygon,
    NPY_MATERIAL_RULES,
    EPSILON,
    USE_RANDOM_POSITIONS,
    Q_FIXED,
    RX_FIXED,
    MIN_Q_RX_DIST,
    MATERIAL_REFLECTIONS,
    MATERIAL_COLORS,
    validate_wall,
    walls_to_polygons,
)


#_______________ Normalenvektor berechen und bedingungen / erwartungen zu erfüllen _________
def calculate_normal_inward(vertices: np.ndarray, center: np.ndarray) -> np.ndarray:    # aus den Eckpunkten einer Wand ihre Flächennormale nach innen zu berechnen.
    """Berechnet die Normalenrichtung, die nach innen (zum Center) zeigt."""
    if len(vertices) < 3:                                                               # weniger als 3 punkte -> kein polygon rückgabe 0 0 0
        return np.zeros(3)
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]                                  # ersten 3 eckpunkte
    vec1, vec2 = v1 - v0, v2 - v0                                                       # bilde 2 richtungsvektoren 
    normal = np.cross(vec1, vec2)                                                       # kreuzprodukt gibt einen vektor der senkrecht zur fläche steht 
    norm_val = np.linalg.norm(normal)                                                   # länge des normalenvektors
    if norm_val < EPSILON:                                                              # wenn die länge fast 0 dann waren dievektoren fast paralel dh fläche degenerieren gib 0 0 0 zurück 
        return np.zeros(3)                                                                  
    normal = normal / norm_val                                                          # einheitsnormale 
    wall_center = np.mean(vertices, axis=0)                                             # mittelpunkt der wand durch mittelpunkt der eckpunkte
    test_vector = center - wall_center                                                  # richtung raumzentrum reigen 
    if np.dot(normal, test_vector) < 0:                                                 # muss 0 zeigen dann zeigt sie auch richtung mitte größer null heißt sie zeigt weg
        normal = -normal                                                                # sonst wird sie einfach umgedreht falls nicht
    return normal


# def validate_wall(w, center=None):
#     """
#     Vereinheitlicht und prüft eine Wanddefinition.
#     Stellt sicher, dass jede Wand die 5 Pflichtfelder korrekt hat:
#       name, material, vertices, normal, point
#     """

#     # ------------------------------------------------------
#     # 1) Pflichtfeld: vertices prüfen
#     # ------------------------------------------------------
#     vertices = np.array(w.get("vertices"))
#     if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
#         raise ValueError(f"[WALL ERROR] Ungültige Vertices in Wand '{w.get('name')}'")

#     w["vertices"] = vertices

#     # ------------------------------------------------------
#     # 2) Pflichtfeld: name
#     # ------------------------------------------------------
#     if "name" not in w:
#         w["name"] = "unnamed_wall"

#     # ------------------------------------------------------
#     # 3) Pflichtfeld: material
#     # ------------------------------------------------------
#     material = w.get("material", None)
#     if material is None or material not in MATERIAL_REFLECTIONS:
#         print(f"[WARNUNG] Wand '{w['name']}' hat unbekanntes Material. Setze Beton als Default.")
#         material = "beton"

#     w["material"] = material

#     # ------------------------------------------------------
#     # 4) Pflichtfeld: point
#     # ------------------------------------------------------
#     point = w.get("point", None)
#     if point is None:
#         point = vertices[0]
#     w["point"] = np.array(point)

#     # ------------------------------------------------------
#     # 5) Normalenvektor prüfen
#     # ------------------------------------------------------
#     normal = w.get("normal", None)

#     if normal is None or np.linalg.norm(normal) < EPSILON:
#         # Normale fehlt → neu berechnen
#         if center is None:
#             center = np.mean(vertices, axis=0)
#         normal = calculate_normal_inward(vertices, center)

#     # Länge checken
#     if np.linalg.norm(normal) < EPSILON:
#         raise ValueError(f"[WALL ERROR] Normale degeneriert in Wand '{w['name']}'")

#     # Einheitlich speichern
#     w["normal"] = normal / np.linalg.norm(normal)

#     return w              #### das war zu teuer
                                                        
@jit(nopython=True, cache=True)
def _is_point_in_polygon_jit(p, vertices, normal, eps=1e-6):
    """
    Kompilierte Version des Point-in-Polygon Tests.
    Läuft ca. 50-100x schneller als reines Python.
    """
    # 1. Ebenen-Check (Ist der Punkt überhaupt auf der Ebene?)
    dx = p[0] - vertices[0, 0]
    dy = p[1] - vertices[0, 1]
    dz = p[2] - vertices[0, 2]
    dist = dx*normal[0] + dy*normal[1] + dz*normal[2]
    
    if np.abs(dist) > eps:
        return False

    # 2. Beste Projektionsachse finden
    nx = np.abs(normal[0])
    ny = np.abs(normal[1])
    nz = np.abs(normal[2])

    if nx > ny and nx > nz:
        i0, i1 = 1, 2 # YZ Ebene
    elif ny > nz:
        i0, i1 = 0, 2 # XZ Ebene
    else:
        i0, i1 = 0, 1 # XY Ebene
    
    px, py = p[i0], p[i1]
    
    # Ray-Casting Algorithmus
    inside = False
    n_poly = len(vertices)
    
    for i in range(n_poly):
        next_i = i + 1
        if next_i == n_poly:
            next_i = 0
            
        v1 = vertices[i]
        v2 = vertices[next_i]
        
        v1x, v1y = v1[i0], v1[i1]
        v2x, v2y = v2[i0], v2[i1]
        
        if ((v1y > py) != (v2y > py)):
            if np.abs(v2y - v1y) > eps:
                x_intersect = (v2x - v1x) * (py - v1y) / (v2y - v1y) + v1x
                if px < x_intersect:
                    inside = not inside
                    
    return inside

def is_point_in_polygon_3d(p: np.ndarray, vertices: np.ndarray, normal: np.ndarray, eps: float = 1e-6) -> bool:
    """Wrapper, der sicherstellt, dass Numba die richtigen Datentypen (floats) bekommt."""
    return _is_point_in_polygon_jit(
        p.astype(np.float64), 
        vertices.astype(np.float64), 
        normal.astype(np.float64), 
        eps
    )



# '''Funktion prüft: Liegt Punkt p innerhalb des Polygons, das in irgendeiner 3D-Ebene liegt?
# Strategie: Prüfe, ob p in der Ebene liegt (Toleranz eps).Projiziere Polygon + Punkt in 2D (indem wir eine Achse weglassen, die am besten zur Ebenennormale passt). Mache 2D-Ray-Casting (klassischer „gerade nach rechts“-Schnittzähl-Trick).'''

# def is_point_in_polygon_3d(p: np.ndarray, vertices: np.ndarray, normal: np.ndarray, eps: float = EPSILON) -> bool:
#     """Prüft, ob ein Punkt auf der Ebene innerhalb der Polygon-Grenzen liegt (2D-Projektion)."""
#     v_to_plane = p - vertices[0]                                            # vektor von einem polygonpkt zur prüfstelle p 
#     if abs(np.dot(v_to_plane, normal)) > eps:                               # skalarprodukt mit normale und misst die abweichung senkrecht zur ebene, if > eps liegt p nicht auf der ebene 
#         return False
    

#     abs_n = np.abs(normal);                                                 # abs beträge normalenkomponenten 
#     skip_axis = np.argmax(abs_n);                     #die dominante Achse (z. B. 2 für z), also die, die wir weglassen, um am besten in 2D zu projizieren.
#     axes = [i for i in range(3) if i != skip_axis]                  # beide verbleibende achsen zB 0 und 1 für x und y
#     p_2d, poly_2d = p[axes], np.array([v[axes] for v in vertices])                  #projeziere p und jedes vertices  element auf diese 2d ebene 

# # Extrahiere die 2D-Koordinaten des Prüfpunkts (x, y),
# # die Anzahl der Eckpunkte des Polygons (n_poly)
# # und setze den Startwert 'inside' auf False.
# # 'inside' wird im Ray-Casting-Verfahren später bei jedem
# # Schnittpunkt der Prüfgeraden mit einer Polygonkante umgeschaltet (True/False).
#     x, y, n_poly, inside = p_2d[0], p_2d[1], len(poly_2d), False                    
    
#     for i in range(n_poly):
#         p1, p2 = poly_2d[i], poly_2d[(i + 1) % n_poly]                  # alle kanten p1 -> p2 durchgehen % modulo wie vorher selbe trick wenns am ende ist wieder bei 0
#         p1x, p1y, p2x, p2y = p1[0], p1[1], p2[0], p2[1]
#                                                                         # prüfe ob horizontale linie auf höhe y die kante schneidet 
#         if ((p1y > y) != (p2y > y)):                                    # falls ja berechnet schnittpunkt x_intersect auf dieser linie 
#             if abs(p2y - p1y) > eps:                                    # Vermeide Division durch fast 0 (Kante fast horizontal)
#                 x_intersect = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
#             # Liegt unser Punkt links vom Schnittpunkt (x < x_intersect),
#             # dann kreuzt der „Strahl nach rechts“ die Kante.
#             # Wir schalten den Zustand inside (True/False) jedes Mal um.
#                 if x < x_intersect:
#                     inside = not inside

#     # Nach der Schleife gilt:
#     # inside = True  → der Punkt liegt innerhalb des Polygons (ungerade Schnittzahl)
#     # inside = False → der Punkt liegt außerhalb (gerade Schnittzahl)                
#     return inside

# _______ define_*_room Funktionen 

#-_______________Einfacher_raum
def define_box_walls(dimensions: np.ndarray) -> List[Dict[str, Any]]:
    L, B, H = dimensions    
    P = {                                                                                   # 8 eckpunkte                
        'p0': np.array([0., 0., 0.]), 'p1': np.array([L, 0., 0.]),                                              
        'p2': np.array([L, B, 0.]), 'p3': np.array([0., B, 0.]), 
        'p4': np.array([0., 0., H]), 'p5': np.array([L, 0., H]), 
        'p6': np.array([L, B, H]), 'p7': np.array([0., B, H]),
    }
    center = np.array([L/2, B/2, H/2])                                                      # einfacher raummittelpunkt
    walls_raw = [
        ('z_min', [P['p0'], P['p1'], P['p2'], P['p3']]), ('z_max', [P['p4'], P['p7'], P['p6'], P['p5']]),                       # liste von wänden- jede wand: name und eckpunkte
        ('y_min', [P['p0'], P['p1'], P['p5'], P['p4']]), ('y_max', [P['p3'], P['p2'], P['p6'], P['p7']]),
        ('x_min', [P['p0'], P['p3'], P['p7'], P['p4']]), ('x_max', [P['p1'], P['p2'], P['p6'], P['p5']]),
    ]
    walls = []                                                                                                                  # Rechne Normalenrichtung nach innen
    for name, vertices in walls_raw:                                                                                            # erzeugt ein dict: name alpha ecken, 
        normal = calculate_normal_inward(np.array(vertices), center)                                                            # normale, points erster eckpunkt
        walls.append({'name': name, 'vertices': vertices, 'normal': normal, 'point': vertices[0]}
)
    return walls                                                                                                                # Rückgabe: Liste von Wand-Dictionarie



def define_wedge_room(L: float, B: float, H_min: float, H_max: float) -> List[Dict[str, Any]]:                # keilraum grundfläche rechteck
    P = {                                                                                                                           #  aber hat eine zmax höhe und zmin
        'p0': np.array([0., 0., 0.]), 'p1': np.array([L, 0., 0.]),
        'p2': np.array([L, B, 0.]), 'p3': np.array([0., B, 0.]),
        'p4': np.array([0., 0., H_min]), 'p5': np.array([L, 0., H_max]),
        'p6': np.array([L, B, H_max]), 'p7': np.array([0., B, H_min]),
    }
    center = np.array([L/2, B/2, (H_min + H_max)/3])                                                                      # Grobe Schätzung für Raumzentrum, etwas näher zum Boden (durch /3)
    walls_raw = [
        ('Boden', [P['p0'], P['p1'], P['p2'], P['p3']]),                                                                    # boden schräge decke und alle 4 seiten definiert
        ('Decke_Schraeg', [P['p4'], P['p5'], P['p6'], P['p7']]), 
        ('Y_min', [P['p0'], P['p1'], P['p5'], P['p4']]),
        ('Y_max', [P['p3'], P['p2'], P['p6'], P['p7']]),
        ('X_min', [P['p0'], P['p3'], P['p7'], P['p4']]),
        ('X_max', [P['p1'], P['p2'], P['p6'], P['p5']]),
    ]
    walls = []                                                                                                              #erneut Normale berechnen und wand dict aufbauen
    for name, vertices in walls_raw:
        normal = calculate_normal_inward(np.array(vertices), center)
        walls.append({'name': name, 'vertices': vertices, 'normal': normal, 'point': vertices[0]}
)
    return walls


def define_trapezoidal_room(L: float, H: float, B_bottom: float, B_top: float) -> List[Dict[str, Any]]:          
    Y_b, Y_t = B_bottom / 2.0, B_top / 2.0                                                                                  # unterscheidung der breiten oben und unten durch unterschiedl. breiten entsteht schräge
    P_trap = {
        'p0': np.array([0., -Y_b, 0.]),  'p1': np.array([L, -Y_b, 0.]),   
        'p2': np.array([L, Y_b, 0.]),    'p3': np.array([0., Y_b, 0.]),   
        'p4': np.array([0., -Y_t, H]),   'p5': np.array([L, -Y_t, H]),    
        'p6': np.array([L, Y_t, H]),     'p7': np.array([0., Y_t, H]),    
    }
    center = np.array([L/2, 0.0, H/2])                                                                                      # raumzentrum
    walls_raw = [
        ('Boden', [P_trap['p0'], P_trap['p1'], P_trap['p2'], P_trap['p3']]),                                        # Sechs Wände des Trapezraums (Boden, Decke, 2 Seiten, 2 schrägwände)
        ('Decke', [P_trap['p4'], P_trap['p7'], P_trap['p6'], P_trap['p5']]),
        ('Seite_Links', [P_trap['p0'], P_trap['p1'], P_trap['p5'], P_trap['p4']]),
        ('Seite_Rechts', [P_trap['p3'], P_trap['p2'], P_trap['p6'], P_trap['p7']]),
        ('Rückwand', [P_trap['p0'], P_trap['p3'], P_trap['p7'], P_trap['p4']]),
        ('Vorderwand', [P_trap['p1'], P_trap['p2'], P_trap['p6'], P_trap['p5']]),
    ]
    walls = []                                                                                                          #erneut Normale berechnen und wand dict aufbauen
    for name, vertices in walls_raw:
        normal = calculate_normal_inward(np.array(vertices), center)
        walls.append({'name': name, 'vertices': vertices, 'normal': normal, 'point': vertices[0]})
    return walls



def define_heptagonal_room(H: float) -> List[Dict[str, Any]]:                                                # heptagonaler raum definition
    P_base = [                                                                                                                  # P_base hat 7 punkte in ebene z=0 
        np.array([0.0, 3.0, 0.0]), np.array([5.0, 6.0, 0.0]), np.array([10.0, 6.0, 0.0]),                                           
        np.array([12.0, 4.0, 0.0]), np.array([10.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), 
        np.array([2.0, 2.0, 0.0]),
    ]
    P_top = [p + np.array([0., 0., H]) for p in P_base]                                                                         # für jeden pkt füge höhe H auf z, dies wird zur decke 
    all_vertices = np.vstack(P_base + P_top)                                                                                    # alle eckpunkte erfassen base+top
    min_c, max_c = np.min(all_vertices, axis=0), np.max(all_vertices, axis=0)                                                   # bestimmung minimaler und maximaler x y z wert
    center = (min_c + max_c) / 2.0                                                                                              # mittelpunkt der box !IST NICHT 100% MITTE!
    walls_raw = [('Boden', P_base), ('Decke', P_top[::-1])]                                                                     # hier werden punkte umgedreht damit normalenrichtung passt 
    
    num_sides = len(P_base)                                                                                                     # Anzahl der eckpunkte vom boden  
    for i in range(num_sides):                                                                                                  # Schleife über alle seiten des 7eck - 4 pkt bilden eine seitenwand 
        p_b1, p_b2 = P_base[i], P_base[(i + 1) % num_sides]                                                                     # der letzte Punkt wieder mit dem ersten verbunden wird (also die Figur geschlossen wird)
        p_t2, p_t1 = P_top[(i + 1) % num_sides], P_top[i]                                                                       # % ist modulo operator er sorgt dafür dass nach i wieder 0 genommen wird, mann muss also garnicht wissen wie viele seiten das hat 
        walls_raw.append((f'Seite_{i+1}', [p_b1, p_b2, p_t2, p_t1]))                                                            # hier werden dann die 7 seiten wände erzeugt Seite_0+1 ... Seite_i+1
    
    walls = []                                                                                                                  # wie bei box wird die normale berechnet an wand/ebene
    for name, vertices in walls_raw:                                                                                            # und Dict pro wand erzeugen
        vertices_np = np.array(vertices) 
        normal = calculate_normal_inward(vertices_np, center)
        walls.append({'name': name, 'vertices': vertices, 'normal': normal, 'point': vertices[0]})
    return walls



def define_room_5x5x5_layered() -> List[Dict[str, Any]]:
    """
    5x5x5 m Raum:
      0-3 m: Regupol (Gummimatten)
      3-5 m: Beton
      Decke: Wasseroberfläche
    """
    walls = []

    # Eckpunkte des 5x5x5-Würfels
    L = B = H = 5.0

    # Grundpunkte (Boden)
    p0 = np.array([0., 0., 0.])
    p1 = np.array([L, 0., 0.])
    p2 = np.array([L, B, 0.])
    p3 = np.array([0., B, 0.])

    # Zwischenpunkte bei 3m Höhe
    p0m = np.array([0., 0., 3.])
    p1m = np.array([L, 0., 3.])
    p2m = np.array([L, B, 3.])
    p3m = np.array([0., B, 3.])

    # Decke bei 5 m
    p0t = np.array([0., 0., H])
    p1t = np.array([L, 0., H])
    p2t = np.array([L, B, H])
    p3t = np.array([0., B, H])

    # Mittelpunkt des Raums (für Normalen)
    center = np.array([L/2, B/2, H/2])

    # -------------------------------
    # 1) Boden (0 m): Regupol
    # -------------------------------
    verts = [p0, p1, p2, p3]
    normal = calculate_normal_inward(np.array(verts), center)
    walls.append({
        'name': 'Boden',
        'material': 'regupol_fx',
        'vertices': verts,
        'normal': normal,
        'point': verts[0]
    })

    # -------------------------------
    # 2) Seitenwände unten (0–3 m): Regupol
    # -------------------------------
    bottom_sections = [
        ('Seite_Xmin_unten', [p0, p3, p3m, p0m]),
        ('Seite_Xmax_unten', [p1, p2, p2m, p1m]),
        ('Seite_Ymin_unten', [p0, p1, p1m, p0m]),
        ('Seite_Ymax_unten', [p3, p2, p2m, p3m]),
    ]
    for name, verts in bottom_sections:
        normal = calculate_normal_inward(np.array(verts), center)
        walls.append({
            'name': name,
            'material': 'regupol_fx',
            'vertices': verts,
            'normal': normal,
            'point': verts[0]
        })

    # -------------------------------
    # 3) Seitenwände oben (3–5 m): Beton
    # -------------------------------
    top_sections = [
        ('Seite_Xmin_oben', [p0m, p3m, p3t, p0t]),
        ('Seite_Xmax_oben', [p1m, p2m, p2t, p1t]),
        ('Seite_Ymin_oben', [p0m, p1m, p1t, p0t]),
        ('Seite_Ymax_oben', [p3m, p2m, p2t, p3t]),
    ]
    for name, verts in top_sections:
        normal = calculate_normal_inward(np.array(verts), center)
        walls.append({
            'name': name,
            'material': 'beton',
            'vertices': verts,
            'normal': normal,
            'point': verts[0]
        })

    # -------------------------------
    # 4) Decke (5 m): Wasseroberfläche
    # -------------------------------
    verts = [p0t, p3t, p2t, p1t]  # Reihenfolge gedreht = Normale nach unten
    normal = calculate_normal_inward(np.array(verts), center)
    walls.append({
        'name': 'Wasseroberflaeche',
        'material': 'water_surface',
        'vertices': verts,
        'normal': normal,
        'point': verts[0]
    })

    return walls



def define_npy_room(file_names: List[str], base_path: str) -> List[Dict[str, Any]]:
    """
    Lädt Geometriedaten aus NPY-Dateien und weist Materialien zu.
    """
    walls_raw = []
    all_vertices_combined = []

    # -----------------------------
    # 1) Dateien einlesen
    # -----------------------------
    for i, filename in enumerate(file_names):
        file_path = os.path.join(base_path, filename)
        print(f"[INFO] Lese Geometrie aus: {file_path}")

        try:
            raw = np.load(file_path, allow_pickle=True)

            # Fall B: mehrere Polygone
            if raw.dtype == object and raw.ndim == 1:
                for j, poly in enumerate(raw):
                    vertices = np.array(poly, dtype=float)

                    if vertices.ndim != 2 or vertices.shape[1] != 3:
                        continue

                    all_vertices_combined.append(vertices)
                    walls_raw.append({
                        'name': f'NPY_Wand_{i+1}_{j+1}',
                        'material': None,
                        'vertices': vertices,
                        'normal': None,
                        'point': None
                    })

            # Fall A: einzelnes Polygon
            else:
                vertices = raw.astype(float)

                if vertices.ndim != 2 or vertices.shape[1] != 3:
                    continue

                all_vertices_combined.append(vertices)
                walls_raw.append({
                    'name': f'NPY_Wand_{i+1}',
                    'material': None,
                    'vertices': vertices,
                    'normal': None,
                    'point': None
                })

        except Exception as e:
            print(f"[FEHLER] {e}")

    if not all_vertices_combined:
        print("[WARNUNG] Keine Wände geladen.")
        return []

    # -----------------------------
    # 2) Raumzentrum bestimmen
    # -----------------------------
    all_vertices = np.vstack(all_vertices_combined)
    min_c, max_c = np.min(all_vertices, axis=0), np.max(all_vertices, axis=0)
    center = (min_c + max_c) / 2.0
    print(f"[INFO] Raumzentrum: {center}")

    # -----------------------------
    # 3) Normalen berechnen
    # -----------------------------
    walls = []
    for w in walls_raw:
        verts = w['vertices']
        normal = calculate_normal_inward(verts, center)

        if np.linalg.norm(normal) > EPSILON:
            w['normal'] = normal
            w['point'] = verts[0]
            walls.append(w)

    # ----------------------------------------
    # 4) MATERIAL-ZUWEISUNG
    # ----------------------------------------
    TOL = 0.05

    z_min = np.min(all_vertices[:,2])
    z_max = np.max(all_vertices[:,2])

    boden_walls = []
    decken_walls = []
    seiten_walls = []

    for w in walls:
        z_vals = w['vertices'][:,2]

        # Boden
        if np.all(z_vals <= z_min + TOL):
            w['material'] = NPY_MATERIAL_RULES["boden_material"]
            boden_walls.append(w)
            continue

        # Decke
        if np.all(z_vals >= z_max - TOL):
            w['material'] = NPY_MATERIAL_RULES["decke"]
            decken_walls.append(w)
            continue

        # Seiten
        seiten_walls.append(w)

    # Seiten 50/50 aufteilen
    mats = NPY_MATERIAL_RULES["wand_materials"]
    half = len(seiten_walls) // 2

    for i, w in enumerate(seiten_walls):
        w['material'] = mats[0] if i < half else mats[1]

    print(f"[INFO] Insgesamt {len(walls)} Wände geladen.")
    return walls



def generate_random_positions(walls: List[Dict[str, Any]], min_wall_distance: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:        # random positionen für quelle und empfänger
    all_vertices = np.vstack([w['vertices'] for w in walls])                                                                        #sammelt eckpunkte alle wände dessen min ud max koordinaten
    min_c, max_c = np.min(all_vertices, axis=0), np.max(all_vertices, axis=0)                                                       # eine art bounding box wo sie sich befinden dürfen
    L_min, B_min, H_min = min_c; L_max, B_max, H_max = max_c
    tol = min_wall_distance                                                                                                         # mindestabstand p zur wand einhalten
    def get_valid_point():
        MAX_ATTEMPTS = 500
        attempts = 0
                                                                                                      # innere hilfsfunktion, ziehe zufällige xyz innerhalb BBox aber mit abstand tol
        while True:                                                                                                         
            x = np.random.uniform(L_min + tol, L_max - tol)                                                                 # samplet p=[x,y,z] innerhalb der bbox mit randabstand tol
            y = np.random.uniform(B_min + tol, B_max - tol)
            z = np.random.uniform(H_min + tol, H_max - tol)
            p = np.array([x, y, z])                                                                                         # p punkt für jede wand gilt 
            is_valid = True
            for wall in walls:
                v = p - wall['point']                                                                                       # vektor (wand zu p)  
                # Abstand des Punktes zur Wand gemessen entlang der Normale
                dist = np.dot(v, wall['normal'])

                # Nur wenn dist < min_wall_distance ist → zu nah
                if dist < min_wall_distance:
                    is_valid = False
                    break
                                           # wie weit p in richtung der innen normale liegt, durch die normale nach innen stellt man sicher dass es auch nach innen zeigt
            if is_valid: 
                return p
            attempts += 1
            if attempts >= MAX_ATTEMPTS:
                raise Exception("Konnte nach 500 Versuchen keinen gültigen Punkt im Raum finden. Normalenprüfung oder Raumgeometrie fehlerhaft.") # <--- FEHLER BEI NICHT-FINDEN                                                                                               # p wiedergeben wenn es nicht zu nah an der wand liegt oder eben nicht weil evtl sogar außerhalb befindet 
    q_pos = get_valid_point()                                                                                       # zufälliger pkt für Q, in einer schleife für E, abernur bid abstand Q bis E größer 2* tol
    while True:
        e_pos = get_valid_point()
        if np.linalg.norm(q_pos - e_pos) > MIN_Q_RX_DIST  : return q_pos, e_pos                                                   # Rückgabe: (q_pos, e_pos) als Tuple[np.ndarray, np.ndarray]


def get_source_and_receiver_positions(walls: List[Dict[str, Any]], min_wall_distance: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gibt (q_pos, rx_pos) zurück, je nachdem ob USE_RANDOM_POSITIONS True oder False ist.
    """
    if USE_RANDOM_POSITIONS:
        q_pos, rx_pos = generate_random_positions(walls, min_wall_distance)
        print("[INFO] Zufällige Positionen gewählt:")
        print("   Quelle (Q):", q_pos)
        print("   Empfänger (Rx):", rx_pos)
    else:
        q_pos, rx_pos = Q_FIXED, RX_FIXED
        print("[INFO] Feste Positionen verwendet:")
        print("   Quelle (Q):", q_pos)
        print("   Empfänger (Rx):", rx_pos)
    return q_pos, rx_pos

# def walls_to_polygons(walls_dict: List[Dict[str, Any]]) -> List[Polygon]:
#     """
#     Konvertiert die generischen Wand-Dictionaries in Polygon-Objekte.
#     KORRIGIERT: Übergibt jetzt auch den Material-Namen.
#     """
#     polygons = []

#     for w in walls_dict:

#         w = validate_wall(w)
#         # 1. Materialnamen holen (Fallback auf 'beton', falls keiner da ist)
#         mat_name = w.get('material', 'beton')
        
#         # 2. Passende Reflexionskurve dazu holen
#         # Falls das Material unbekannt ist, nehmen wir Beton als Standard
#         try:
#             reflection_curve = get_material_reflection(mat_name)
#         except ValueError:
#             # Falls Tippfehler im Namen -> Fallback Beton
#             mat_name = 'beton' 
#             reflection_curve = get_material_reflection('beton')

#         vertices_np = np.array(w['vertices'])

#         # 3. Polygon erstellen UND material übergeben
#         polygons.append(Polygon(
#             name=w['name'],
#             vertices=vertices_np,
#             normal=w['normal'],
#             point=w['point'],
#             reflection=reflection_curve,
#             material=mat_name  # <--- DAS HAT GEFEHLT!
#         ))

#     return polygons

def room_walls_from_scenario(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Wählt basierend auf 'geometrie_typ' die richtige Wand-Erzeugungsfunktion.
    """
    typ = params.get("geometrie_typ")

    if typ == "becken_5x5x5_layered":
        # Keine Parameter nötig → feste Geometrie
        return define_room_5x5x5_layered()

    elif typ == "npy_custom":
        # NPY erfordert file_names + npy_path
        file_names = params.get("file_names")
        base_path = params.get("npy_path")
        if file_names is None or base_path is None:
            raise ValueError("NPY-Raum benötigt 'file_names' und 'npy_path'")
        return define_npy_room(file_names, base_path)

    elif typ == "wedge":
        # params["raum_dim"] = [L, B, H_max]
        L, B, H_max = params["raum_dim"]
        H_min = params.get("H_min", 2.0)
        return define_wedge_room(L, B, H_min, H_max)

    elif typ == "trapezoidal":
        L, B_bottom, H = params["raum_dim"]
        B_top = params.get("B_top")
        if B_top is None:
            raise ValueError("Trapezraum benötigt Parameter 'B_top'")
        return define_trapezoidal_room(L, H, B_bottom, B_top)

    elif typ == "heptagonal":
        # params["raum_dim"][2] = H
        H = params["raum_dim"][2]
        return define_heptagonal_room(H)

    else:
        raise ValueError(f"Unbekannter geometrie_typ: {typ}")

def definiere_waende(params: Dict[str, Any]) -> List[Polygon]:
    """
    Erstellt Polygon-Objekte für die Wände:
    1) Geometrie vom Szenario holen (room_walls_from_scenario)
    2) Jede Wand durch validate_wall() schicken
    3) Dann in Polygon-Objekte umwandeln
    """
    # 1. Roh-Wände vom Szenario holen (Box, Wedge, NPY, Becken...)
    walls_raw = room_walls_from_scenario(params)

    # 2. globalen Raum-Mittelpunkt bestimmen (für Normalenrichtung nach innen)
    all_vertices = np.vstack([np.array(w["vertices"]) for w in walls_raw])
    center = np.mean(all_vertices, axis=0)

    # 3. Jede Wand normieren / prüfen
    cleaned_walls = []
    for w in walls_raw:
        w_valid = validate_wall(w, center=center)
        cleaned_walls.append(w_valid)

    # 4. In Polygon-Objekte umwandeln (inkl. reflection & material)
    walls = walls_to_polygons(cleaned_walls)

    return walls

def sanity_check_polygons_for_ism(walls: List[Polygon]):
    """
    Prüft NUR die Wände, die übergeben werden.
    Diese Funktion wird nur für aktive Räume aufgerufen.
    """
    for w in walls:

        # Normale normiert?
        n_norm = np.linalg.norm(w.normal)
        if not (0.9 < n_norm < 1.1):
            raise ValueError(
                f"[WALL CHECK] Normale von '{w.name}' hat falsche Länge: {n_norm}"
            )

        # Vertices gültig?
        if w.vertices is None or w.vertices.shape[0] < 3:
            raise ValueError(f"[WALL CHECK] Wand '{w.name}' hat zu wenige Vertices.")

        if np.isnan(w.vertices).any():
            raise ValueError(
                f"[WALL CHECK] NaN in Vertices von '{w.name}'."
            )

        # Material korrekt gesetzt?
        if w.material is None or w.material == "unknown":
            print(f"[WARNUNG] Wand '{w.name}' hat unbekanntes Material.")

    print(f"[INFO] {len(walls)} Wände wurden erfolgreich geprüft.")
