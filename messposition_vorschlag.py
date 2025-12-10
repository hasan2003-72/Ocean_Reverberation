# visualize_positions_standalone.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

# =============================================================================
# 1. GEOMETRIE DEFINITION (5x5x5 Becken)
#    Hardcoded, damit keine externen Files nötig sind.
# =============================================================================
def get_tank_walls():
    """Erzeugt die Wände des 5x5x5 Beckens direkt als Liste mit Farben."""
    L, B, H = 5.0, 5.0, 5.0
    H_split = 3.0 # Übergang Regupol/Beton

    # Farben
    C_REGUPOL = (0.2, 0.8, 0.2, 0.2) # Grün, transparent
    C_BETON   = (0.6, 0.6, 0.6, 0.2) # Grau, transparent
    C_WASSER  = (0.0, 0.0, 1.0, 0.2) # Blau, transparent

    walls = []

    # Eckpunkte Boden
    p0, p1 = np.array([0,0,0]), np.array([L,0,0])
    p2, p3 = np.array([L,B,0]), np.array([0,B,0])
    
    # Eckpunkte Split (3m)
    p0m, p1m = np.array([0,0,H_split]), np.array([L,0,H_split])
    p2m, p3m = np.array([L,B,H_split]), np.array([0,B,H_split])

    # Eckpunkte Oben (5m)
    p0t, p1t = np.array([0,0,H]), np.array([L,0,H])
    p2t, p3t = np.array([L,B,H]), np.array([0,B,H])

    # 1. BODEN (Regupol)
    walls.append({"verts": [p0, p1, p2, p3], "color": C_REGUPOL, "name": "Boden (Regupol)"})

    # 2. SEITENWÄNDE UNTEN (0-3m, Regupol)
    walls.append({"verts": [p0, p1, p1m, p0m], "color": C_REGUPOL, "name": "Wand U (Regupol)"})
    walls.append({"verts": [p1, p2, p2m, p1m], "color": C_REGUPOL, "name": "Wand R (Regupol)"})
    walls.append({"verts": [p2, p3, p3m, p2m], "color": C_REGUPOL, "name": "Wand O (Regupol)"})
    walls.append({"verts": [p3, p0, p0m, p3m], "color": C_REGUPOL, "name": "Wand L (Regupol)"})

    # 3. SEITENWÄNDE OBEN (3-5m, Beton)
    walls.append({"verts": [p0m, p1m, p1t, p0t], "color": C_BETON, "name": "Wand U (Beton)"})
    walls.append({"verts": [p1m, p2m, p2t, p1t], "color": C_BETON, "name": "Wand R (Beton)"})
    walls.append({"verts": [p2m, p3m, p3t, p2t], "color": C_BETON, "name": "Wand O (Beton)"})
    walls.append({"verts": [p3m, p0m, p0t, p3t], "color": C_BETON, "name": "Wand L (Beton)"})

    # 4. DECKE (Wasser)
    walls.append({"verts": [p0t, p1t, p2t, p3t], "color": C_WASSER, "name": "Wasseroberfläche"})

    return walls

# =============================================================================
# 2. SZENARIEN (Positionen + Erklärungstext)
# =============================================================================
SCENARIOS = [
    {
        "id": "1_Diagonale",
        "title": "Szenario 1: Die Raum-Diagonale",
        "Q":  [1.0, 1.0, 1.0],
        "Rx": [4.0, 4.0, 4.0],
        "text": (
            "INTERESSANT WEIL: Maximale Distanz im Raum.\n"
            "- Bestimmung der globalen Nachhallzeit (T60).\n"
            "- Wenig Einfluss einzelner Wände (Moden).\n"
            "- Direktschall und Reflexionen sind zeitlich maximal getrennt.\n"
            "- 'Referenzmessung' für den durchschnittlichen Raumklang."
        )
    },
    {
        "id": "2_Oberflaeche",
        "title": "Szenario 2: Der Oberflächen-Effekt (Lloyd's Mirror)",
        "Q":  [1.5, 2.5, 4.5],
        "Rx": [3.5, 2.5, 4.5],
        "text": (
            "INTERESSANT WEIL: Extreme Nähe zur Wasseroberfläche (Z=4.5m).\n"
            "- Die Wasseroberfläche ist ein fast perfekter Reflektor.\n"
            "- Direktschall und das Oberflächen-Echo kommen fast zeitgleich an.\n"
            "- Dies führt zu starken Kammfilter-Effekten (Auslöschungen).\n"
            "- Kritisch für flache Unterwasser-Kommunikation."
        )
    },
    {
        "id": "3_Materialgrenze",
        "title": "Szenario 3: Der Vertikale Material-Check",
        "Q":  [2.5, 2.5, 0.5],
        "Rx": [2.5, 2.5, 4.5],
        "text": (
            "INTERESSANT WEIL: Schall muss durch die Materialgrenze (Z=3m).\n"
            "- Quelle steht tief im 'Sumpf' (Regupol = Absorbierend).\n"
            "- Empfänger steht hoch im 'Hall' (Beton = Hart).\n"
            "- Testet, ob die Simulation den Energieverlust durch den Boden\n"
            "  und den Übergang zum harten Bereich korrekt abbildet."
        )
    },
    {
        "id": "4_Wandnahe",
        "title": "Szenario 4: Das Flatterecho (Flutter Echo)",
        "Q":  [0.5, 2.5, 3.5],
        "Rx": [4.5, 2.5, 3.5],
        "text": (
            "INTERESSANT WEIL: Parallel zu den harten Betonwänden.\n"
            "- Der Schall wird zwischen den Wänden (Y-Achse) hin- und hergeworfen.\n"
            "- Erzeugt eine hohe Echodichte in einer Ebene.\n"
            "- Typisches Problem in quaderförmigen Räumen ('Metallischer Klang')."
        )
    }
]

# =============================================================================
# 3. MAIN LOOP & PLOTTING
# =============================================================================
def main():
    output_dir = "mess_positionen_bilder"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generiere Bilder in Ordner: {output_dir}")

    walls = get_tank_walls()

    for s in SCENARIOS:
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Titel
        ax.set_title(s["title"], fontsize=14, fontweight='bold')
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")

        # Wände zeichnen
        all_verts = []
        for w in walls:
            v = np.array(w["verts"])
            all_verts.extend(v)
            poly = Poly3DCollection([v], alpha=0.2, facecolor=w["color"], edgecolor='grey', linewidth=0.5)
            ax.add_collection3d(poly)
        
        # Skalierung fixieren (Box Aspect)
        all_verts = np.array(all_verts)
        ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_zlim(0, 5)
        ax.set_box_aspect([1,1,1])

        # Positionen zeichnen
        q = np.array(s["Q"])
        rx = np.array(s["Rx"])
        
        # Rote Linie (Sichtlinie)
        ax.plot([q[0], rx[0]], [q[1], rx[1]], [q[2], rx[2]], 'r--', lw=2, label='Sichtlinie')
        
        # Punkte
        ax.scatter(*q, c='red', s=150, marker='*', edgecolors='black', zorder=100, label=f'Quelle {q}')
        ax.scatter(*rx, c='green', s=150, marker='^', edgecolors='black', zorder=100, label=f'Empfänger {rx}')

        # Legende im Plot
        ax.legend(loc='upper right')
        ax.view_init(elev=20, azim=-50)

        # TEXTBOX UNTEN
        explanation = s["text"]
        fig.text(0.5, 0.02, explanation, 
                 ha='center', va='bottom', fontsize=11, family='monospace',
                 bbox=dict(facecolor='#f8f9fa', alpha=1.0, edgecolor='#333', boxstyle='round,pad=0.8'))

        # Speichern
        filename = os.path.join(output_dir, f"{s['id']}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f" -> Bild erstellt: {filename}")
        
        plt.close(fig)

    print("Fertig.")

if __name__ == "__main__":
    main()
