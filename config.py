# config.py
from dataclasses import dataclass, field
import numpy as np
import os
from typing import List, Dict, Any, Tuple

# =============================================================================
# = DATENSTRUKTUREN
# =============================================================================

@dataclass
class Polygon:
    """Repräsentiert eine reflektierende Fläche (Wand oder Hindernis)."""
    name: str
    vertices: np.ndarray  # array der eckpunkte von der fläche Form(N,3) x,y,z N-punkte 
    normal: np.ndarray    #3d vektor gibt richtung der flächennormale an, nach innen zeigend, prüf. bed. ob Q,punkt innerhalb des raums liegt nx,ny,nz
    point: np.ndarray     # ein Punkt auf der Fläche, um abstand und skalarprodukt zur wand berechnen 
    reflection: np.ndarray     # Reflexionsfaktor R (frequenzabhängig)
    material: str = "unknown"
    color: Tuple[float,float,float] | None = None 
    # NEU: Bounding Box für schnelle Vorab-Prüfung
    # field(init=False) bedeutet: Wir müssen das nicht beim Erstellen angeben, es wird automatisch berechnet.
    min_coords: np.ndarray = field(init=False)
    max_coords: np.ndarray = field(init=False)

    def __post_init__(self):
        # Automatische Berechnung der Bounding Box direkt nach Erstellung
        # +/- 0.01 Sicherheitspuffer
        self.min_coords = np.min(self.vertices, axis=0) - 0.01
        self.max_coords = np.max(self.vertices, axis=0) + 0.01

@dataclass
class ImageSource:
    """Repräsentiert eine potentielle Spiegelquelle."""
    pos: np.ndarray       # pos der spiegelquelle x,y,z
    amp_mult: float       # Mult.Faktor fpr amp (produktaller reflexionsfaktoren entlang eines pfades)
    order: int            # Ordnung zb direktstrahl =0
    history: List[str]    # Liste der stationen um später zu schauen an welchen wänden der pfad reflektiert wurde ["S", "Wand1", "Wand2", ...]

@dataclass
class Path:
    """Repräsentiert einen validierten Pfad von Q zu Rx."""
    image_source: ImageSource   # in jedem pathobjekt steckt ein verweis auf genau eine ImageSource - ich gehöre zu dieser spiegelquelle hier 
    points: List[np.ndarray]    # Q → Reflexionspunkte → Rx , zwischen den punkten linien zeichnen und den weg im 3d plot anzeigen für die längenberechnung laufzeit t = weg/c
    color: str | None = None    # Farbe fürs plotten kann auch ohne farbe 
    segment_length: List[float]  | None = None   
    total_length: float | None = None 

#das ist eine ISM-Kerninformation: Wie lang ist jedes Segment? Wurde wirklich korrekt reflektiert? Stimmt die geometrische Länge, Wo kommen die Laufzeiten wirklich her?
#=======================================================================
# = STEUERUNG: Welche Räume sollen simuliert und geplottet werden?


# Namen der Räume, die du unten in ALLE_RAEUME definierst
ACTIVE_RAEUME = [
    #"keilraum",
    #trapezraum",
    #"heptagon",
    "finnraum",
    "becken_5x5x5_layered",
]

# =============================================================================
# = PARAMETER (Einstellungen)


SAVE_PLOT = False 
OUTPUT_FOLDER = "plots_mess_28.11"
USE_RANDOM_POSITIONS = False # Wenn True → Zufall, wenn False → feste Werte - bezogen Q und E pos.
# Falls feste Positionen 
Q_FIXED = np.array([2.9, 3.6, 2.0])    # Quelle
RX_FIXED = np.array([1.2, 1.1, 2.0])   # Empfänger

SCHALLGESCHWINDIGKEIT = 1474.7  #m/s
FREQUENZ_BAND = 48000            # künftig um passende reflexionsfaktoren R(f)  
MAX_ORDNUNG = 7           # höchste refl Ordnung die in ISM generiert 
AMP_MIN_CUTOFF = 0.005          #Pfade, deren Amplitudenfaktor kleiner als dieser Wert ist, werden verworfen -sehr schwache Echos ignorieren
FS = 192000 #sampling rate in Hz
ECHO_DENSITY_BIN_WIDTH_MS = 13 #fensterbreite echodichte in ms

USE_FREQUENCY_DEPENDENT_RIR = False   # oder False
MAX_PFAD_ORDNUNG_PLOT = 2           # Plotten der Pfade evtl. maximale Ordnung, die sichtbar sein soll
RIR_TIME_FIXED_MS = 120.0   #  feste Zeit
#RIR_TIME_LIMIT_MS_DEFAULT = 75.0
#X_LIMIT_RIR_MS = 4750.0               # Zeitachse rir 0 bis 75ms
EPSILON = 1e-6                      # kleiner wert um division durch null numerische instabilität zu vermeiden 
CMAP_NAME = 'hsv'                   # colormap
MAX_ANNOTATED_IMPULSES = 3          # Maximal so viele Impulse in der RIR werden im Plot „beschriftet“  Textlabels
MIN_Q_RX_DIST = 0.15  # 15 cm → realistisch für Beckenmessungen

USE_JITTER = False                      # Zeitliche Verschmierung von Reflexionspfaden
JITTER_AMOUNT_MS = 0.005               # 0.005 ms bei 192 kHz Sampling

# ================================================================
#  WASSERPARAMETER  (environmental settings)
#  Diese definieren das Gewässer / Becken / Meerwasser
# ================================================================

WATER_TEMPERATURE_C = 20.0      # [°C]
WATER_SALINITY_PSU = 5.0        # 0 = Süßwasser, 5 = Brackwasser, 35 = Ozean
WATER_PH = 7                  # pH im Becken dss-kie.de
WATER_DEPTH_M = 5.0             # Effektive Tiefe / Druckhöhe [m]
WATER_DENSITY = 998.0           # [kg/m^3] Süßwasser: ~998; Meerwasser: ~1025
WATER_SOUND_SPEED = 1474.7      # [m/s] Temperaturabhängig (kann dynamisch berechnet werden)


def absorption_fg_db_per_m(freq_hz, T=WATER_TEMPERATURE_C,
                           S=WATER_SALINITY_PSU,
                           depth=WATER_DEPTH_M,
                           pH=WATER_PH):
    """
    Absorptionskoeffizient α [dB/m] nach François & Garrison (1982)
    freq_hz : Frequenz in Hz
    """
    f = freq_hz / 1000.0  # Hz → kHz weil nach francois in khz gerechnet wird

    # --- Boric Acid ---
    A1 = (1.0 / 3.0) * 10**((pH - 8.0) / 0.56)
    f1 = 0.78 * (S / 35.0)**0.5 * np.exp(T / 26.0)
    A1 *= (1 + 0.023 * T) * (1 + depth / 1000)
    alpha1 = (A1 * f1 * f**2) / (f1**2 + f**2)

    # --- Magnesium Sulfate ---
    A2 = 0.52 * (1 + T / 43.0) * (S / 35.0)
    f2 = 42.0 * np.exp(T / 17.0)
    A2 *= (1 + 0.0013 * S + depth / 1000)
    alpha2 = (A2 * f2 * f**2) / (f2**2 + f**2)

    # --- Pure Water ---
    A3 = 0.00049 * np.exp(-(T / 27.0))
    alpha3 = A3 * f**2

    return alpha1 + alpha2 + alpha3  # in dB/m



#  ABSORPTION → LINEARER VERLUSTFAKTOR

def water_absorption_linear(freq_hz, distance_m):
    alpha_db = absorption_fg_db_per_m(freq_hz)
    return 10 ** (-alpha_db * distance_m / 20.0)

# ================================================================
#  FREQUENZRRASTER FÜR MATERIALMODELLE (bis 96 kHz)
# ================================================================
# Wir definieren ein logarithmisches Frequenzraster, auf dem alle Material-Reflexionskurven R(f) definiert werden. Dieses Raster wird später für alle Materialien benutzt.
FREQUENCIES = np.logspace(np.log10(50), np.log10(96000), 80)
FREQ_GRID_REFLECTION = FREQUENCIES

# FREQUENCIES hat also 80 Punkte von 50 Hz bis 96 kHz (logarithmisch verteilt).
# ================================================================
#  MATERIAL-STÜTZSTELLEN (12 Oktavpunkte)
#  An diesen Frequenzen definieren wir R(f) für jedes Material.
# ================================================================
BASE_FREQS_MATERIAL = np.array([
    50.0,
    100.0,
    200.0,
    400.0,
    800.0,
    1600.0,
    3200.0,
    6400.0,
    12000.0,
    24000.0,
    48000.0,
    96000.0
])

# --------------------------------------------------------
# 1) BETON – harte Wand, schwach absorbierend
# --------------------------------------------------------
R_BETON_STUETZ = np.array([
    0.97,  # 50 Hz
    0.97,  # 100 Hz
    0.96,  # 200 Hz
    0.95,  # 400 Hz
    0.93,  # 800 Hz
    0.92,  # 1.6 kHz
    0.90,  # 3.2 kHz
    0.88,  # 6.4 kHz
    0.85,  # 12 kHz
    0.80,  # 24 kHz
    0.75,  # 48 kHz
    0.70   # 96 kHz
])

# --------------------------------------------------------
# 2) REGUPOL-GUMMIMATTE – starker Absorber (Beckenwände)
# --------------------------------------------------------
R_REGUPOL_FX_STUETZ = np.array([
    0.90,  # 50 Hz   – tiefe f, noch relativ hohe Reflexion
    0.85,  # 100 Hz
    0.80,  # 200 Hz
    0.65,  # 400 Hz  – Beginn starke Absorption
    0.55,  # 800 Hz
    0.40,  # 1.6 kHz – maximale Dämpfzone
    0.30,  # 3.2 kHz
    0.25,  # 6.4 kHz
    0.20,  # 12 kHz
    0.15,  # 24 kHz
    0.12,  # 48 kHz
    0.10   # 96 kHz – sehr starke HF-Absorption
])

# --------------------------------------------------------
# 3) SCHLICK – weicher Untergrund / Meeresboden
# --------------------------------------------------------
R_SCHLICK_STUETZ = np.array([
    0.70,  # 50 Hz
    0.65,  # 100 Hz
    0.55,  # 200 Hz
    0.40,  # 400 Hz
    0.30,  # 800 Hz
    0.20,  # 1.6 kHz
    0.15,  # 3.2 kHz
    0.12,  # 6.4 kHz
    0.10,  # 12 kHz
    0.08,  # 24 kHz
    0.06,  # 48 kHz
    0.05   # 96 kHz
])

# --------------------------------------------------------
# 4) GLAS / ACRYL – hart und glatt, stark reflektierend
# --------------------------------------------------------
R_GLAS_STUETZ = np.array([
    0.99,  # 50 Hz
    0.98,  # 100 Hz
    0.98,  # 200 Hz
    0.97,  # 400 Hz
    0.96,  # 800 Hz
    0.95,  # 1.6 kHz
    0.93,  # 3.2 kHz
    0.92,  # 6.4 kHz
    0.90,  # 12 kHz
    0.88,  # 24 kHz
    0.86,  # 48 kHz
    0.85   # 96 kHz
])

# --------------------------------------------------------
# 5) WASSEROBERFLÄCHE – nahezu perfekter Spiegel
# --------------------------------------------------------
R_WATER_SURFACE_STUETZ = np.array([
    0.995,  # 50 Hz
    0.995,  # 100 Hz
    0.995,  # 200 Hz
    0.992,  # 400 Hz
    0.990,  # 800 Hz
    0.990,  # 1.6 kHz
    0.988,  # 3.2 kHz
    0.985,  # 6.4 kHz
    0.982,  # 12 kHz
    0.980,  # 24 kHz
    0.980,  # 48 kHz
    0.980   # 96 kHz
])

# ================================================================
#  HILFSFUNKTION: Interpolation auf FREQUENCIES
# ================================================================
def interpolate_reflection_to_grid(stuetzwerte: np.ndarray) -> np.ndarray:
    """
    Interpoliert eine R(f)-Kurve, die auf BASE_FREQS_MATERIAL definiert ist,
    auf das globale FREQUENCIES-Raster.
    
    stuetzwerte: Array der Länge len(BASE_FREQS_MATERIAL) mit Reflexionsfaktoren.
    Rückgabe   : Array der Länge len(FREQUENCIES) mit interpolierten Werten.
    """
    return np.interp(FREQUENCIES, BASE_FREQS_MATERIAL, stuetzwerte)

# ================================================================
#  MATERIALIEN: fertige freq-abhängige Reflexionskurven R(f)
# ================================================================
# Hier erzeugen wir für jedes Material eine freq-abhängige Kurve
# R(f) über das Raster FREQUENCIES. Diese Arrays werden später
# in den Polygon-Objekten gespeichert.

MATERIAL_REFLECTIONS: Dict[str, np.ndarray] = {
    "beton":          interpolate_reflection_to_grid(R_BETON_STUETZ),
    "regupol_fx":     interpolate_reflection_to_grid(R_REGUPOL_FX_STUETZ),
    "schlick":        interpolate_reflection_to_grid(R_SCHLICK_STUETZ),
    "glas":           interpolate_reflection_to_grid(R_GLAS_STUETZ),
    "water_surface":  interpolate_reflection_to_grid(R_WATER_SURFACE_STUETZ),
}
# ====================================================================
# MATERIAL → FARBE für Plot & Legende
# ====================================================================
MATERIAL_COLORS = {
    "beton": "gray",
    "regupol_fx": "green",
    "schlick": "saddlebrown",
    "glas": "cyan",
    "water_surface": "blue"
}

# ====================================================================
#  MATERIAL-ZUORDNUNG FÜR DAS 5x5x5-BECKEN
# ====================================================================
BECKEN_MATERIAL = {
    "unten": "regupol_fx",          # Boden 
    "seiten_0_3": "regupol_fx",     # Seitenwände von 0 bis 3 m
    "seiten_3_5": "beton",          # Seitenwände von 3 bis 5 m
    "decke": "water_surface"        # Decke bei 5m
}

# ====================================================================
#  SPEZIELLE MATERIAL-ZUORDNUNG FÜR NPY-RÄUME
# ====================================================================
NPY_MATERIAL_RULES = {
    "boden_material": "schlick",      # ganze Bodenfläche
    "wand_materials": ["beton", "regupol_fx"],   # 50/50 zufällig oder alternierend
    "decke": "water_surface"        # Decke
}

# =============================================================================
# = WAND-VALIDIERUNG UND GEOMETRIE-DISPATCHER
# =============================================================================

def validate_wall(w: Dict[str, Any], center: np.ndarray) -> Dict[str, Any]:
    """
    Prüft und normiert eine einzelne Wand:
    - Normalenrichtung nach innen drehen
    - Punkt auf Ebene bestimmen
    - Material & Reflexionskurve setzen
    """
    vertices = np.array(w["vertices"], dtype=float)
    normal = np.array(w["normal"], dtype=float)

    # 1) Normale normieren
    n_norm = np.linalg.norm(normal)
    if n_norm > 1e-12:
        normal = normal / n_norm

    # 2) Punkt auf der Ebene (erster Vertex)
    point = vertices[0]

    # 3) Prüfen, ob die Normale zum Raummittelpunkt zeigt
    vec_center = center - point
    if np.dot(vec_center, normal) < 0:
        normal = -normal

    # 4) Material zuweisen (fallback = beton)
    material = w.get("material", "beton")
    if material not in MATERIAL_REFLECTIONS:
        material = "beton"

    # 5) Reflection-Kurve & Farbe
    reflection = MATERIAL_REFLECTIONS[material]
    color = MATERIAL_COLORS.get(material, "gray")

    # 6) Rückgabe in standardisierter Form
    return {
        "name": w["name"],
        "vertices": vertices,
        "normal": normal,
        "point": point,
        "material": material,
        "reflection": reflection,
        "color": color
    }


def walls_to_polygons(walls_raw: List[Dict[str, Any]]) -> List[Polygon]:
    """
    Wand-Dictionaries → Polygon-Objekte
    """
    polys = []
    for w in walls_raw:
        p = Polygon(
            name=w["name"],
            vertices=w["vertices"],
            normal=w["normal"],
            point=w["point"],
            reflection=w["reflection"],
            material=w["material"],
            color=w["color"]
        )
        polys.append(p)
    return polys


# # =============================================================================
# # = GEOMETRIE-DISPATCHER FÜR ALLE SZENARIEN
# # =============================================================================

# def room_walls_from_scenario(params: Dict[str, Any]) -> List[Dict[str, Any]]:
#     """
#     Wählt basierend auf 'geometrie_typ' den richtigen Wand-Generator aus.
#     """
#     typ = params.get("geometrie_typ")

#     if typ == "becken_5x5x5_layered":
#         return define_becken_5x5x5(params)

#     elif typ == "npy_custom":
#         return define_npy_room(params)

#     elif typ == "wedge":
#         return define_wedge_room(params)

#     elif typ == "trapezoidal":
#         return define_trapezraum(params)

#     elif typ == "heptagonal":
#         return define_heptagon(params)

#     else:
#         raise ValueError(f"Unbekannter geometrie_typ: {typ}")


# Optional: kleine Hilfsfunktion, falls du später mal dynamisch abfragen willst
def get_material_reflection(material_name: str) -> np.ndarray:
    """
    Gibt die freq-abhängige Reflexionskurve R(f) für ein Material zurück.
    material_name: z.B. 'beton', 'regupol_fx', 'schlick', 'glas', 'water_surface'
    """
    if material_name not in MATERIAL_REFLECTIONS:
        raise ValueError(f"Unbekanntes Material: {material_name}")
    return MATERIAL_REFLECTIONS[material_name]
                                                                                    ############

# =============================================================================
# = ALLE DEFINIERTEN RÄUME / SZENARIEN
# =============================================================================

ALLE_RAEUME = {
    "keilraum": {
        'name': "1. Geneigter Keilraum (H: 2m bis 4m)", 
        'geometrie_typ': 'wedge',
        'raum_dim': np.array([10.0, 5.0, 4.0]),
        'min_dist_wand': 0.7, 
        'rir_time_limit_ms': 250.0
    },

    "trapezraum": {
        'name': "2. Trapezraum (B: 6m unten, 4m oben)",
        'geometrie_typ': 'trapezoidal',
        'raum_dim': np.array([10.0, 6.0, 3.0]),
        'B_top': 4.0,
        'min_dist_wand': 0.5,
        'rir_time_limit_ms': 250.0
    },

    "heptagon": {
        'name': "3. Heptagonal-Raum (7-Eck Grundriss, H=3m)",
        'geometrie_typ': 'heptagonal',
        'raum_dim': np.array([12.0, 6.0, 3.0]),
        'min_dist_wand': 0.8, 
        'rir_time_limit_ms': 250.0
    },
    

    "finnraum": {
        'name': "4. Finnraum (NPY Geometrie)", 
        'geometrie_typ': 'npy_custom', 
        'raum_dim': np.array([800, 600, 50]),
        'file_names': ['Room.npy'],
        'tx_file': 'TxPosition.npy',
        'rx_file': 'RxPosition.npy',
                # 'npy_path': r"C:\bachelorarbeit-hasan-yesil\Ocean_Reverberation\finnraum", # Original für Windows
        'npy_path': "Ocean_Reverberation/finnraum",
        'min_dist_wand': 0.2,
        'rir_time_limit_ms': 5500.0
    },

    "becken_5x5x5_layered" : {
    'name': "5x5x5 Becken ",
    'geometrie_typ': 'becken_5x5x5_layered',
    'raum_dim': np.array([5.0, 5.0, 5.0]),
    'min_dist_wand': 0.5,
    'rir_time_limit_ms': 40.0,
    'x_unit': 'ms'
}

}

# =============================================================================
# = Finale Liste der aktiven Szenarien
# =============================================================================

szenarien = [ALLE_RAEUME[name] for name in ACTIVE_RAEUME]



def octave_band_mean(freqs, values):
    """Berechne Oktavband-Mittelwerte |R| energetisch."""
    centers = np.array([
        63,125,250,500,1000,2000,4000,8000,
        16000,32000,64000
    ])
    means = []
    for fc in centers:
        f_low = fc / np.sqrt(2)
        f_high = fc * np.sqrt(2)
        mask = (freqs >= f_low) & (freqs <= f_high)
        if np.any(mask):
            means.append(np.sqrt(np.mean(values[mask]**2)))
        else:
            means.append(np.nan)
    return centers, np.array(means)


# ================================================
#  MATERIAL-PLOT (R(f)-Kurven)
# ================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Hole Materialnamen und Farben
    materials = list(MATERIAL_REFLECTIONS.keys())

    plt.figure(figsize=(12, 6))

    for mat in materials:
        R = MATERIAL_REFLECTIONS[mat]            # Interpolierte R(f)
        color = MATERIAL_COLORS.get(mat, None)   # Farbe falls definiert
        plt.semilogx(FREQUENCIES, R, label=mat, linewidth=2, color=color)

    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("Reflexionskoeffizient |R(f)|")
    plt.title("Material-Reflexionskurven aus config.py")
    plt.legend()
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(12,6))

    for mat in MATERIAL_REFLECTIONS:
        centers, R_oct = octave_band_mean(FREQUENCIES, MATERIAL_REFLECTIONS[mat])
        plt.semilogx(centers, R_oct, "o-", label=mat, linewidth=2)

    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xlabel("Oktavband-Mittenfrequenz [Hz]")
    plt.ylabel("Bandmittelwert |R|")
    plt.title("Oktavband-Reflexionskoeffizienten")
    plt.legend()
    plt.tight_layout()
    plt.show()
