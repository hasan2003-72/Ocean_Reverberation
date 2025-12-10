# plotting.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from typing import List, Dict, Any
from scipy.signal import butter, lfilter




from config import (
    MAX_PFAD_ORDNUNG_PLOT, Polygon, RIR_TIME_FIXED_MS , SCHALLGESCHWINDIGKEIT, ECHO_DENSITY_BIN_WIDTH_MS,       
    MAX_ANNOTATED_IMPULSES, Path, MAX_ORDNUNG, FS,
)
from ism_core import berechne_t60
from geometry import definiere_waende

# =============================================================================
# 1. FARB-DEFINITIONEN
# =============================================================================
MATERIAL_COLORS = {
    "beton":          (0.6, 0.6, 0.6),       # Grau
    "regupol_fx":     (0.2, 0.8, 0.2),       # Grün
    "glas":           (0.0, 1.0, 1.0),       # Cyan
    "water_surface":  (0.0, 0.0, 1.0),       # Blau
    "schlick":        (0.55, 0.27, 0.07),    # Braun
    "boden_material": (0.55, 0.27, 0.07),    # Braun (NPY-Boden)
    "unknown":        (1.0, 0.0, 0.0)        # Rot (Fehler/Unbekannt)
}

def get_material_color(mat_name: str):
    """
    Hilfsfunktion: Bestimmt die Farbe anhand des Materialnamens.
    Ist tolerant gegenüber Groß-/Kleinschreibung und Teil-Strings.
    """
    if not mat_name: 
        return MATERIAL_COLORS["unknown"]
    
    m = mat_name.lower()
    
    # 1. Exakter Match
    if m in MATERIAL_COLORS: 
        return MATERIAL_COLORS[m]
    
    # 2. Intelligente Suche (Teilwörter)
    if "beton" in m: return MATERIAL_COLORS["beton"]
    if "regupol" in m: return MATERIAL_COLORS["regupol_fx"]
    if "water" in m or "wasser" in m: return MATERIAL_COLORS["water_surface"]
    if "schlick" in m or "boden" in m: return MATERIAL_COLORS["schlick"]
    if "glas" in m or "glass" in m: return MATERIAL_COLORS["glas"]
    
    return MATERIAL_COLORS["unknown"]

def assign_polygon_color(poly: Polygon):
    """Schreibt die Farbe direkt in das Polygon-Objekt."""
    poly.color = get_material_color(poly.material)


# =============================================================================
# 2. 3D-GEOMETRIE PLOT
# =============================================================================
def plotte_geometrie_3d(ax: plt.Axes, valide_pfade: List[Path], params: Dict[str, Any], color_map_k: Dict[int, Any]):
    """
    Zeichnet den 3D-Raum inkl. zwei Legenden (Materialien & Akustik).
    """
    
    # --- A) Setup ---
    ax.set_title(f"Raum: {params['name']}", fontsize=12, fontweight='bold')
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")

    # --- B) Wände zeichnen & Materialien für Legende sammeln ---
    waende = definiere_waende(params)
    all_coords = []
    verwendete_materialien = set() # Set verhindert Duplikate in der Legende

    for wand in waende:
        assign_polygon_color(wand) # Farbe zuweisen
        
        # Namen für Legende merken (z.B. "regupol_fx")
        mat_name = wand.material if wand.material else "Unbekannt"
        verwendete_materialien.add(mat_name)
        
        vertices = wand.vertices
        all_coords.extend(vertices)

        # Wand zeichnen: alpha=0.3 sorgt für Transparenz
        poly = Poly3DCollection([vertices], alpha=0.3, facecolor=wand.color, 
                                edgecolor=(0.3,0.3,0.3,0.5), linewidth=0.5)
        ax.add_collection3d(poly)

    # --- C) Achsen skalieren (Equal Aspect Ratio) ---
    if all_coords:
        all_coords = np.array(all_coords)
        min_c, max_c = np.min(all_coords, axis=0), np.max(all_coords, axis=0)
        center = (min_c + max_c) / 2
        radius = np.max(max_c - min_c) / 2 * 1.1
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_box_aspect([1, 1, 1])

    # --- D) Quelle & Empfänger ---
    ax.scatter(*params['quelle_pos'], c='red', s=80, marker='*', edgecolors='black', zorder=200)
    ax.scatter(*params['empfaenger_pos'], c='green', s=80, marker='^', edgecolors='black', zorder=200)

    # --- E) Pfade zeichnen (ohne Textlabels) ---
    pfade_plot = [p for p in valide_pfade if p.image_source.order <= MAX_PFAD_ORDNUNG_PLOT]
    # Sortieren: Hohe Ordnung zuerst (hinten), Direktschall zuletzt (vorne/oben)
    pfade_plot.sort(key=lambda p: p.image_source.order, reverse=True)

    for p in pfade_plot:
        pts = np.array(p.points)
        k = p.image_source.order
        
        c = p.color if p.color else 'black'
        lw = 2.5 if k == 0 else 1.5  # Direktschall dicker
        alpha = 1.0 if k == 0 else 0.7
        
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color=c, linewidth=lw, alpha=alpha)


    # ==========================================
    # LEGENDE 1: MATERIALIEN (Links Oben)
    # ==========================================
    mat_handles = []
    for mat_name in sorted(list(verwendete_materialien)):
        c = get_material_color(mat_name)
        # Erstelle ein farbiges Quadrat für die Legende
        patch = mpatches.Patch(color=c, label=mat_name, alpha=0.6)
        mat_handles.append(patch)
    
    if mat_handles:
        legend_mat = ax.legend(handles=mat_handles, title="Wand-Materialien", 
                               loc='upper left', fontsize=8, framealpha=0.9,
                               bbox_to_anchor=(0, 1))
        ax.add_artist(legend_mat) # Wichtig! Fügt Legende hinzu, ohne den Slot für die nächste zu blockieren

    # ==========================================
    # LEGENDE 2: AKUSTIK (Rechts Oben)
    # ==========================================
    path_handles = []
    # Symbole für Q und Rx
    path_handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='Quelle'))
    path_handles.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Empfänger'))
    # Linie für Direktschall
    path_handles.append(Line2D([0], [0], color='darkgreen', lw=2.5, label='Direkt (K=0)'))
    
    # Linien für Reflexionen (nur die ersten paar anzeigen, damit Legende nicht explodiert)
    for k in range(1, min(4, MAX_PFAD_ORDNUNG_PLOT + 1)):
        c_k = color_map_k.get(k, 'gray')
        path_handles.append(Line2D([0], [0], color=c_k, lw=1.5, label=f'Refl. K={k}'))
    
    if MAX_PFAD_ORDNUNG_PLOT > 3:
         path_handles.append(Line2D([0], [0], color='gray', lw=1.0, linestyle='--', label='Höhere Ordn.'))

    ax.legend(handles=path_handles, title="Pfad-Legende", 
              loc='upper right', fontsize=8, framealpha=0.9,
              bbox_to_anchor=(1, 1))

    # Standard-Ansicht
    ax.view_init(elev=30, azim=-60)

# =

# =============================================================================
# 3. RIR PLOT (Dynamisch: ms oder s)
# =============================================================================
# def plotte_rir(ax: plt.Axes, rir_daten: List[Dict[str, Any]], params: Dict[str, Any], color_map_k: Dict[int, Any]):
#     """
#     Plottet die RIR. Entscheidet anhand von params['x_unit'] ob ms oder s.
#     """
#     # Standard ist 's' (Sekunden), außer es steht 'ms' im Szenario
#     unit = params.get('x_unit', 's') 
    
#     if unit == 'ms':
#         faktor = 1.0       # Daten sind schon in ms
#         label = "Zeit [ms]"
#     else:
#         faktor = 1000.0    # Umrechnung ms -> s
#         label = "Zeit [s]"

#     ax.set_title("Impulsantwort & Energieabfall", fontsize=11, fontweight='bold')
#     ax.set_xlabel(label) 
#     ax.set_ylabel("Pegel [dB]")
#     ax.grid(True, ls=':', alpha=0.6)

#     # Limits setzen
#     limit_ms = RIR_TIME_FIXED_MS
#     ax.set_xlim(0, limit_ms / faktor)
#     ax.set_ylim(-90, 5)

#     if not rir_daten:
#         ax.text((limit_ms/faktor)/2, -40, "Keine Pfade", ha='center')
#         return

#     # 1. Impulse zeichnen
#     rir_sorted = sorted(rir_daten, key=lambda x: x['delay_ms'])
    
#     for imp in rir_sorted:
#         k = imp['ord']
#         t_val = imp['delay_ms'] / faktor # <--- Hier wird geteilt (oder nicht)
#         amp = imp['amp_db']
        
#         if k == 0: 
#             c = 'darkgreen'; lw = 2.5; z=10
#         else:      
#             c = color_map_k.get(k, 'gray')
#             lw = 1.2; z=5
            
#         ax.vlines(t_val, -120, amp, color=c, linewidth=lw, zorder=z)

#     # 2. T60 / EDC Kurve
#     zeit_ms, edc, t60, fit_x, fit_y, _, _, _, _ = berechne_t60(rir_daten)
#     t60_label = "T60: N/A"
    
#     if zeit_ms is not None:
#         zeit_plot = zeit_ms / faktor
        
#         ax2 = ax.twinx()
#         ax2.set_yticks([]) 
#         ax2.set_ylim(-90, 5)
#         ax2.plot(zeit_plot, edc, color='darkorange', alpha=0.9, lw=2, label='EDC')
        
#         if t60 is not None:
#             ax2.plot(fit_x / faktor, fit_y, 'r--', lw=1.5)
#             t60_label = f"T60 = {t60/1000:.2f} s" # T60 bleibt im Text immer Sekunde (Standard)

#     # 3. Legende (Alle Farben)
#     handles = []
#     handles.append(Line2D([0], [0], color='darkgreen', lw=2.5, label='Direkt (K=0)'))
    
#     # HIER WAR DIE ÄNDERUNG:
#     # Wir iterieren jetzt direkt über die Schlüssel der Farb-Map (= alle berechneten Ordnungen)
#     vorhandene_ordnungen = sorted(color_map_k.keys())
    
#     for k in vorhandene_ordnungen:
#         col = color_map_k[k]
#         handles.append(Line2D([0], [0], color=col, lw=1.5, label=f'Ordnung K={k}'))
    
#     if zeit_ms is not None:
#         handles.append(Line2D([0], [0], color='darkorange', lw=2, label='Energie (EDC)'))
#         handles.append(Line2D([0], [0], color='red', linestyle='--', lw=1.5, label=t60_label))

#     ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)



def plotte_rir(ax: plt.Axes, rir_daten: List[Dict[str, Any]], params: Dict[str, Any], color_map_k: Dict[int, Any]):
    """
    Plottet die RIR – abhängig vom Modus:
    - Dirac-Modus (klassisch)
    - FIR-Modus (freq.-abhängig)
    """

    USE_FIR = params.get("USE_FREQUENCY_DEPENDENT_RIR", False)
    rir_signal = params.get("rir", None)
    fs = params.get("fs", 192000)

    unit = params.get('x_unit', 's')
    faktor = 1.0 if unit == 'ms' else 1000.0
    label = "Zeit [ms]" if unit == "ms" else "Zeit [s]"
    
    ax.set_title("Impulsantwort & Energieabfall", fontsize=11, fontweight='bold')
    ax.set_xlabel(label)
    ax.set_ylabel("Pegel [dB]")
    ax.grid(True, ls=':', alpha=0.6)

    limit_ms = RIR_TIME_FIXED_MS
    ax.set_xlim(0, limit_ms / faktor)
    ax.set_ylim(-90, 5)

    # ===============================================================
    # 🟩 1) FIR-MODUS  → echtes Signal, kein Dirac-Plot!
    # ===============================================================
    if USE_FIR and rir_signal is not None:
        N = len(rir_signal)
        t_ms = (np.arange(N) / fs) * 1000.0 / faktor
        rir_db = 20*np.log10(np.abs(rir_signal) + 1e-12)

        ax.plot(t_ms, rir_db, color='blue', lw=1.0, alpha=0.8, label="FIR-RIR")
        ax.legend(loc='upper right')
        return

    # ===============================================================
    # 🟥 2) KLASSISCHER DIRAC-MODUS
    # ===============================================================
    if not rir_daten:
        ax.text((limit_ms/faktor)/2, -40, "Keine Pfade", ha='center')
        return

    # Impulse nach Zeit sortieren
    rir_sorted = sorted(rir_daten, key=lambda x: x['delay_ms'])

    for imp in rir_sorted:
        k = imp['ord']
        t_val = imp['delay_ms'] / faktor
        amp = imp['amp_db']

        if k == 0:
            c = 'darkgreen'; lw = 2.5; z = 10
        else:
            c = color_map_k.get(k, 'gray')
            lw = 1.2; z = 5

        ax.vlines(t_val, -120, amp, color=c, linewidth=lw, zorder=z)

    # ------------------------------------------------------------
    # EDC & T60 (klassisch)
    # ------------------------------------------------------------
    zeit_ms, edc, t60, fit_x, fit_y, _, _, _ = berechne_t60(rir_daten)
    if zeit_ms is not None:
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.plot(zeit_ms / faktor, edc, color='darkorange', lw=2, label='EDC')
        if t60 is not None:
            ax2.plot(fit_x / faktor, fit_y, 'r--', lw=1.5)




def plotte_raum_rir_und_echodichte(valide_pfade, rir_daten, rir_array, echo_t, echo_density, params, color_map_k):
    """
    Erstellt das finale Fenster inkl. Frequenzgang im FIR-Modus.
    # """
    # fig = plt.figure(figsize=(18, 8))

    # # 2 Zeilen, 3 Spalten → 3 Plots oben, 3 unten
    # gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1.0, 1.0])

    # ax_raum = fig.add_subplot(gs[:, 0], projection='3d')
    # ax_rir  = fig.add_subplot(gs[0, 1])
    # ax_echo = fig.add_subplot(gs[1, 1])
    # ax_freq = fig.add_subplot(gs[0, 2])     # <-- NEU

    # # --------- 1. GEOMETRIE ---------
    # plotte_geometrie_3d(ax_raum, valide_pfade, params, color_map_k)

    # # --------- 2. RIR ---------
    # plotte_rir(ax_rir, rir_daten, params, color_map_k)

    # Layout breiter machen (3 Spalten)
    fig = plt.figure(figsize=(18, 8)) 
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1]) 

    # Spalte 1: Der 3D Raum (groß)
    ax_raum = fig.add_subplot(gs[:, 0], projection='3d')
    
    # Spalte 2: Oben Linear (Waveform), Unten dB (Log)
    ax_wave = fig.add_subplot(gs[0, 1])
    ax_rir_db = fig.add_subplot(gs[1, 1])
    
    # 1. Raum
    plotte_geometrie_3d(ax_raum, valide_pfade, params, color_map_k)
    
    # 3. dB Plot (Logarithmisch)
    plotte_rir(ax_rir_db, rir_daten, params, color_map_k)

    plt.tight_layout()
    return fig, (ax_raum, ax_wave, ax_rir_db)


def plot_edc_with_t60(zeit_ms, edc_db, t60_ms, x_fit, fit_y):
    """Visualisiert EDC und T60 Extrapolation"""
    plt.figure(figsize=(10, 6))
    
    # Plot der EDC-Kurve
    plt.plot(zeit_ms, edc_db, label="Energy Decay Curve (EDC)", color="blue", linewidth=2)
    
    # Plot der T60 Extrapolation
    plt.plot(x_fit, fit_y, label=f"T60 Extrapolation", color="red", linestyle="--")

    # Markiere den Schnittpunkt (T60)
    plt.axvline(x=t60_ms, color="green", linestyle=":", label=f"T60 = {t60_ms:.2f} ms")
    
    # Achsen und Titel
    plt.xlabel("Zeit [ms]")
    plt.ylabel("Pegel [dB]")
    plt.title("EDC und T60 Extrapolation")
    plt.legend()

    plt.grid(True)
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter



#uboot sonar ähnlich einfach sinus ping 
# def plotte_waveform_linear(ax: plt.Axes, rir_array: np.ndarray, fs: float, params: Dict[str, Any]):
#     """
#     Plottet die Wellenform.
#     TRICK: Faltet die RIR mit einem Bandpass-Signal (Pinger), 
#     damit es aussieht wie eine echte hydroakustische Messung.
#     """
#     # 1. Ein künstliches "Pinger-Signal" erzeugen (z.B. kurzer 2 kHz Sinus-Burst)
#     # Das imitiert deinen Lautsprecher/Hydrofon.
#     f_pinger = 2000.0  # Frequenz des Pingers (z.B. 2 kHz)
#     duration_pinger = 0.002 # 2 ms lang
#     t_pinger = np.arange(int(duration_pinger * fs)) / fs
#     # Ein Sinus, der sanft ein- und ausschwingt (Hanning-Fenster)
#     pinger_signal = np.sin(2 * np.pi * f_pinger * t_pinger) * np.hanning(len(t_pinger))

#     # 2. Faltung: RIR * Pinger = Das was das Mikrofon hört
#     # Modus 'full' macht es länger, wir schneiden es danach ab
#     measured_signal = np.convolve(rir_array, pinger_signal, mode='full')
    
#     # Länge wieder anpassen (auf original RIR Länge beschränken für den Plot)
#     measured_signal = measured_signal[:len(rir_array)]

#     # ---------------------------------------------------------
#     # PLOTTEN
#     # ---------------------------------------------------------
#     t = np.arange(len(measured_signal)) / fs
    
#     unit = params.get('x_unit', 's')
#     if unit == 'ms':
#         t = t * 1000.0
#         x_label = "Zeit [ms]"
#     else:
#         x_label = "Zeit [s]"

#     # Plotten (schwarz, dünn)
#     ax.plot(t, measured_signal, color='black', linewidth=0.6)
    
#     ax.set_title("Simuliertes Mikrofonsignal (RIR * Pinger)", fontsize=11, fontweight='bold')
#     ax.set_xlabel(x_label)
#     ax.set_ylabel("Amplitude (a.u.)")
#     ax.grid(True, alpha=0.5)
    
#     limit_ms = params.get('rir_time_limit_ms', 100.0)
#     if unit == 'ms':
#         ax.set_xlim(0, limit_ms)
#     else:
#         ax.set_xlim(0, limit_ms / 1000.0)

#     # Y-Limits symmetrisch
#     max_val = np.max(np.abs(measured_signal))
#     if max_val > 0:
#         ax.set_ylim(-max_val*1.1, max_val*1.1)