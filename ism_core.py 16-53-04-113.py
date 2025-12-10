# ism_core.py
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from config import (
    ImageSource, Path, Polygon, 
    MAX_ORDNUNG, AMP_MIN_CUTOFF, SCHALLGESCHWINDIGKEIT,
    ECHO_DENSITY_BIN_WIDTH_MS, RIR_TIME_FIXED_MS, EPSILON,
    USE_FREQUENCY_DEPENDENT_RIR, USE_JITTER, JITTER_AMOUNT_MS,
    water_absorption_linear, FREQ_GRID_REFLECTION
)

from geometry import (
    is_point_in_polygon_3d
)

# =============================================================================
# = FREQUENZBEZOGENE HILFSFUNKTIONEN FÜR REFLEXIONEN
# =============================================================================

# Frequenzraster, auf dem die Wand-Reflexionskoeffizienten definiert sind.
# Muss zur Erzeugung der Polygon.reflection-Arrays passen!
# FREQ_GRID_REFLECTION = np.logspace(np.log10(50.0), np.log10(96000.0), 80)


def _get_R_for_band(poly: Polygon, freq_hz: float) -> float:
    """
    Liefert den skalaren Reflexionskoeffizienten R(f) einer Wand
    für eine gegebene Frequenz freq_hz.

    - poly.reflection ist im neuen Modell ein 1D-Array mit R(f_i) auf FREQ_GRID_REFLECTION.
    - Wir suchen den Index im Raster, der freq_hz am nächsten liegt.
    - Falls reflection ausnahmsweise noch ein float ist, geben wir ihn direkt zurück.
    """
    R = poly.reflection

    # Falls noch "altes" Modell (nur float), robust behandeln:
    if np.isscalar(R):
        return float(R)

    R = np.asarray(R).flatten()
    if R.size != FREQ_GRID_REFLECTION.size:
        # Sicherheitsfallback: wenn die Länge nicht passt, nimm einfach den Mittelwert
        return float(np.mean(R))

    # Index des Frequenzpunkts, der freq_hz am nächsten liegt:
    idx = int(np.argmin(np.abs(FREQ_GRID_REFLECTION - freq_hz)))
    return float(R[idx])

# def _get_medium_loss_linear(length_m: float, freq_hz: float, params: Dict[str, Any]) -> float:
#     """
#     Berechnet die lineare Amplitudendämpfung durch das Medium (z.B. Wasser)
#     für eine Strecke length_m bei einer Frequenz freq_hz.

#     Erwartet (optional) in params:
#         - 'water_absorption_db_per_m':
#               * float  -> konstanter Wert für alle Frequenzen
#               * 1D-Array gleicher Länge wie FREQ_GRID_REFLECTION -> freq-abhängig
#     Rückgabe:
#         Faktor <= 1.0 (1 = keine Dämpfung)
#     """
#     alpha = params.get('water_absorption_db_per_m', 0.0)

#     # Kein Eintrag -> keine Dämpfung
#     if alpha is None:
#         return 1.0

#     # Skalarer Wert (keine freq-Abhängigkeit)
#     if np.isscalar(alpha):
#         alpha_db_per_m = float(alpha)
#     else:
#         alpha_arr = np.asarray(alpha).flatten()
#         if alpha_arr.size != FREQ_GRID_REFLECTION.size:
#             # Fallback: Mittelwert, wenn etwas nicht passt
#             alpha_db_per_m = float(np.mean(alpha_arr))
#         else:
#             idx = int(np.argmin(np.abs(FREQ_GRID_REFLECTION - freq_hz)))
#             alpha_db_per_m = float(alpha_arr[idx])

#     # Gesamt-Dämpfung in dB entlang der Strecke
#     loss_db = alpha_db_per_m * max(length_m, 0.0)

#     # Wir arbeiten mit Amplituden (nicht Leistung) -> /20
#     loss_linear = 10.0 ** (-loss_db / 20.0)
#     return loss_linear



def _get_medium_loss_linear(dist_m, freq_hz, params):
    return water_absorption_linear(freq_hz, dist_m)

def _get_color_for_order(order: int, max_order: int = MAX_ORDNUNG):
    """
    Liefert eine Farbe für die gegebene Reflexionsordnung.
    Die Farben werden aus einer Matplotlib-Colormap gewählt.
    """
    cmap = plt.get_cmap("viridis")
    if order == 0:
        return "darkgreen"
    return cmap(order / (max_order + 1e-9))


# =============================================================================
# = GLOBALE VARIABLEN FÜR MULTIPROCESSING (Performance-Fix)
# =============================================================================

@dataclass                                  #erstellt automatisch Konstruktor (__init__), Vergleichsfunktionen
class WorkerContext:                #Dient als Behälter, um Daten für jeden CPU-Kern zu speichern. pro Worker-Prozess bekommt einmal diesen Kontext, anstatt jedes Mal alle Wände neu zu laden.
    """Kontext-Objekt für Worker-Prozesse. Wird nur einmal pro CPU-Kern initialisiert."""
    params: Dict[str, Any]                              #enthält quell empfängerpostion, einstellungen usw
    alle_reflektoren_dict: Dict[str, Polygon] # Wände für schnellen Zugriff per Name als dict
    alle_reflektoren_list: List[Polygon]     # Wände für die Sichtbarkeitsprüfung dieselben wände als listen für schleifen validierung 

_worker_ctx: Optional[WorkerContext] = None         # wird global im modul def, der speicherort in dem jeder worker-prozess seinen individuellen kontext hält 

def init_worker(params: Dict[str, Any], alle_reflektoren: List[Polygon]):                       # fkt die aufgerufen wird, wenn ein worker prozess startet 
    """
    Performance-Fix: Initialisiert den globalen Kontext im Worker-Prozess.
    Die Wände werden hier nur einmal geladen, nicht bei jedem Job neu berechnet.
    """
    global _worker_ctx                                                                          # sagt python dass globale variablie innerhalb der fkt geändert werden darf
    alle_reflektoren_dict = {r.name: r for r in alle_reflektoren}                               # umwandeln liste der wände in ein dict 
    _worker_ctx = WorkerContext(                                                        #Erstellt ein neues WorkerContext-Objekt und speichert
        params=params,                                                                  # Simulationsparameter (params), die Dictionary-Version aller Wände, Liste aller Wänd
        alle_reflektoren_dict=alle_reflektoren_dict,
        alle_reflektoren_list=alle_reflektoren                  #alle benötigten Daten pro Worker nur einmal im Speicher, nicht bei jedem Pfad-Aufruf neu. spart Zeit bei ISM
    )

# =============================================================================
# = KERN-ISM-LOGIK
# =============================================================================

def reflect_point_over_plane(p_in: np.ndarray, wall: Polygon) -> np.ndarray: 
    """Berechnet die Spiegelposition eines Punktes an einer Ebene (Wand)."""
    n = wall.normal                 #Ebenennormale n - Formel setzt voraus, dass normiert ∥n∥=1 Das ist typischerweise schon in Polygon.normal so vorgesehen.
    p_plane = wall.point                        # beliebiger punkt auf der ebene p0
    projection_vector = np.dot(p_in - p_plane, n) * n                   #Projektionsvektor: Der Vektor vom Punkt zur Ebene entlang der Normalen
    # Spiegelpunkt = Ursprungspunkt - 2 * Projektionsvektor
    return p_in - 2 * projection_vector                         # refl. formel p´ = p-2[(p-p0)*n]n 
# Vom Punkt entlang der Normalen zur Ebene (einmal abziehen), und dann nochmal genauso weit (zweites Mal), also insgesamt doppelt – dadurch landet „spiegelbildlich“ auf der anderen Seite.
'wenn n nicht normiert wäre müsste durch ||n||^2 hinweis : Funktion liefert nur die Spiegelposition; ob der spätere Reflexionspunkt innerhalb der Polygonfläche liegt, wird anderswo geprüft'

def _get_line_plane_intersection_general(p1: np.ndarray, p2: np.ndarray, poly: Polygon, eps: float = EPSILON) -> Optional[np.ndarray]:
    """Berechnet den Schnittpunkt eines Liniensegments (p1->p2) mit einer Polygon-Ebene."""
    line_vec = p2 - p1              # richt.vekt. der geraden d = p2-p1 parametergl. p(t)= p1+d
    normal = poly.normal 
    point_on_plane = poly.point         # Ebene: {x|n*(x-p0)=0}
    
    # Nenner: Wenn 0, ist die Linie parallel zur Ebene
    denominator = np.dot(line_vec, normal)          # in formel t = n(p0-p1)/n*d  , ist der nenner nd 
    #parallelfall liegt vor wenn |n*d| < eps -> keine eindeutige schnittstelle (parallel oder nahezu paralellel -> NONE)
    if abs(denominator) < eps:
        return None 
        
    # Zähler: Distanz von p1 zur Ebene
    numerator = np.dot(point_on_plane - p1, normal) 
    t = numerator / denominator # Skalar t: p_schnitt = p1 + t * line_vec dies ist exakte lösung für schnittpunkt kandidat 
    
    # Schnittpunkt muss innerhalb des Segments liegen (0 < t < 1)
    if not (eps < t < 1.0 - eps):                   #segement test nur innere SP gelten , kleine sicherheitsabstände eps bringt numerische stabilität 
        return None
        
    intersection_point = p1 + t * line_vec      #tatsächlicher punkt 
    
    # TEUERER SCHRITT: Prüfen, ob der Schnittpunkt innerhalb der Polygon-Grenzen liegt,Prüft nicht nur die Ebene, sondern die Flächenbegrenzung
    if is_point_in_polygon_3d(intersection_point, poly.vertices, normal, eps):          #True → Schnitt innerhalb der Wand → gib den Punkt zurück; sonst None
        return intersection_point
        
    return None



def finde_potentielle_quellen(params: Dict[str, Any], alle_reflektoren: List[Polygon]) -> List[ImageSource]:
    """Generiert Spiegelquellen mit Back-Face Culling (Rückseiten-Verwerfung)."""
    q_start_pos = params['quelle_pos']
    e_pos = params['empfaenger_pos']
    
    max_ord = params.get('max_ordnung', MAX_ORDNUNG)
    max_path_length = SCHALLGESCHWINDIGKEIT * (RIR_TIME_FIXED_MS / 1000.0)
    sim_freq_hz = params.get('freq_hz', params.get('center_freq_hz', 1000.0))

    quellen_liste: List[ImageSource] = [ImageSource(
        pos=q_start_pos.copy(), amp_mult=1.0, order=0, history=['S']
    )]      
    q_idx = 0
    reflektoren_dict = {r.name: r for r in alle_reflektoren}

    while q_idx < len(quellen_liste):
        basis_quelle = quellen_liste[q_idx]
        q_idx += 1

        if basis_quelle.order >= max_ord: continue
        if basis_quelle.amp_mult < AMP_MIN_CUTOFF: continue

        letzte_wand_name = basis_quelle.history[-1]

        for reflektor_name, reflektor_obj in reflektoren_dict.items():
            if reflektor_name == letzte_wand_name: continue

            # --- OPTIMIERUNG: BACK-FACE CULLING ---
            # Liegt die Quelle VOR der Wand? Falls dahinter: Ignorieren.
            vec_to_source = basis_quelle.pos - reflektor_obj.point
            if np.dot(vec_to_source, reflektor_obj.normal) <= EPSILON:
                continue
            # --------------------------------------

            neue_spiegel_pos = reflect_point_over_plane(basis_quelle.pos, reflektor_obj)

            dist_min = np.linalg.norm(neue_spiegel_pos - e_pos)
            if dist_min > max_path_length: continue
            
            R_f =_get_R_for_band(reflektor_obj, sim_freq_hz)
            neue_amp_mult = basis_quelle.amp_mult * R_f

            quellen_liste.append(ImageSource(
                pos=neue_spiegel_pos,
                amp_mult=neue_amp_mult,
                order=basis_quelle.order + 1,
                history=basis_quelle.history + [reflektor_name]
            ))
            
    return quellen_liste


def is_path_segment_obstructed(p1: np.ndarray, p2: np.ndarray, all_walls: List[Polygon], ignore_wall_name: Optional[str] = None) -> bool:
    """Prüft auf Blockierung mit schnellem Bounding-Box Check."""
    
    # Bounding Box des Strahls berechnen
    ray_min = np.minimum(p1, p2) - 0.01
    ray_max = np.maximum(p1, p2) + 0.01

    for wall in all_walls:
        if wall.name == ignore_wall_name: continue

        # --- OPTIMIERUNG: BOUNDING BOX CHECK ---
        # Wenn die Box des Strahls die Box der Wand nicht berührt -> Weiter.
        if (ray_max[0] < wall.min_coords[0] or ray_min[0] > wall.max_coords[0] or
            ray_max[1] < wall.min_coords[1] or ray_min[1] > wall.max_coords[1] or
            ray_max[2] < wall.min_coords[2] or ray_min[2] > wall.max_coords[2]):
            continue
        # ---------------------------------------

        if _get_line_plane_intersection_general(p1, p2, wall) is not None: 
            return True
            
    return False



# def finde_potentielle_quellen(params: Dict[str, Any], alle_reflektoren: List[Polygon]) -> List[ImageSource]:        # fkt erhält params und all refl - liste von imagesource objekten return, jede davon mögliche spiegelquelle im raum 
#     """Generiert Spiegelquellen bis MAX_ORDNUNG (exponentieller Engpass) mit Pruning."""
#     q_start_pos = params['quelle_pos']                                                              #gibt alle spiegelquellen bis maxorder zurück
#     e_pos = params['empfaenger_pos']                                                                # echte Q E pos aus params 
    
#     max_ord = params.get('max_ordnung', MAX_ORDNUNG)                                                    # obergrenze erlaubt Override durch params um zu erfahren was maxorder ist
#     #time_limit_ms = params.get('rir_time_limit_ms', RIR_TIME_LIMIT_MS_DEFAULT)
#     # Pruning-Distanz: Maximale Pfadlänge, die noch in der RIR sichtbar ist
#     max_path_length = SCHALLGESCHWINDIGKEIT * (RIR_TIME_FIXED_MS / 1000.0)                                 # Wie weit darf ein Signal maximal unterwegs sein, bis es aus der RIR-Zeitachse herausfällt? erstes pruninng

#     # *** NEU: Simulationsfrequenz für die Reflexionsfaktoren ***
#     # Versuche verschiedene Keys, je nachdem wie du params befüllst:
#     sim_freq_hz = params.get('freq_hz', params.get('center_freq_hz', 1000.0))

#     # Initialisierung mit Direktschall (K=0)
#     quellen_liste: List[ImageSource] = [ImageSource(                                                    # START IST DIREKTWEG ORDER = 0, amp ist 1 und 1/r^2 wird später berücksichtigt
#         pos=q_start_pos.copy(), amp_mult=1.0, order=0, history=['S']                                    
#     )]      
#     q_idx = 0                                                                                           # iteration wie in Breitensuche (BFS) über dioese liste und hängen neu spiegelquellen hinten an
#     reflektoren_dict = {r.name: r for r in alle_reflektoren}                                            # schneller zugriff auf wände per name (später wichtig für historie regeln)

#     while q_idx < len(quellen_liste):                                                   #BFS schleifen- basis quelle ist de expandierte Knoten (spiegelquellen) 
#         basis_quelle = quellen_liste[q_idx]                                             # neue kinder ans ende der liste level order
#         q_idx += 1                              

#         # PRUNING 1: Maximale Ordnung erreicht
#         if basis_quelle.order >= max_ord:                                               # wenn max order erreicht dann spiegeln stoppen
#             continue
#         # PRUNING 2: Amplitude zu gering (Energie-Cutoff)
#         if basis_quelle.amp_mult < AMP_MIN_CUTOFF:                                      #  amp mult unter cutoff, lohnt ssich weiteres spiegeln nicht energetisch vernachlässigbar
#             continue                                                                            #SPÄTER RAUSCHPEGEL UNTER RAUSCHPEGEL WIRDS LEISER ALS HÖRBAR

#         letzte_wand_name = basis_quelle.history[-1]                                                 #LETZTES EREIGNIS IN HIST BEI ORDER 0 IST 'S' '

#         for reflektor_name, reflektor_obj in reflektoren_dict.items():          #Schleife über alle wände die potentielle nächste reflexion haben
#             ''' Doppel-Reflexions-Regel: Nicht an derselben Wand spiegeln !'''
#             if reflektor_name == letzte_wand_name: 
#                 continue

#             neue_spiegel_pos = reflect_point_over_plane(basis_quelle.pos, reflektor_obj)                    #erzeugt die neue spiegelquelle der basis_quelle an gewählte wand 

#             # PRUNING 3: Zeit-Cutoff (Pfad ist zu lang für RIR-Fenster) 
#             dist_min = np.linalg.norm(neue_spiegel_pos - e_pos)             # setze eine untere schranke, damit wenn selbst diese minimale distanz schon länger  
#             if dist_min > max_path_length:                                      #als die maximale sichtbare fenster der rir machts kein sinn weiterzumachen
#                 continue        #dist_min= || spiegelquell -empfänger||
            
#             R_f =_get_R_for_band(reflektor_obj, sim_freq_hz)            #frqabhängiger reflfaktor für diese wand
#             neue_amp_mult = basis_quelle.amp_mult * R_f

#             # hänge neue Quelle zur Liste
#             quellen_liste.append(ImageSource(
#                 pos=neue_spiegel_pos,
#                 amp_mult = neue_amp_mult , #die alte Zeile produziert einen ganzen R(f)-Vektor, wo du einen einzigen float brauchst. Damit ist die komplette ISM-Energieberechnung physikalisch falsch.
#                 order=basis_quelle.order + 1,       #ordnung erhöht sich
#                 history=basis_quelle.history + [reflektor_name]         # hist erweitern mit den neuen werten daten
#             ))
#     '''Gibt die komplette Liste von potentiellen Quellen zurück (Direktschall + alle generierten Spiegelquellen vor Sichtbarkeitsprüfung)'''
#     return quellen_liste
# ''' bis hier Historie speichert die Wandfolge → später Pfadrekonstruktion (nächster Teil)'''


# def is_path_segment_obstructed(p1: np.ndarray, p2: np.ndarray, all_walls: List[Polygon], ignore_wall_name: Optional[str] = None) -> bool:
#     """Prüft, ob das Segment [p1, p2] durch eine andere Wand blockiert wird (Direktprüfung)."""
#     for wall in all_walls:                                  # liste alle wände die als hinderniss wirken können
#         if wall.name == ignore_wall_name: continue              #wand die ignorieert werden soll zb die wand an der gerade reflektiert wird 
#         # Wenn ein Schnittpunkt existiert, ist der Pfad blockiert
#         if _get_line_plane_intersection_general(p1, p2, wall) is not None: return True          #schnediet das segment die ebene der wand, SPkt im Polygon? wenn nicht NONE - dann VALIDE da es einen SPkt gibt, also wand liegt im weg 
#     return False                            # Wenn nach der Schleife keine Wand das Segment geschnitten hat da kein hindernis schalweg wäre frei -> false 




def validiere_eine_quelle(q_spiegel: ImageSource) -> Optional[Path]:
    """
    Wird parallel ausgeführt. Rekonstruiert den Pfad und prüft die Sichtbarkeit. 
    Aus dieser Spiegelquelle den echten physikalischen Pfad (mit Reflexionspunkten) rekonstruieren und prüfen, ob er geometrisch/sichtbar gültig ist.
    """
    global _worker_ctx                                                                #Zugriff auf die globale Variable _worker_ctx, in der pro Worker-Prozess alle wichtigen Daten (Parameter und Wände) gespeichert sind.
    if _worker_ctx is None: return None                  # falls noch nicht initialisiert wurde kann es ohne infos über wände und parameter kann fkt nicht arbeiten 
    
    params = _worker_ctx.params                         
    alle_reflektoren_dict = _worker_ctx.alle_reflektoren_dict
    alle_reflektoren_list = _worker_ctx.alle_reflektoren_list # Zugriff auf einmalig geladene Wände
    max_order = _worker_ctx.params.get("max_ordnung", MAX_ORDNUNG)

    q_echt_pos = params['quelle_pos']           # reale pos der physikalischen quelle
    e_pos = params['empfaenger_pos']            # pos des empfängers mikrofon/hydrofon

    # Direktschall (K=0)
    if q_spiegel.order == 0:                #
        if not is_path_segment_obstructed(q_echt_pos, e_pos, alle_reflektoren_list):                # prüft freie bahn 
            return Path(image_source=q_spiegel, points=[q_echt_pos.copy(), e_pos.copy()])       # ist weg frei erzeugt path objekt mit image_source = q_spiegel (ordnung 0) und points = [Q,E] #R+ckgabe repräsentiert direktschall
        return None     #blockierter weg -> NONE

    rekonstruierter_pfad: List[np.ndarray] = [e_pos.copy()]                         # anfang des pfades liste mit nur einem array dann pfad rückwärts aufbauen
    aktueller_sichtpunkt = e_pos.copy()                                     # bestimmt von welchem Punkt aus wir gerade „zurückschauen“
    aktuelle_spiegelquelle_pos = q_spiegel.pos.copy()                   #Position der Spiegelquelle der höchsten Ordnung - aus Spiegelquelle + E rekonstruieren wir rückwärts die reflexionspkte

    # Pfad-Rekonstruktions-Schleife (läuft K-mal rückwärts)
    for i in range(q_spiegel.order, 0, -1):                     #schlefe läuft  von maxorder bis 1 runter 
        wand_name = q_spiegel.history[i]                                # ['s', 'wand1', 'wand3', 'wand2'] Index 0 s = source, 1...k wände in reihenfolge die histi holt bei i von K ->1 
        wand = alle_reflektoren_dict.get(wand_name)                 # holt passende polygon objekt zur wand über ihren namen aus dem dict 
        if wand is None: return None                                # falls nicht gefunden exists gib NONE 

        '''
        mathmatische berechnung reflexuionspunkt 
        Geometrische Interpretation:

        Wir betrachten das Segment [aktueller_sichtpunkt, aktuelle_spiegelquelle_pos]:

        Beim ersten Durchlauf:
        aktueller_sichtpunkt = Empfänger
        aktuelle_spiegelquelle_pos = Spiegelquelle höchster Ordnung.

        Der reale Schallweg in der ISM kann so rekonstruiert werden:
        Er muss an der Wand wand reflektieren. Der Reflexionspunkt liegt am Schnittpunkt des Segments mit der Wandebene.

        _get_line_plane_intersection_general(...):
        Berechnet diesen Schnittpunkt und prüft, ob er: im Liniensegment liegt innerhalb der Wandfläche (Polygon) liegt.
        '''

        # 1. Reflexionspunkt-Berechnung
        reflexionspunkt = _get_line_plane_intersection_general(             #eingänge empfäger/vorheriger reflexionspunkt, relevante Spiegelquelle und das Objekt mit Ebenengleichung und normalenvektor
            aktueller_sichtpunkt, aktuelle_spiegelquelle_pos, wand              
        )
        if reflexionspunkt is None:
            return None # Pfad ungültig (Schnittpunkt außerhalb der Wandfläche)

        # 2. Sichtbarkeitsprüfung des Segments (Rx/Ri -> Ri-1)
        if is_path_segment_obstructed(aktueller_sichtpunkt, reflexionspunkt, alle_reflektoren_list, ignore_wall_name=wand_name):        # ist Segment zwischen dem aktuellen Sichtpunkt und dem Reflexionspunkt von anderen Wänden blockiert ?
            return None # Pfad blockiert                                                        #ignore wll weil eigene wand nicht als hinderniss gelten soll

        rekonstruierter_pfad.append(reflexionspunkt.copy())
        aktueller_sichtpunkt = reflexionspunkt.copy()
        # Spiegele die Quelle weiter, um den nächsten Reflexionspunkt zu finden
        aktuelle_spiegelquelle_pos = reflect_point_over_plane(aktuelle_spiegelquelle_pos, wand)

    # 3. Letztes Segment: Reflexionspunkt[1] -> Q_echt
    if q_spiegel.order > 0:         # kein direktschalll 
        erste_wand_name = q_spiegel.history[1]              # aus hist hole erste reflektierende wand i=1 weil 0 ist 's'
        if is_path_segment_obstructed(aktueller_sichtpunkt, q_echt_pos, alle_reflektoren_list, ignore_wall_name=erste_wand_name):
            return None

    rekonstruierter_pfad.append(q_echt_pos.copy())                  # relfexionspunkt kommt ans ende der der pfadliste
    rekonstruierter_pfad.reverse() # Pfad von Q nach E
    
    # Pfadlängen bestimmen
    punkte = np.array(rekonstruierter_pfad)
    seg_vecs = punkte[1:] - punkte[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    total_len = float(np.sum(seg_lens))

    pfad = Path(
        image_source=q_spiegel,
        points=rekonstruierter_pfad,
        color=_get_color_for_order(q_spiegel.order, max_order),
        segment_length=seg_lens.tolist(),
        total_length=total_len)


    return pfad

    #return Path(image_source=q_spiegel, points=rekonstruierter_pfad)        #zeuge ein Path-Objekt mit: image_source (die zugehörige Spiegelquelle) points (Liste aller Punkte des Weges) 


# def calculate_rir_from_sources(valide_pfade: List[Path], params: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
#     """
#     Berechnet RIR-Daten. 
#     NEU: Fügt 'Jitter' (zeitliches Verschmieren) hinzu, um raue Wände zu simulieren.
#     """
#     rir_daten: List[Dict[str, Any]] = []
#     if not valide_pfade: 
#         return [], {}
    
#     sim_freq_hz = params.get('freq_hz', 1000.0)
    
#     # --- JITTER EINSTELLUNG ---
#     # Echozeiten nicht exakt auf einem idealen Raster liegen, Spitzen im RIR leicht verschoben auftreten, die Echodichte realistischer wird, sich Reflexionspfade nicht perfekt gleichmäßig verteilen, Frequenzgänge glatter und natürlicher wirken
#     # Wieviel ms darf eine Reflexion 'wackeln'? 
#     # 0.1 ms bei kleinen Räumen, 2.0 ms bei riesigen Hallen.
#     # Da du 192kHz (0.005ms Samplezeit) hast, ist 0.005 ms schon effektiv.
#     JITTER_AMOUNT_MS = 0.005  
    
#     # Referenz berechnen (wie gehabt)
#     direktschall_pfad = next((obj for obj in valide_pfade if obj.image_source.order == 0), None)
#     if direktschall_pfad:
#         pts = np.array(direktschall_pfad.points)
#         dist_ref = float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))
#     else:
#         dist_ref = 1.0
    
#     medium_loss_ref = _get_medium_loss_linear(dist_ref, sim_freq_hz, params)
#     ref_amp_calc = (1.0 * medium_loss_ref) / dist_ref if dist_ref > 1e-6 else 1.0
#     ref_info = { "ref_dist_m": dist_ref, "ref_amp_calc": ref_amp_calc }

#     for pfad_obj in valide_pfade:
#         q = pfad_obj.image_source
        
#         # Geometrische Länge berechnen
#         pts = np.array(pfad_obj.points)
#         total_length = float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))

#         if total_length > EPSILON:
#             medium_loss = _get_medium_loss_linear(total_length, sim_freq_hz, params)
#             amp_abstand = (q.amp_mult * medium_loss) / total_length
#             amp_linear = amp_abstand / ref_amp_calc
#             amp_db = 20 * np.log10(amp_linear + 1e-12)
            
#             # --- JITTER BERECHNUNG ---
#             delay_geom = (total_length / SCHALLGESCHWINDIGKEIT) * 1000.0
            
#             if q.order > 0:
#                 # Zufallswert zwischen -JITTER und +JITTER
#                 # Je höher die Ordnung, desto mehr 'verschmiert' es oft in der Realität
#                 jitter = np.random.uniform(-JITTER_AMOUNT_MS, JITTER_AMOUNT_MS)
#                 final_delay = delay_geom + jitter
#             else:
#                 # Direktschall bleibt knackscharf
#                 final_delay = delay_geom

#             rir_daten.append({
#                 'delay_ms': final_delay,  # <-- Hier nutzen wir den 'wackeligen' Wert
#                 'amp_linear': amp_linear, 
#                 'amp_db': amp_db,
#                 'ord': q.order,
#                 'color': pfad_obj.color,
#                 'hist': q.history[1:],
#                 'reflection_product': q.amp_mult,
#                 'medium_loss': medium_loss,
#                 'points': pts
#             })
#     return rir_daten, ref_info


def calculate_rir_from_sources(valide_pfade: List[Path], params: Dict[str, Any]):
    """
    Berechnet die Raumimpulsantwort (RIR).

    Modi:
    - Klassisch (Dirac-Impulse):          USE_FREQUENCY_DEPENDENT_RIR = False
    - Frequenzabhängig (Mini-FIR / IFFT): USE_FREQUENCY_DEPENDENT_RIR = True
    """

    if not valide_pfade:
        return [], {}

    # ----------------------------------------------------------------------
    # Basisparameter
    # ----------------------------------------------------------------------
    fs = params.get("fs", 192000)
    freq_grid = FREQ_GRID_REFLECTION
    Nf = len(freq_grid)

    # Wand-Dict (falls nicht vorhanden)
    if "alle_wand_dict" not in params:
        params["alle_wand_dict"] = {w.name: w for w in params.get("alle_reflektoren", [])}

    # ----------------------------------------------------------------------
    # Referenzpfad finden (für Normalisierung)
    # ----------------------------------------------------------------------
    direktschall = next((p for p in valide_pfade if p.image_source.order == 0), None)

    if direktschall:
        pts = np.array(direktschall.points)
        dist_ref = float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))
    else:
        dist_ref = 1.0

    sim_freq_hz = params.get("freq_hz", 1000.0)
    medium_loss_ref = _get_medium_loss_linear(dist_ref, sim_freq_hz, params)
    ref_amp_calc = (1.0 * medium_loss_ref) / dist_ref if dist_ref > 1e-9 else 1.0

    rir_daten: List[Dict[str, Any]] = []

    # ======================================================================
    # 🟩 MODUS A — FREQUENZABHÄNGIGE MINI-FIR RIR
    # ======================================================================
    if USE_FREQUENCY_DEPENDENT_RIR:

        fir_paths = []
        max_delay_samples = 0
        # Definiere dein Sende-Band (10 kHz bis 80 kHz)
        F_MIN = 10000.0
        F_MAX = 80000.0

        for pfad in valide_pfade:
            pts = np.array(pfad.points)
            total_length = float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))

            # 1) Verzögerung in Samples
            delay_s = total_length / SCHALLGESCHWINDIGKEIT
            delay_samples = int(round(delay_s * fs))

            # 2) Jitter in Samples
            if USE_JITTER and pfad.image_source.order > 0:
                jitter_s = JITTER_AMOUNT_MS / 1000.0
                jitter_samples = int(round(jitter_s * fs))
                if jitter_samples > 0:
                    delay_samples += np.random.randint(-jitter_samples,
                                                       jitter_samples + 1)

            # keine negativen Indizes
            delay_samples = max(0, delay_samples)

            max_delay_samples = max(max_delay_samples, delay_samples)

            # 3) Frequenzgang H(f)
            # H = np.ones(Nf, dtype=complex)

            # 1. Wir starten mit einem leeren Spektrum (Alles 0)
            H = np.zeros(Nf, dtype=complex)
            
            # 2. Wir setzen das Spektrum NUR im Bereich 10k-80k auf 1.0
            # Das simuliert "konstante Sendeleistung" in diesem Band.
            # freq_grid kommt aus config.py
            mask = (freq_grid >= F_MIN) & (freq_grid <= F_MAX)
            H[mask] = 1.0

            for wall_name in pfad.image_source.history[1:]:
                wall = params["alle_wand_dict"][wall_name]
                H *= wall.reflection

            A = np.array([
                _get_medium_loss_linear(total_length, f, params)
                for f in freq_grid
            ])
            H *= A

            # 4) Mini-FIR
            h_path = np.real(np.fft.irfft(H, n=Nf * 2))

            # 5) Normalisierung
            medium_loss = _get_medium_loss_linear(total_length, sim_freq_hz, params)
            amp_abstand = (pfad.image_source.amp_mult * medium_loss) / total_length
            norm_amp = amp_abstand / ref_amp_calc
            h_path *= norm_amp
            amp_linear = norm_amp   # ← lokale Kopie
            amp_db = 20*np.log10(amp_linear + 1e-12)


            fir_paths.append((delay_samples, h_path, pfad))

        # 6) Globale RIR zusammenbauen
        if not fir_paths:
            return [], {}

        fir_len = len(fir_paths[0][1])
        rir_len = max_delay_samples + fir_len + 50
        rir = np.zeros(rir_len)

        for delay, h_path, pfad in fir_paths:
            end = delay + fir_len
            if end > rir_len:
                end = rir_len
                h_slice = h_path[:rir_len - delay]
            else:
                h_slice = h_path

            rir[delay:end] += h_slice

            rir_daten.append({
                "delay_ms": delay * 1000.0 / fs,
                "ord": pfad.image_source.order,
                "amp_linear": amp_linear,
                "amp_db":amp_db,
                "color": pfad.color,
                "hist": pfad.image_source.history[1:],
                "points": pfad.points
            })

        return rir_daten, {"rir": rir}

    # ======================================================================
    # 🟥 MODUS B — KLASSISCHER DIRAC-MODUS
    # ======================================================================
    else:
        for pfad_obj in valide_pfade:
            pts = np.array(pfad_obj.points)
            total_length = float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))

            # Amplitude
            medium_loss = _get_medium_loss_linear(total_length, sim_freq_hz, params)
            amp_abstand = (pfad_obj.image_source.amp_mult * medium_loss) / total_length
            amp_linear = amp_abstand / ref_amp_calc
            amp_db = 20 * np.log10(amp_linear + 1e-12)

            # Verzögerung in ms
            delay_s = total_length / SCHALLGESCHWINDIGKEIT
            delay_ms = delay_s * 1000.0

            if USE_JITTER and pfad_obj.image_source.order > 0:
                delay_ms += np.random.uniform(-JITTER_AMOUNT_MS, JITTER_AMOUNT_MS)
                delay_ms = max(0.0, delay_ms)

            rir_daten.append({
                'delay_ms': delay_ms,
                'amp_linear': amp_linear,
                'amp_db': amp_db,
                'ord': pfad_obj.image_source.order,
                'color': pfad_obj.color,
                'hist': pfad_obj.image_source.history[1:],
                'points': pts
            })

        return rir_daten, {}


# =============================================================================
# = T60-BERECHNUNG (Unverändert übernommen)
# =============================================================================

# def berechne_t60(rir_daten: List[Dict[str, Any]], sample_rate: int = 44100) -> tuple:
#     """Berechnet die Abklingkurve (EDC) und die Nachhallzeit T60 (T30-Extrapolation)."""
#     if not rir_daten or len(rir_daten) < 2: return None, None, None, None, None, None, None, None
    
#     # Hilfsfunktion: Tukey-Fenster für sanften Abfall am Ende der RIR
#     def _tukey_fade_tail(x, frac=0.06):
#         n = len(x); m = int(max(1, n * frac))
#         if m <= 1: return x
#         w = np.ones(n); t = np.linspace(0, np.pi, m); w[-m:] = 0.5 * (1 + np.cos(t)); return x * w
        
#     # Hilfsfunktion: Robuste lineare Regression (entfernt Ausreißer)
#     def _robust_line_fit(x, y, n_iter=2):
#         x = x.copy(); y = y.copy()
#         if len(x) < 2: return 0, 0
#         for _ in range(n_iter):
#             if len(x) < 2: return 0, 0
#             A = np.vstack([x, np.ones_like(x)]).T
#             m, b = np.linalg.lstsq(A, y, rcond=None)[0]
#             resid = y - (m * x + b); q1, q3 = np.percentile(resid, [25, 75])
#             iqr = q3 - q1; keep = (resid >= q1 - 1.5 * iqr) & (resid <= q3 + 1.5 * iqr)
#             x, y = x[keep], y[keep];
#             if len(x) < 5: break
#         if len(x) < 2: return 0, 0
#         A = np.vstack([x, np.ones_like(x)]).T 
#         m, b = np.linalg.lstsq(A, y, rcond=None)[0]
#         return m, b
        
#     # 1. RIR-Erstellung
#     max_delay_s = max(r['delay_ms'] for r in rir_daten) / 1000.0
#     rir_len = max(2, int(np.ceil(max_delay_s * sample_rate)) + 1)
#     rir = np.zeros(rir_len)
#     for impuls in rir_daten:
#         j = int(round(impuls['delay_ms'] / 1000.0 * sample_rate))
#         if 0 <= j < rir_len: rir[j] += impuls['amp_linear']
    
#     rir = _tukey_fade_tail(rir, frac=0.06)
#     eps = 1e-20
#     energie = rir * rir  
    
#     # 2. Energy Decay Curve (EDC)
#     edc = np.cumsum(energie[::-1])[::-1]; edc = np.maximum.accumulate(edc[::-1])[::-1]
#     if edc[0] <= eps: return None, None, None, None, None, None, None, None
#     edc_db = 10.0 * np.log10((edc + eps) / (edc[0] + eps))
#     zeit_ms = np.arange(rir_len) / sample_rate * 1000.0
    
#     # 3. T30-Bereich finden (-5 dB bis -35 dB)
#     if np.nanmin(edc_db) > -35.0: return zeit_ms, edc_db, None, None, None, None, None, None
#     t_lo = float(np.interp(-5.0,  edc_db[::-1], zeit_ms[::-1]))
#     t_hi = float(np.interp(-35.0, edc_db[::-1], zeit_ms[::-1]))
#     if t_hi <= t_lo: return zeit_ms, edc_db, None, None, None, None, None, None
    
#     # 4. Lineare Regression (T30)
#     sel = (zeit_ms >= t_lo) & (zeit_ms <= t_hi); x_fit = zeit_ms[sel]; y_fit = edc_db[sel]
#     if len(x_fit) < 8: return zeit_ms, edc_db, None, None, None, None, None, None
#     m_db_per_ms, b_db = _robust_line_fit(x_fit, y_fit, n_iter=2)
    
#     # 5. T60-Extrapolation
#     if m_db_per_ms >= 0: return zeit_ms, edc_db, None, None, None, None, None, None
#     t60_ms = -60.0 / m_db_per_ms
#     fit_x = np.array([t_lo, t_hi]); fit_y = m_db_per_ms * fit_x + b_db
    
#     return zeit_ms, edc_db, t60_ms, fit_x, fit_y, m_db_per_ms, b_db,  (t_lo, t_hi)


def berechne_t60(rir_daten: List[Dict[str, Any]], sample_rate: int = 44100) -> tuple:
    """Berechnet die Abklingkurve (EDC) und die Nachhallzeit T60 (T30-Extrapolation)."""
    if not rir_daten or len(rir_daten) < 2:
        return None, None, None, None, None, None, None, None
    
    
    
    # RIR-Berechnung
    max_delay_s = max(r['delay_ms'] for r in rir_daten) / 1000.0
    rir_len = int(np.ceil(max_delay_s * sample_rate)) + 1
    rir = np.zeros(rir_len)
    for impuls in rir_daten:
        j = int(round(impuls['delay_ms'] / 1000.0 * sample_rate))
        if 0 <= j < rir_len:
            rir[j] += impuls['amp_linear']
    
    # Berechne die EDC
    energie = rir * rir
    edc = np.cumsum(energie[::-1])[::-1]
    edc_db = 10 * np.log10(edc / edc[0])  # in dB
    zeit_ms = np.arange(rir_len) / sample_rate * 1000.0
    
           # 3. T30-Bereich finden (-5 dB bis -35 dB)
    # -----------------------------------------
    # Prüfen: fällt die EDC überhaupt weit genug ab?
    if np.nanmin(edc_db) > -35.0:
        # -> Kein T60 möglich
        return zeit_ms, edc_db, None, None, None, None, None, None

    # Zeiten interpolieren
    t_lo = float(np.interp(-5.0,  edc_db[::-1], zeit_ms[::-1]))
    t_hi = float(np.interp(-35.0, edc_db[::-1], zeit_ms[::-1]))

    # Sicherheit: Reihenfolge korrekt?
    if t_hi <= t_lo:
        return zeit_ms, edc_db, None, None, None, None, None, None

    # 4. Punkte im T30-Bereich auswählen
    sel = (zeit_ms >= t_lo) & (zeit_ms <= t_hi)
    x_fit = zeit_ms[sel]
    y_fit = edc_db[sel]

    if len(x_fit) < 8:
        return zeit_ms, edc_db, None, None, None, None, None, None

    # 5. Robuste Regression
    m_db_per_ms, b_db = _robust_line_fit(x_fit, y_fit, n_iter=2)

    if m_db_per_ms >= 0:
        return zeit_ms, edc_db, None, None, None, None, None, None

    # Gerade definieren
    fit_x = np.array([t_lo, t_hi], dtype=float)
    fit_y = m_db_per_ms * fit_x + b_db

    # 6. T60: wo schneidet die Gerade -60 dB?
    t60_ms = -60.0 / m_db_per_ms

    return (
        zeit_ms,
        edc_db,
        t60_ms,
        fit_x,
        fit_y,
        m_db_per_ms,
        b_db,
        (t_lo, t_hi)
    )


def _tukey_fade_tail(x, frac=0.06):
    n = len(x)
    m = int(max(1, n * frac))
    if m <= 1:
        return x
    w = np.ones(n)
    t = np.linspace(0, np.pi, m)
    w[-m:] = 0.5 * (1 + np.cos(t))
    return x * w

def _robust_line_fit(x, y, n_iter=2):
    x = x.copy()
    y = y.copy()
    if len(x) < 2:
        return 0, 0

    # Wiederholt lineare Regression → entfernt Ausreißer (IQR-Methode)
    for _ in range(n_iter):
        if len(x) < 2:
            return 0, 0

        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]

        resid = y - (m * x + b)
        q1, q3 = np.percentile(resid, [25, 75])
        iqr = q3 - q1
        keep = (resid >= q1 - 1.5 * iqr) & (resid <= q3 + 1.5 * iqr)

        x, y = x[keep], y[keep]

        if len(x) < 5:
            break

    if len(x) < 2:
        return 0, 0

    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b

def berechne_echodichte_impuls_convolve(
    rir_daten: List[Dict[str, Any]],
    fs: float,
    bin_width_ms: float = ECHO_DENSITY_BIN_WIDTH_MS
):
    """
    Berechnet die Impuls-Echodichte mittels einer Faltung des Impuls-Vektors mit einem Fenster.
    Diese Methode verwendet ein Rechteckfenster zur Glättung der Impulse.
    """

    # Prüfen, ob überhaupt Impulse existieren
    # Wenn rir_daten leer ist → keine Pfade → Echo-Dichte = leer
    if not rir_daten:
        return np.array([]), np.array([])


    # ------------------------------------------------------------
    # 1. Umwandlung der Bin-Breite von Millisekunden → Samples
    # ------------------------------------------------------------

    # bin_width_ms ist z.B. 3 ms, 5 ms, ...
    # Umrechnung in Sekunden: bin_width_ms / 1000
    # Multiplikation mit Abtastrate fs → Anzahl Samples, die dieses Zeitfenster hat
    bin_width_samples = int((bin_width_ms / 1000.0) * fs)

    # Sicherstellen, dass die Fenstergröße nie 0 ist
    # Selbst wenn bin_width_ms extrem klein ist
    if bin_width_samples < 1:
        bin_width_samples = 1  # Minimale Fenstergröße


    # ------------------------------------------------------------
    # 2. Maximale Verzögerung berechnen (für die Länge des Arrays)
    # ------------------------------------------------------------

    # Wir suchen den höchsten delay_ms-Wert aller Impulse
    # delay_ms ist in Millisekunden → /1000 = Sekunden
    max_delay_s = max(r['delay_ms'] for r in rir_daten) / 1000.0

    # Länge des Arrays in Samples:
    # max_delay_s * fs → Zeitpunkt des letzten Impulses
    # +1, damit auch der letzte Index vorhanden ist
    N_samples = int(np.ceil(max_delay_s * fs)) + 1


    # ------------------------------------------------------------
    # 3. Zeitachse für die spätere Ausgabe erzeugen
    # ------------------------------------------------------------

    # Wir erzeugen einen Vektor von 0 bis max_delay_s
    # Gleichmäßig verteilt über N_samples Punkte
    time_axis = np.linspace(0, max_delay_s, N_samples)


    # ------------------------------------------------------------
    # 4. Impuls-Vektor erstellen (Diskrete Delta-Impulse)
    # ------------------------------------------------------------

    # Wir erzeugen ein Array voller Nullen:
    # Jedes Element entspricht einer Zeitprobe (Sample)
    impulse_vector = np.zeros(N_samples)

    # Jetzt setzen wir jeden Pfad als "Impuls" an der entsprechenden Sample-Position:
    for r in rir_daten:
        # Zeit des Pfades in Sekunden → * fs = Sampleindex
        delay_idx = int(np.round((r['delay_ms'] / 1000.0) * fs))

        # Nur setzen, wenn innerhalb der Array-Grenzen
        if 0 <= delay_idx < N_samples:
            # Ankunft eines einzelnen Pfades → Wert +1
            # Also: Impulse zählen!
            impulse_vector[delay_idx] += 1.0


    # ------------------------------------------------------------
    # 5. Rechteckfenster zur Glättung erstellen
    # ------------------------------------------------------------

    # Ein Fenster der Länge bin_width_samples voller Einsen
    # Beispiel: L=10 → [1,1,1,1,1,1,1,1,1,1]
    window = np.ones(bin_width_samples)


    # ------------------------------------------------------------
    # 6. Faltung: Impuls-Vektor * Fenster
    # ------------------------------------------------------------
    
    # np.convolve gleicht jeden Samplewert mit der Umgebung ab.
    # mode="same" bedeutet:
    # - Ausgabe hat die gleiche Länge wie impulse_vector
    # - Fenster gleitet über die Impulse
    smoothed_impulses = np.convolve(impulse_vector, window, mode='same')


    # ------------------------------------------------------------
    # 7. Echo-Dichte = geglättete Impulsspur
    # ------------------------------------------------------------

    # Diese geglätteten Werte zeigen:
    # → wie viele Impulse innerhalb eines Bin-Fensters auftreten
    echo_density = smoothed_impulses


    # ------------------------------------------------------------
    # 8. Zeitachse an Ausgabelänge anpassen
    # ------------------------------------------------------------

    # time_axis und echo_density können theoretisch minimal unterschiedliche Längen haben
    # wir schneiden die Zeitachse exakt passend auf die Echo-Dichte zu
    t_echo = time_axis[:len(echo_density)]


    # ------------------------------------------------------------
    # 9. Ergebnis zurückgeben
    # ------------------------------------------------------------

    # t_echo → Zeitachse (in Sekunden)
    # echo_density → geglättete Impulsanzahl pro Fenster
    return t_echo, echo_density
