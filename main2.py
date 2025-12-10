# main_ism.py
import os
import datetime
import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

# Importiere alles aus den Modulen
from config import (
    SAVE_PLOT, OUTPUT_FOLDER, MAX_ORDNUNG, CMAP_NAME,ECHO_DENSITY_BIN_WIDTH_MS,
    ALLE_RAEUME, ACTIVE_RAEUME, FS, USE_RANDOM_POSITIONS)

from geometry import (definiere_waende,get_source_and_receiver_positions)

from ism_core import (
    finde_potentielle_quellen,
    validiere_eine_quelle,
    init_worker,
    calculate_rir_from_sources, berechne_t60,
    berechne_echodichte_impuls_convolve)

from plotting import plotte_raum_rir_und_echodichte, plot_edc_with_t60


# =====================================================================
# HAUPTPROGRAMM
# =====================================================================

if __name__ == "__main__":

    start_total = time.time()

    # -------------------------------------------------------
    # 1. Farbskala vorbereiten
    # -------------------------------------------------------
    K_RANKS = MAX_ORDNUNG
    cmap = plt.get_cmap(CMAP_NAME)
    color_map_k = {k: cmap((k - 1) / K_RANKS) for k in range(1, K_RANKS + 1)}

    #wasserabsorption
    

    # -------------------------------------------------------
    # 2. CPU-Kerne
    # -------------------------------------------------------
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"--- Verwende {num_cores} CPU-Kerne für die Parallelisierung ---")

    # -------------------------------------------------------
    # 3. Szenarien auswählen
    # -------------------------------------------------------
    szenarien = [ALLE_RAEUME[key] for key in ACTIVE_RAEUME]
    print("\n--- Aktive Szenarien ---")#
    for s in ACTIVE_RAEUME:
        print("   →", s)

    # -------------------------------------------------------
    # 4. Szenarien einzeln simulieren (JE RAUM 1 FIGURE)
    # -------------------------------------------------------
    for szenario in szenarien:

        print("\n===============================================")
        print(f"Starte Simulation: {szenario['name']}")
        print("===============================================")

        params = szenario.copy()
        params['max_ordnung'] = MAX_ORDNUNG
        
        # NEU: Volumen für Echodichte-Plot berechnen
        if 'raum_dim' in params:
            params['volume_m3'] = np.prod(params['raum_dim'])

        #wasserabsorption
        params.setdefault("freq_hz", 1000.0)
        params.setdefault("water_absorption_db_per_m", 0.0)


        # -------------------------------------------------------
        # 4.1 Geometrie laden
        # -------------------------------------------------------
        alle_reflektoren = definiere_waende(params)
        # WICHTIG für freq-abhängige RIR
        params["alle_reflektoren"] = alle_reflektoren
        params["alle_wand_dict"] = {w.name: w for w in alle_reflektoren}

        # -------------------------------------------------------
        # 4.2 Positionen bestimmen (Zufällig oder npy-fest)
        # -------------------------------------------------------
        min_dist = params.get('min_dist_wand', 0.5)

        # Wand-Infos für Positionierungsfunktion vorbereiten
        walls_for_pos = [
            {
                'name': w.name,
                'vertices': w.vertices,
                'normal': w.normal,
                'point': w.point
            }
            for w in alle_reflektoren
        ]

        # Logik für Positionen vereinheitlicht
        try:
            # SPEZIALFALL: Feste, "gute" Positionen für den Finnraum, um das Problem zu umgehen
            if 'finnraum' in params['name'].lower():
                print("[WARNUNG] Verwende hartkodierte 'gute' Positionen für Finnraum.")
                q_pos = np.array([24.8, 411.5, -14.3])
                e_pos = np.array([53.1, 389.4, -11.1])
                # Setze die Positionen auch in den params, damit die Plot-Funktion sie hat
                params['quelle_pos'] = q_pos
                params['empfaenger_pos'] = e_pos
            # Wenn NPY und KEINE zufälligen Positionen gewünscht sind, lade aus Datei
            elif params.get('geometrie_typ') == 'npy_custom' and not USE_RANDOM_POSITIONS:
                base_path = params['npy_path']
                q_pos = np.load(os.path.join(base_path, params['tx_file'])).flatten()
                e_pos = np.load(os.path.join(base_path, params['rx_file'])).flatten()
                print(f"[INFO] Feste NPY-Positionen geladen: Q={q_pos}, Rx={e_pos}")
            else:
                # In allen anderen Fällen (auch NPY mit Random-Flag) die Funktion nutzen
                q_pos, e_pos = get_source_and_receiver_positions(
                    walls_for_pos, min_wall_distance=min_dist
                )
        except Exception as e:
            print(f"[FEHLER] Positionierung fehlgeschlagen: {e}")
            continue

        params['quelle_pos'] = q_pos
        params['empfaenger_pos'] = e_pos

        # -------------------------------------------------------
        # 4.3 Potentielle Quellen generieren
        # -------------------------------------------------------
        start_sim = time.time()
        potentielle_quellen = finde_potentielle_quellen(params, alle_reflektoren)

        # -------------------------------------------------------
        # 4.4 Pfade parallel validieren
        # -------------------------------------------------------
        with multiprocessing.Pool(
            processes=num_cores,
            initializer=init_worker,
            initargs=(params, alle_reflektoren)
        ) as pool:
            result_list = pool.map(validiere_eine_quelle, potentielle_quellen, chunksize=64)

        valide_pfade = [p for p in result_list if p is not None]

        # Farben zuweisen
        for p in valide_pfade:
            k = p.image_source.order
            if k == 0:
                p.color = "darkgreen"
            else:
                p.color = color_map_k.get(k, "gray")

        print(f"-> {len(valide_pfade)} valide Pfade (Dauer: {time.time() - start_sim:.2f}s)")

        # 4.5 RIR berechnen
        rir_daten, ref_info = calculate_rir_from_sources(valide_pfade, params)

        # --- DIESER TEIL FEHLTE
        if not rir_daten:
            rir_array = np.zeros(100)
        else:
            # 1. Maximale Zeit finden
            max_delay = max(r['delay_ms'] for r in rir_daten)
            # 2. Länge in Samples berechnen
            num_samples = int(np.ceil((max_delay / 1000.0) * FS)) + 1000
            # 3. Leeres Array
            rir_array = np.zeros(num_samples)
            
            # 4. Werte füllen
            for r in rir_daten:
                idx = int(round((r['delay_ms'] / 1000.0) * FS))
                if 0 <= idx < num_samples:
                    rir_array[idx] += r['amp_linear']
        # ------------------------------------------------------

        print("RIR")
        print("Max delay in rir_daten:", max(r['delay_ms'] for r in rir_daten))
        # FIR-RIR in params für den Plot verfügbar machen
        rir_signal = ref_info.get("rir", None)
        params["rir"] = rir_signal
        params["USE_FREQUENCY_DEPENDENT_RIR"] = rir_signal is not None #flag für den plot setzen

        # 4.6 Echodichte berechnen
        t_echo, echo_density = berechne_echodichte_impuls_convolve(rir_daten, fs=FS, bin_width_ms=ECHO_DENSITY_BIN_WIDTH_MS)
        params['bin_width_ms'] = ECHO_DENSITY_BIN_WIDTH_MS
        print(params)

        # 4.7 T60 und EDC berechnen
        zeit_ms, edc_db, t60_ms, fit_x, fit_y, m_db_per_ms, b_db, rir = berechne_t60(rir_daten)

        #np.save("C:\\bachelorarbeit-hasan-yesil\\npy\\rir_meas_tank_5_ord.npy",rir)
        # 4.8 Plot erstellen
        fig_local, (ax_raum, ax_wave, ax_rir_db) = plotte_raum_rir_und_echodichte(
            valide_pfade=valide_pfade,
            rir_daten=rir_daten,
            rir_array=rir_array,
            echo_t=t_echo,
            echo_density=echo_density,
            params=params,
            color_map_k=color_map_k
        )
        # ... nach der Berechnung von rir_daten und rir_array ...

        print(f"--- DEBUG STATUS FÜR {params['name']} ---")
        print(f"Anzahl Wände: {len(alle_reflektoren)}")
        print(f"Anzahl Valide Pfade: {len(valide_pfade)}")
        print(f"Länge RIR Array: {len(rir_array)}")
        print(f"Max Amplitude: {np.max(np.abs(rir_array)) if len(rir_array)>0 else 0}")
        
        if len(valide_pfade) == 0:
            print("ACHTUNG: Keine Pfade gefunden! Plot wird wahrscheinlich leer sein oder Fehler werfen.")

        # -------------------------------------------------------
        # 4.8 Plot erstellen
        # -------------------------------------------------------
        # ... hier kommt dein Plot Aufruf ...
        # # 4.9 EDC und T60 Plotten
        # if zeit_ms is not None:
        #     plot_edc_with_t60(zeit_ms, edc_db, t60_ms, fit_x, fit_y)

        # 4.10 Speichern optional
        if SAVE_PLOT:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"Order_{MAX_ORDNUNG}_ISM_{params['name'].replace(' ', '_')}_{timestamp}.png"
            fig_local.savefig(os.path.join(OUTPUT_FOLDER, fname), dpi=300)
            print(f"   Bild gespeichert unter: {fname}")
        print("\n⏱️ Gesamtlaufzeit:", round(time.time() - start_total, 2), "s")
        plt.show()

