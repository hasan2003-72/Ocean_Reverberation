"/Users/hasanyesil/Ocean_Reverb/Measurements_hay"


import numpy as np
import matplotlib
matplotlib.use('Agg')  # Keine Fenster öffnen
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
import os

# =======================================================
# 1. EINSTELLUNGEN & PFADE
# =======================================================

PathToMeasurements = "/Users/hasanyesil/Ocean_Reverb/Measurements_hay"
PlotPath = "./RIR_Analysis_T60_Spectrum/"
SoundOfSpeed = 1474.7

# Zeitfenster für die Analyse (Trimmen)
pre_trigger_time = 0.005  # 5 ms Vorlauf
post_trigger_time = 0.150 # 150 ms Nachlauf (für T60 wichtig)

os.makedirs(PlotPath, exist_ok=True)

# =======================================================
# 2. DATEN DURCHGEHEN
# =======================================================

if not os.path.exists(PathToMeasurements):
    print(f"❌ Pfad nicht gefunden: {PathToMeasurements}")
    Files = []
else:
    Files = [f for f in os.listdir(PathToMeasurements)
             if os.path.isdir(os.path.join(PathToMeasurements, f)) and not f.startswith('.')]

print(f"Gefundene Messungen: {len(Files)}")
t60_results = [] 

for FolderName in Files:
    MeasurementFolder = os.path.join(PathToMeasurements, FolderName)
    
    # Wir nutzen wieder Loopback als Referenz (genauer!)
    # Falls du DOCH FileOut willst, ändere Filename_Ref zu "internal_ppfFileOut_0.wav"
    Filename_Meas = "internal_ppfHydIn_0.wav"
    Filename_Ref  = "internal_ppfHydIn_1.wav" 

    PathMeas = os.path.join(MeasurementFolder, Filename_Meas)
    PathRef  = os.path.join(MeasurementFolder, Filename_Ref)

    if not (os.path.exists(PathMeas) and os.path.exists(PathRef)):
        print(f"Überspringe {FolderName}: Dateien fehlen.")
        continue

    print(f"\nBearbeite: {FolderName}")

    try:
        # --- LADEN & NORMIEREN ---
        fs, sig_meas = wavfile.read(PathMeas)
        fs_ref, sig_ref = wavfile.read(PathRef)

        if sig_meas.ndim > 1: sig_meas = sig_meas[:, 0]
        if sig_ref.ndim > 1: sig_ref = sig_ref[:, 0]
        
        sig_meas = np.nan_to_num(sig_meas.astype(float))
        sig_ref = np.nan_to_num(sig_ref.astype(float))

        # Padding
        maxlen = max(len(sig_meas), len(sig_ref))
        sig_meas = np.pad(sig_meas, (0, maxlen - len(sig_meas)))
        sig_ref  = np.pad(sig_ref, (0, maxlen - len(sig_ref)))

        # Normierung (Wichtig!)
        sig_meas /= (np.max(np.abs(sig_meas)) + 1e-10)
        sig_ref  /= (np.max(np.abs(sig_ref)) + 1e-10)

        # =======================================================
        # 3. ENTFALTUNG & TRIMMEN (Wie im vorigen Skript)
        # =======================================================
        n_fft = len(sig_meas)
        Y = fft(sig_meas, n=n_fft)
        X = fft(sig_ref, n=n_fft)

        # Deconvolution
        H = Y * np.conjugate(X) / (np.abs(X)**2 + 1e-4)
        rir_broadband = np.real(ifft(H))

        # Peak finden
        sample_delay = np.argmax(np.abs(rir_broadband))
        
        # --- TRIMMEN (Genau wie vorher) ---
        start_idx = max(0, sample_delay - int(pre_trigger_time * fs))
        end_idx = min(len(rir_broadband), sample_delay + int(post_trigger_time * fs))
        
        # Das ist jetzt unser "Fenster" für T60 und Spektrum
        rir_win = rir_broadband[start_idx:end_idx]
        
        # Zeitachse für dieses Fenster (0 = Start des Fensters)
        t_rir = np.arange(len(rir_win)) / fs 
        # Optional: Zeitachse so schieben, dass 0 beim Peak ist
        t_rir_rel = t_rir - pre_trigger_time

    except Exception as e:
        print(f"Fehler in {FolderName}: {e}")
        continue

    # =======================================================
    # 4. EDC & T60 BERECHNUNG
    # =======================================================
    
    # Schroeder Integration
    rir_sq = rir_win**2
    edc = np.cumsum(rir_sq[::-1])[::-1]
    
    # Normieren auf 0 dB
    edc_max = np.max(edc)
    if edc_max == 0: edc_max = 1e-16
    edc_db = 10 * np.log10(edc / edc_max + 1e-16)

    # Fit (T10 -> T60 extrapolation)
    # Wir suchen den Bereich -2 dB bis -12 dB (Standard für kurze Impulse)
    start_db = -2
    end_db   = -12
    
    idx_start = np.where(edc_db <= start_db)[0]
    idx_end   = np.where(edc_db <= end_db)[0]

    t60_str = "N/A"
    line_fit = None

    if len(idx_start) > 0 and len(idx_end) > 0:
        i1 = idx_start[0]
        i2 = idx_end[0]
        
        if i2 > i1 + 5: # Mindestens 5 Samples Abstand für Fit
            t_fit = t_rir[i1:i2]
            db_fit = edc_db[i1:i2]
            
            slope, intercept = np.polyfit(t_fit, db_fit, 1)
            
            if slope != 0:
                T60 = -60 / slope
                t60_str = f"{T60:.4f} s"
                # Linie für Plot berechnen
                line_fit = slope * t_rir + intercept
    
    t60_results.append((FolderName, t60_str))

    # =======================================================
    # 5. PLOTTEN (Kombinierter Plot)
    # =======================================================
    
    # Wir machen 2 Subplots untereinander:
    # Oben: RIR + EDC
    # Unten: Frequenzspektrum
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # --- PLOT 1: ZEITBEREICH ---
    # RIR in dB umrechnen für Vergleichbarkeit
    rir_abs = np.abs(rir_win) + 1e-16
    rir_db = 20 * np.log10(rir_abs / np.max(rir_abs))
    
    ax1.plot(t_rir_rel, rir_db, color='blue', alpha=0.3, linewidth=0.5, label='RIR [dB]')
    ax1.plot(t_rir_rel, edc_db, color='red', linewidth=1.5, label='EDC')
    
    if line_fit is not None:
        ax1.plot(t_rir_rel, line_fit, 'k--', linewidth=1.0, label=f'Fit (T60={t60_str})')

    short_name = FolderName.replace("Hay_2025-12-10_", "")
    ax1.set_title(f"Impuls & Nachhall: {short_name}")
    ax1.set_ylabel("Pegel [dB]")
    ax1.set_xlabel("Zeit relativ zum Peak [s]")
    ax1.set_ylim([-60, 5])
    ax1.set_xlim([-pre_trigger_time, post_trigger_time])
    ax1.grid(True)
    ax1.legend(loc="upper right")
    
    # --- PLOT 2: FREQUENZSPEKTRUM ---
    # FFT des Fensters
    Y_spec = fft(rir_win)
    freqs = np.fft.fftfreq(len(rir_win), 1/fs)
    
    # Nur positive Frequenzen
    keep = slice(0, len(freqs)//2)
    f_pos = freqs[keep]
    mag_pos = np.abs(Y_spec[keep])
    
    # In dB
    if np.max(mag_pos) > 0:
        mag_db = 20 * np.log10(mag_pos / np.max(mag_pos) + 1e-16)
    else:
        mag_db = np.zeros_like(mag_pos)

    ax2.semilogx(f_pos, mag_db, color='darkgreen')
    ax2.set_title("Amplitudenspektrum (normiert)")
    ax2.set_xlabel("Frequenz [Hz]")
    ax2.set_ylabel("Magnitude [dB]")
    ax2.set_xlim([1000, 90000]) # 1 kHz bis 90 kHz anzeigen
    ax2.set_ylim([-60, 5])
    ax2.grid(True, which="both", alpha=0.5)

    plt.tight_layout()
    
    # Speichern
    save_file = os.path.join(PlotPath, f"{FolderName}_Analysis.png")
    plt.savefig(save_file, dpi=150)
    plt.close(fig)

# =======================================================
# 6. ZUSAMMENFASSUNG AUSGEBEN
# =======================================================
print("\n" + "="*40)
print(f"{'Messung':<30} | {'T60 [s]':<10}")
print("-" * 40)
for name, val in t60_results:
    short = name.replace("Hay_2025-12-10_", "")
    print(f"{short:<30} | {val}")
print("="*40)
print(f"Plots gespeichert in: {PlotPath}")
