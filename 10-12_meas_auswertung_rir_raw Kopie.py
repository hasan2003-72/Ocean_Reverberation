import numpy as np
import matplotlib
matplotlib.use('Agg') # Keine Fenster öffnen (schneller)
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import butter, sosfilt, hilbert
import os

# =======================================================
# 1. PFADE & SETUP
# =======================================================


PathToMeasurements = "/Users/hasanyesil/Ocean_Reverb/Measurements_hay" 
PlotPath = "./RIR_Envelope_Long_Plots-3/"
SoundOfSpeed = 1474.7

os.makedirs(PlotPath, exist_ok=True)

TargetFrequencies = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
Bandbreite_Hz = 4000 

# =======================================================
# 2. ZEIT-EINSTELLUNGEN (HIER LÄNGE STEUERN)
# =======================================================

# Vorlauf (um den Anstieg sauber zu sehen)
pre_trigger_time = 0.0  # 2 ms

# Nachlauf (HIER LÄNGER MACHEN für "langanhaltend")
# 0.050 = 50ms | 0.100 = 100ms
post_trigger_time = 0.1 

# =======================================================
# 3. FUNKTIONEN
# =======================================================

def get_bandpass(center_freq, fs, bandwidth):
    nyquist = 0.5 * fs
    low = (center_freq - bandwidth/2) / nyquist
    high = (center_freq + bandwidth/2) / nyquist
    if high >= 0.99: high = 0.99
    if low <= 0.01: low = 0.01
    return butter(4, [low, high], btype='band', output='sos')

# =======================================================
# 4. DATENVERARBEITUNG
# =======================================================

if not os.path.exists(PathToMeasurements):
    print(f"❌ Pfad nicht gefunden: {PathToMeasurements}")
    Files = []
else:
    Files = [f for f in os.listdir(PathToMeasurements)
             if os.path.isdir(os.path.join(PathToMeasurements, f)) and not f.startswith('.')]

print(f"Gefundene Ordner: {len(Files)}")

for FolderName in Files:
    MeasurementFolder = os.path.join(PathToMeasurements, FolderName)
    
    # Wir nutzen Loopback (HydIn_1) als Referenz
    Filename_Meas = "internal_ppfHydIn_0.wav"
    Filename_Ref  = "internal_ppfHydIn_1.wav"

    PathMeas = os.path.join(MeasurementFolder, Filename_Meas)
    PathRef  = os.path.join(MeasurementFolder, Filename_Ref)

    if not (os.path.exists(PathMeas) and os.path.exists(PathRef)):
        continue

    print(f"\nBearbeite: {FolderName}")

    try:
        # Laden
        fs, sig_meas = wavfile.read(PathMeas)
        fs_ref, sig_ref = wavfile.read(PathRef)
        
        # Stereo -> Mono
        if sig_meas.ndim > 1: sig_meas = sig_meas[:, 0]
        if sig_ref.ndim > 1: sig_ref = sig_ref[:, 0]
        
        # Float
        sig_meas = np.nan_to_num(sig_meas.astype(float))
        sig_ref = np.nan_to_num(sig_ref.astype(float))

        # Padding
        maxlen = max(len(sig_meas), len(sig_ref))
        sig_meas = np.pad(sig_meas, (0, maxlen - len(sig_meas)))
        sig_ref  = np.pad(sig_ref, (0, maxlen - len(sig_ref)))

        # Normierung (Wichtig für Stabilität)
        sig_meas /= (np.max(np.abs(sig_meas)) + 1e-10)
        sig_ref  /= (np.max(np.abs(sig_ref)) + 1e-10)

        # --- ENTFALTUNG (Loopback rausrechnen) ---
        n_fft = len(sig_meas)
        Y = fft(sig_meas, n=n_fft)
        X = fft(sig_ref, n=n_fft)
        
        # Deconvolution im Frequenzbereich
        H = Y * np.conjugate(X) / (np.abs(X)**2 + 1e-4)
        rir_broadband = np.real(ifft(H))

        # Peak finden (Laufzeit)
        sample_delay = np.argmax(np.abs(rir_broadband))
        dist_m = (sample_delay / fs) * SoundOfSpeed
        print(f"-> Distanz: {dist_m:.3f} m")

        # --- AUSSCHNITT WÄHLEN ---
        start_idx = max(0, sample_delay - int(pre_trigger_time * fs))
        end_idx = min(len(rir_broadband), sample_delay + int(post_trigger_time * fs))
        
        if end_idx <= start_idx: continue

        plot_len = end_idx - start_idx
        t_axis = (np.arange(plot_len) - (sample_delay - start_idx)) / fs

        # --- PLOTTEN (STACKED ENVELOPE) ---
        fig, ax1 = plt.subplots(figsize=(14, 8)) # Breiteres Bild
        
        offset = 0
        offset_step = 1.2 # Abstand der Kurven
        
        for freq in TargetFrequencies:
            sos = get_bandpass(freq, fs, Bandbreite_Hz)
            
            # 1. Filtern
            filtered_full = sosfilt(sos, rir_broadband)
            
            # 2. BETRAG BILDEN (Hilbert-Envelope)
            envelope_full = np.abs(hilbert(filtered_full))
            
            # 3. Ausschneiden
            sig_cut = envelope_full[start_idx:end_idx]
            
            # 4. Normieren (auf 1.0)
            local_max = np.max(sig_cut)
            if local_max > 0:
                sig_cut = sig_cut / local_max
            
            # 5. Plotten (mit Füllung)
            # Linie
            ax1.plot(t_axis, sig_cut + offset, color='black', linewidth=0.8)
            # Füllung (sieht bei Envelopes besser aus)
            ax1.fill_between(t_axis, sig_cut + offset, offset, color='blue', alpha=0.1)
            
            # Label
            ax1.text(t_axis[0], offset + 0.2, f"{freq/1000:.0f} kHz", 
                     fontweight='bold', color='darkblue', ha='right', va='center')
            
            offset += offset_step

        # Design
        short_name = FolderName.replace("Hay_2025-12-10_", "")
        ax1.set_title(f"Envelope (Langzeit): {short_name}\nDist: {dist_m:.3f} m", fontsize=14)
        ax1.set_xlabel("Zeit relativ zum Direktschall [s]", fontsize=12)
        ax1.set_yticks([])
        ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax1.set_xlim([t_axis[0], t_axis[-1]])

        # Obere Achse (Meter)
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        def time_to_dist(t): return t * SoundOfSpeed
        xticks = ax1.get_xticks()
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([f"{time_to_dist(t):.2f}" for t in xticks])
        ax2.set_xlabel(f"Wegstrecke [m] (relativ)", fontsize=11, color='red')
        ax2.tick_params(axis='x', colors='red')

        plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.1)

        save_path = os.path.join(PlotPath, f"{FolderName}_EnvLong.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    except Exception as e:
        print(f"Fehler in {FolderName}: {e}")

print("Fertig.")
