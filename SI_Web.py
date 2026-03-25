import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Core Physics & Math Engine
# ==========================================
class TDRSimulatorCore:
    def __init__(self):
        self.c0 = 3e8  
        self.mu_0 = 4 * np.pi * 1e-7 
        self.sigma_cu = 5.8e7 

    def calc_lossy_gamma(self, z0, f, er, df, type_mode, geom_1, geom_2):
        omega = 2 * np.pi * f
        f_safe = np.where(f == 0, 1e-10, f)
        alpha_d = (np.pi * f_safe * np.sqrt(er) / self.c0) * df
        R_s = np.sqrt(np.pi * f_safe * self.mu_0 / self.sigma_cu)
        
        if type_mode == 'microstrip':
            w_m = geom_1 / 1000.0  
            R_ac = R_s / w_m 
        elif type_mode == 'coaxial':
            d_m = geom_1 / 1000.0  
            D_m = geom_2 / 1000.0  
            R_ac = (R_s / np.pi) * ((1.0 / d_m) + (1.0 / D_m))
        else:
            R_ac = R_s / (0.1 / 1000.0) 

        alpha_c = R_ac / (2 * z0)
        alpha = alpha_c + alpha_d
        v = self.c0 / np.sqrt(er)
        beta = omega / v
        return alpha + 1j * beta

    def abcd_transmission_line_lossy(self, z0, length_mm, f, er, df, type_mode, geom_1, geom_2):
        length_m = length_mm / 1000.0
        gamma = self.calc_lossy_gamma(z0, f, er, df, type_mode, geom_1, geom_2)
        A = np.cosh(gamma * length_m)
        B = z0 * np.sinh(gamma * length_m)
        C = (1 / z0) * np.sinh(gamma * length_m)
        D = np.cosh(gamma * length_m)
        return A, B, C, D

    def abcd_open_stub_lossy(self, z_via, length_mm, f, er_via, df_via, drill_mm):
        if length_mm <= 0: return np.ones_like(f), np.zeros_like(f), np.zeros_like(f), np.ones_like(f)
        length_m = length_mm / 1000.0
        gamma = self.calc_lossy_gamma(z_via, f, er_via, df_via, 'microstrip', drill_mm, 0)
        Y_in = np.tanh(gamma * length_m) / z_via
        A = np.ones_like(f, dtype=complex)
        B = np.zeros_like(f, dtype=complex)
        C = Y_in
        D = np.ones_like(f, dtype=complex)
        return A, B, C, D

    def abcd_shunt_capacitor(self, c_pf, f):
        c_f = c_pf * 1e-12
        omega = 2 * np.pi * f
        Y = 1j * omega * c_f
        A = np.ones_like(f, dtype=complex)
        B = np.zeros_like(f, dtype=complex)
        C = Y
        D = np.ones_like(f, dtype=complex)
        return A, B, C, D

    def multiply_abcd(self, A1, B1, C1, D1, A2, B2, C2, D2):
        A = A1*A2 + B1*C2
        B = A1*B2 + B1*D2
        C = C1*A2 + D1*C2
        D = C1*B2 + D1*D2
        return A, B, C, D

    def get_channel_abcd(self, p, f):
        A1, B1, C1, D1 = self.abcd_transmission_line_lossy(p['z0'], p['l1'], f, p['er'], p['df'], p['type_l1'], p['geom_l1_1'], p['geom_l1_2'])
        Aa, Ba, Ca, Da = self.abcd_transmission_line_lossy(p['zvia'], p['l_active'], f, p['er'], p['df'], 'microstrip', p['drill'], 0) 
        As, Bs, Cs, Ds = self.abcd_open_stub_lossy(p['zvia'], p['l_stub'], f, p['er'], p['df'], p['drill'])
        
        At, Bt, Ct, Dt = self.multiply_abcd(A1, B1, C1, D1, Aa, Ba, Ca, Da)
        At, Bt, Ct, Dt = self.multiply_abcd(At, Bt, Ct, Dt, As, Bs, Cs, Ds)

        if p.get('conn_en', False):
            Ac1, Bc1, Cc1, Dc1 = self.abcd_shunt_capacitor(p['conn_c'], f)
            Ac2, Bc2, Cc2, Dc2 = self.abcd_transmission_line_lossy(p['conn_z0'], p['conn_len'], f, er=3.0, df=0.01, type_mode='microstrip', geom_1=10, geom_2=0)
            Ac3, Bc3, Cc3, Dc3 = self.abcd_shunt_capacitor(p['conn_c'] * 0.5, f)
            At, Bt, Ct, Dt = self.multiply_abcd(At, Bt, Ct, Dt, Ac1, Bc1, Cc1, Dc1)
            At, Bt, Ct, Dt = self.multiply_abcd(At, Bt, Ct, Dt, Ac2, Bc2, Cc2, Dc2)
            At, Bt, Ct, Dt = self.multiply_abcd(At, Bt, Ct, Dt, Ac3, Bc3, Cc3, Dc3)

        A2, B2, C2, D2 = self.abcd_transmission_line_lossy(p['z0'], p['l2'], f, p['er'], p['df'], p['type_l2'], p['geom_l2_1'], p['geom_l2_2'])
        At, Bt, Ct, Dt = self.multiply_abcd(At, Bt, Ct, Dt, A2, B2, C2, D2)
        return At, Bt, Ct, Dt

    def calculate_ffe_sbr_zeroforcing(self, p):
        baud_rate = p['dr_gbps'] / 2.0 if p['modulation'] == "PAM4" else p['dr_gbps']
        ui = 1.0 / (baud_rate * 1e9)
        samps_per_ui = 32
        dt = ui / samps_per_ui
        N = 256 * samps_per_ui  
        
        tx_pulse = np.zeros(N)
        pulse_start = 64 * samps_per_ui 
        tx_pulse[pulse_start : pulse_start + samps_per_ui] = 1.0
        
        freqs = np.fft.rfftfreq(N, dt)
        f_valid = freqs[freqs <= p['fmax'] * 1e9]
        At, Bt, Ct, Dt = self.get_channel_abcd(p, f_valid)
        S21_base = (2 * p['z0']) / (At*p['z0'] + Bt + Ct*(p['z0']**2) + Dt*p['z0'])
        
        if p['skew_ps'] > 0:
            S21_base = S21_base * np.cos(np.pi * f_valid * (p['skew_ps'] * 1e-12))
            
        S21 = np.zeros_like(freqs, dtype=complex)
        S21[freqs <= p['fmax'] * 1e9] = S21_base
        window = np.exp(-0.5 * (freqs / (p['fmax'] * 1e9 / 2.0))**2)
        rx_pulse = np.fft.irfft(np.fft.rfft(tx_pulse) * S21 * window, n=N)
        
        idx_main = np.argmax(rx_pulse)
        if idx_main < 2 * samps_per_ui or idx_main > N - 2 * samps_per_ui: return 0.0, 1.0, 0.0
            
        h_0 = rx_pulse[idx_main]
        h_m1 = rx_pulse[idx_main - samps_per_ui]
        h_m2 = rx_pulse[idx_main - 2 * samps_per_ui]
        h_p1 = rx_pulse[idx_main + samps_per_ui]
        h_p2 = rx_pulse[idx_main + 2 * samps_per_ui]
        
        H_matrix = np.array([[h_0, h_m1, h_m2], [h_p1, h_0, h_m1], [h_p2, h_p1, h_0]])
        target = np.array([0, 1, 0])
        try: taps = np.linalg.solve(H_matrix, target)
        except np.linalg.LinAlgError: return 0.0, 1.0, 0.0
            
        sum_taps = np.abs(taps).sum()
        if sum_taps == 0: sum_taps = 1.0
        return float(taps[0]/sum_taps), float(taps[1]/sum_taps), float(taps[2]/sum_taps)

    def run_tdr_simulation(self, p):
        f = np.linspace(0, p['fmax'] * 1e9, 4096)
        At, Bt, Ct, Dt = self.get_channel_abcd(p, f)
        z0 = p['z0']
        denom = At*z0 + Bt + Ct*(z0**2) + Dt*z0
        denom[denom == 0] = 1e-10 
        
        S11 = (At*z0 + Bt - Ct*(z0**2) - Dt*z0) / denom
        S21_base = (2 * z0) / denom
        if p['skew_ps'] > 0: S21 = S21_base * np.cos(np.pi * f * (p['skew_ps'] * 1e-12))
        else: S21 = S21_base

        IL_dB = 20 * np.log10(np.abs(S21) + 1e-12)
        f_c = p['fmax'] * 1e9 / 2.5 
        window = np.exp(-0.5 * (f / f_c)**2)
        step_response = np.cumsum(np.fft.irfft(S11 * window))
        Z_tdr = z0 * (1 + step_response) / (1 - step_response)
        
        v = self.c0 / np.sqrt(p['er'])
        distance_mm = ((v * (np.arange(len(Z_tdr)) * (1.0 / (2 * p['fmax'] * 1e9)))) / 2) * 1000
        return distance_mm, Z_tdr.real, S11, f, IL_dB

    def measure_eye_compliance(self, rx_steady, samps_per_ui, mod_type):
        num_symbols = len(rx_steady) // samps_per_ui
        folded = rx_steady[:num_symbols * samps_per_ui].reshape(-1, samps_per_ui)
        center_idx = np.argmax(np.var(folded, axis=0))
        samples = folded[:, center_idx]
        metrics = {"center_idx": center_idx}
        
        if mod_type == "NRZ":
            L1 = samples[samples > 0]; L0 = samples[samples <= 0]
            if len(L1)==0 or len(L0)==0: return {"Pass": False, "EH_mV": 0, "msg": "Eye fully closed"}
            eh = (np.min(L1) - np.max(L0)) * 1000 
            metrics.update({"EH_mV": eh, "Pass": eh > 15, "msg": f"EH: {eh:.1f} mV"})
        elif mod_type == "PAM4":
            max_v = np.max(samples)
            L3 = samples[samples > 0.5 * max_v]; L2 = samples[(samples > 0) & (samples <= 0.5 * max_v)]
            L1 = samples[(samples > -0.5 * max_v) & (samples <= 0)]; L0 = samples[samples <= -0.5 * max_v]
            if any(len(lvl) == 0 for lvl in [L3, L2, L1, L0]):
                return {"Pass": False, "EH_min_mV": 0, "RLM": 0, "msg": "PAM4 eye severely corrupted"}
            eh_top = (np.min(L3) - np.max(L2)) * 1000
            eh_mid = (np.min(L2) - np.max(L1)) * 1000
            eh_bot = (np.min(L1) - np.max(L0)) * 1000
            min_eh = min(eh_top, eh_mid, eh_bot)
            V3, V2, V1, V0 = np.mean(L3), np.mean(L2), np.mean(L1), np.mean(L0)
            V_avg = (V3 - V0) / 3.0
            rlm = min(V3-V2, V2-V1, V1-V0) / V_avg if V_avg > 0 else 0
            is_pass = (min_eh > 10) and (rlm > 0.85)
            metrics.update({"EH_min_mV": min_eh, "RLM": rlm, "Pass": is_pass, "msg": f"Min EH: {min_eh:.1f} mV | RLM: {rlm:.3f}"})
        return metrics

    def run_eye_diagram(self, p):
        baud_rate_gbaud = p['dr_gbps'] / 2.0 if p['modulation'] == "PAM4" else p['dr_gbps']
        ui = 1.0 / (baud_rate_gbaud * 1e9) 
        samps_per_ui = 32
        dt = ui / samps_per_ui
        N = 800 * samps_per_ui
        np.random.seed(42) 
        levels = [-1.0, -0.3333, 0.3333, 1.0] if p['modulation'] == "PAM4" else [-1.0, 1.0]
        tx_signal_raw = np.repeat(np.random.choice(levels, size=800), samps_per_ui)
        
        tx_pre = np.roll(tx_signal_raw, -samps_per_ui)
        tx_post = np.roll(tx_signal_raw, samps_per_ui)
        tx_signal = p['ffe_pre'] * tx_pre + p['ffe_main'] * tx_signal_raw + p['ffe_post'] * tx_post
        
        freqs = np.fft.rfftfreq(N, dt)
        f_valid = freqs[freqs <= p['fmax'] * 1e9]
        At, Bt, Ct, Dt = self.get_channel_abcd(p, f_valid)
        S21_base = (2 * p['z0']) / (At*p['z0'] + Bt + Ct*(p['z0']**2) + Dt*p['z0'])
        
        if p['skew_ps'] > 0: S21_base = S21_base * np.cos(np.pi * f_valid * (p['skew_ps'] * 1e-12))
            
        S21 = np.zeros_like(freqs, dtype=complex)
        S21[freqs <= p['fmax'] * 1e9] = S21_base
        window = np.exp(-0.5 * (freqs / (p['fmax'] * 1e9 / 2.0))**2)
        rx_signal = np.fft.irfft(np.fft.rfft(tx_signal) * S21 * window, n=N)
        
        rx_steady = rx_signal[100 * samps_per_ui:]
        metrics = self.measure_eye_compliance(rx_steady, samps_per_ui, p['modulation'])
        return np.arange(N) * dt, rx_steady, ui, samps_per_ui, baud_rate_gbaud, metrics


# ==========================================
# Streamlit Web UI
# ==========================================
st.set_page_config(page_title="SI Studio Web", layout="wide", initial_sidebar_state="expanded")

# --- State Management ---
if 'ffe_pre' not in st.session_state: st.session_state.ffe_pre = 0.0
if 'ffe_main' not in st.session_state: st.session_state.ffe_main = 1.0
if 'ffe_post' not in st.session_state: st.session_state.ffe_post = 0.0

simulator = TDRSimulatorCore()

# --- Sidebar ---
st.sidebar.title("⚙️ System Parameters")

st.sidebar.header("1. Standard & Protocol")
mod_type = st.sidebar.selectbox("Modulation", ["NRZ", "PAM4"], index=1)
dr_gbps = st.sidebar.number_input("Data Rate (Gbps)", value=112.0, step=1.0)

st.sidebar.header("2. Channel & Material")
z0 = st.sidebar.number_input("Target Impedance Z0 (Ω)", value=100.0)
fmax = st.sidebar.number_input("Max Bandwidth fmax (GHz)", value=60.0)
er = st.sidebar.number_input("Dielectric Constant (Dk/Er)", value=3.8)
df = st.sidebar.number_input("Dissipation Factor (Df)", value=0.005, format="%.4f")
trace_w = st.sidebar.number_input("Trace Width W (mil)", value=4.0)

st.sidebar.header("3. Via & Advanced SI")
auto_zvia = st.sidebar.checkbox("Auto-calculate Via Impedance", value=True)
gnd_vias = st.sidebar.selectbox("GND Return Vias", ["0 (None)", "2 (Symmetric)", "4 (Coaxial)"], index=2)
drill = st.sidebar.number_input("Via Drill Size (mil)", value=10.0)
antipad = st.sidebar.number_input("Antipad Size (mil)", value=45.0)
pitch = st.sidebar.number_input("Diff Pair Pitch (mil)", value=39.4)
sg_pitch = st.sidebar.number_input("Signal-to-GND Pitch (mil)", value=35.0)
skew_ps = st.sidebar.number_input("Intra-pair Skew (ps)", value=0.0, step=1.0)

def calc_zvia(er, drill, antipad, pitch, sg_pitch, gnd_vias):
    if drill <= 0: return 50.0
    if gnd_vias == "0 (None)": return 140.0
    z_single = (60.0 / np.sqrt(er)) * np.log((2 * sg_pitch) / drill)
    if gnd_vias == "4 (Coaxial)": z_single *= 0.85 
    coupling_factor = 1.0 - (drill / pitch) if pitch > drill else 1.0
    z_diff = 2 * z_single * coupling_factor + (antipad - 30) * 0.5 
    return max(10.0, z_diff)

if auto_zvia:
    zvia = calc_zvia(er, drill, antipad, pitch, sg_pitch, gnd_vias)
    st.sidebar.info(f"💡 Estimated Z_via: **{zvia:.1f} Ω**")
else:
    zvia = st.sidebar.number_input("Custom Via Z0 (Ω)", value=85.0)

st.sidebar.header("4. 🔌 Connector Model")
conn_en = st.sidebar.checkbox("Enable Connector Effects", value=True)
conn_z0 = st.sidebar.number_input("Connector Z0 (Ω)", value=90.0, disabled=not conn_en)
conn_len = st.sidebar.number_input("Connector Length (mm)", value=15.0, disabled=not conn_en)
conn_c = st.sidebar.number_input("SMT Pad Capacitance (pF)", value=0.15, step=0.05, disabled=not conn_en)

st.sidebar.header("5. Length Routing (mm)")
l1 = st.sidebar.number_input("L1 Length (TX -> Via)", value=50.0)
l_active = st.sidebar.number_input("L_active (Active Via Length)", value=1.6)
l_stub = st.sidebar.number_input("Stub Length", value=0.0)
l2 = st.sidebar.number_input("L2 Length (Via -> RX)", value=100.0)

st.sidebar.header("6. TX FFE (Equalizer)")
col_eq1, col_eq2, col_eq3 = st.sidebar.columns(3)
with col_eq1: st.session_state.ffe_pre = st.number_input("Pre (c-1)", value=st.session_state.ffe_pre, step=0.01)
with col_eq2: st.session_state.ffe_main = st.number_input("Main (c0)", value=st.session_state.ffe_main, step=0.01)
with col_eq3: st.session_state.ffe_post = st.number_input("Post (c+1)", value=st.session_state.ffe_post, step=0.01)

p = {
    'modulation': mod_type, 'dr_gbps': dr_gbps, 'z0': z0, 'fmax': fmax,
    'er': er, 'df': df, 'zvia': zvia, 'l1': l1, 'l_active': l_active,
    'l_stub': l_stub, 'l2': l2, 'trace_w': trace_w, 'skew_ps': skew_ps, 'drill': drill,
    'ffe_pre': st.session_state.ffe_pre, 'ffe_main': st.session_state.ffe_main, 'ffe_post': st.session_state.ffe_post,
    'conn_en': conn_en, 'conn_z0': conn_z0, 'conn_len': conn_len, 'conn_c': conn_c,
    'type_l1': 'microstrip', 'type_l2': 'microstrip',
    'geom_l1_1': trace_w, 'geom_l2_1': trace_w, 'geom_l1_2': 0, 'geom_l2_2': 0
}

if st.sidebar.button("✨ Auto-Tune FFE (SBR Zero-Forcing)", use_container_width=True, type="primary"):
    with st.spinner("Running SBR Link Training..."):
        c_pre, c_main, c_post = simulator.calculate_ffe_sbr_zeroforcing(p)
        st.session_state.ffe_pre = c_pre
        st.session_state.ffe_main = c_main
        st.session_state.ffe_post = c_post
        st.rerun()

# --- Main Panel ---
st.title("🚀 SI Studio Web - High-Speed Channel System Assessment")
st.markdown("An end-to-end AI Server channel simulator based on true dielectric loss, multiple reflections, GND return path effects, and TX FFE equalization.")

with st.spinner("Executing matrix operations and waveform reconstruction..."):
    dist, z_tdr, s11_array, f_array, IL_dB = simulator.run_tdr_simulation(p)
    t_eye, rx_steady, ui, samps_per_ui, baud_rate, metrics = simulator.run_eye_diagram(p)

min_z, max_z = np.min(z_tdr), np.max(z_tdr)
nyq_f = baud_rate / 2.0
il_nyq = IL_dB[np.argmin(np.abs(f_array - (nyq_f*1e9)))]

conn_len_added = p['conn_len'] if p['conn_en'] else 0
total_len_mm = p['l1'] + p['l_active'] + conn_len_added + p['l2']
tof_ns = total_len_mm / ((3e8 / np.sqrt(p['er'])) * 1e-6)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Physical Length / ToF", f"{total_len_mm:.1f} mm", f"{tof_ns:.3f} ns delay", delta_color="off")
col2.metric("Nyquist Loss", f"{nyq_f:.1f} GHz", f"{il_nyq:.2f} dB", delta_color="inverse")
col3.metric("TDR Impedance Variation", f"{min_z:.1f} ~ {max_z:.1f} Ω", "Reflection Index", delta_color="off")
col4.metric("Eye Spec (EH / RLM)", f"{metrics.get('EH_min_mV', metrics.get('EH_mV', 0)):.1f} mV", f"{metrics.get('RLM', 0):.3f} RLM" if mod_type == "PAM4" else "N/A", delta_color="normal" if metrics['Pass'] else "inverse")

if not metrics['Pass']:
    st.error(f"🛑 Conclusion: FAIL - {metrics['msg']} (Consider running Auto-Tune FFE or adjusting channel geometry)")
else:
    st.success(f"🏆 Conclusion: PASS - {metrics['msg']} (Compliant with {mod_type} optical/electrical interface specs)")

if p['skew_ps'] > 0:
    st.warning(f"⚠️ Skew Warning: {p['skew_ps']}ps of intra-pair skew causes an insertion loss null (destructive interference) at {1 / (2 * p['skew_ps'] * 1e-12) / 1e9:.1f} GHz.")

st.divider()

plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), dpi=120)
fig.patch.set_facecolor('#0e1117') 

# Plot 1: TDR
max_plot_dist = total_len_mm + 15
mask = (dist >= 0) & (dist <= max_plot_dist)
ax1.plot(dist[mask], z_tdr[mask], color='#00e5ff', linewidth=2)
ax1.axhline(p['z0'], color='gray', linestyle='--')
ax1.axvspan(p['l1'], p['l1'] + p['l_active'], color='#ff5252', alpha=0.3, label='Via')
if p['conn_en']:
    conn_start = p['l1'] + p['l_active']
    ax1.axvspan(conn_start, conn_start + p['conn_len'], color='#ffcc00', alpha=0.3, label='Connector')
ax1.set_title("1. TDR Impedance Profile", loc='left', color='white', fontsize=14)
ax1.grid(True, color='#333333', linestyle=':')
ax1.set_ylim(40, 140)
ax1.set_ylabel("Impedance (Ω)")
ax1.legend(loc='upper right')

# Plot 2: IL
ax2.plot(f_array/1e9, IL_dB, color='#ffcc00', linewidth=2)
ax2.axvline(nyq_f, color='red', linestyle='--', alpha=0.5, label=f'Nyquist ({nyq_f} GHz)')
ax2.set_title("2. Insertion Loss (S21)", loc='left', color='white', fontsize=14)
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("dB")
ax2.grid(True, color='#333333', linestyle=':')
ax2.legend(loc='lower left')
ax2.set_ylim(-80, 5)
ax2.set_xlim(0, p['fmax'])

# Plot 3: Eye Diagram
s_fold = 2 * samps_per_ui 
t_mod = np.linspace(0, 2, s_fold) 
for i in range(min(len(rx_steady) // s_fold, 300)):
    idx = i * s_fold
    ax3.plot(t_mod, rx_steady[idx:idx+s_fold], color='#39ff14', alpha=0.15, linewidth=1)
    
center_t = (metrics.get('center_idx', 0) / samps_per_ui)
if center_t > 1.0: center_t -= 1.0 
ax3.axvline(center_t, color='red', linestyle=':', label='Sampling Phase')

if p['modulation'] == "PAM4":
    for thresh in [-0.666, 0, 0.666]: ax3.axhline(thresh, color='gray', linestyle=':', alpha=0.5)
    
ax3.set_title(f"3. Receiver Eye Diagram ({mod_type} with FFE)", loc='left', color='white', fontsize=14)
ax3.set_xlabel("Unit Interval (UI)")
ax3.set_ylabel("Voltage")
ax3.set_xlim(0, 2)
ax3.set_ylim(-1.5, 1.5)
ax3.grid(True, color='#333333', linestyle=':')
ax3.legend(loc='upper right')

fig.tight_layout(pad=3.0)
st.pyplot(fig)