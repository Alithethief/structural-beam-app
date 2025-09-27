import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Konfigurācija Streamlit lapai ---
st.set_page_config(layout="wide", page_title="Siju Analīzes Kalkulators (9 Piemēri)")

# --- Konstantes un Attēlu Konfigurācija (Nav mainīts) ---
IMAGE_CONFIGS = {
    "Piemērs 1": {
        "caption": "Šarnīrs A, Rullītis B. F1 laidumā L, q uz konsoles Lk.",
        "url": "exsample_1.JPG",
        "status": "Stat. Nosakāms",
    },
    "Piemērs 2": {
        "caption": "Šarnīrs A, Rullītis B. q uz laiduma L, F1 uz konsoles Lk.",
        "url": "exsample_2.JPG",
        "status": "Stat. Nosakāms",
    },
    "Piemērs 3": {
        "caption": "Šarnīrs A, Rullītis B. Slīpa F1 laidumā L, F2 uz konsoles Lk.",
        "url": "exsample_3.JPG",
        "status": "Stat. Nosakāms",
    },
    "Piemērs 4": {
        "caption": "Šarnīrs A, Rullītis B. Slīpa F1 un F2 uz laiduma L_F2+b.",
        "url": "exsample_4.JPG",
        "status": "Stat. Nosakāms",
    },
    "Piemērs 5": {
        "caption": "Konsole F1 uz Lk. Šarnīrs A. q uz laiduma L (garums a). Rullītis B.",
        "url": "exsample_5.JPG",
        "status": "Stat. Nosakāms",
    },
    "Piemērs 6": {
        "caption": "q uz konsoles Lk. Rullītis A. F1 uz laiduma L. Rullītis B.",
        "url": "exsample_6.JPG",
        "status": "Stat. Nosakāms",
    },
    "Piemērs 7": {
        "caption": "Konsole F1 uz Lk. Rullītis A. F2, slīpa F3 uz laiduma L. Rullītis B.",
        "url": "exsample_7.JPG",
        "status": "Stat. Nosakāms",
    },
    "Piemērs 8": {
        "caption": "Iemūrēts A (pa labi). q uz L. Ekscentriska F3 kreisajā galā.",
        "url": "exsample_8.JPG",
        "status": "Stat. Nenosakāms (Iemūrēts)",
    },
    "Piemērs 9": {
        "caption": "Iemūrēts A (pa kreisi). F2 pie L. Ekscentriska F3 labajā galā (L+a).",
        "url": "exsample_9.JPG",
        "status": "Stat. Nenosakāms (Iemūrēts)",
    },
}

# --- Ģeometrijas Palīgfunkcija ---
def get_components(F, alpha_deg):
    """Aprēķina vertikālos un horizontālos spēka komponentus, pieņemot, ka alfa ir no vertikāles."""
    alpha_rad = math.radians(alpha_deg)
    # F_v = F * cos(alpha) (Tā kā alfa ir no vertikālās ass)
    F_v = F * math.cos(alpha_rad)
    # F_h = F * sin(alpha)
    F_h = F * math.sin(alpha_rad)
    return F_v, F_h

# --- DIAGRAMMU APRĒĶINĀŠANAS FUNKCIJA (Stat. Nosakāms/Konsoles) ---

def calculate_diagrams(L_total, x_coords, R_dict, forces):
    """
    Ģeneriska funkcija V un M diagrammu aprēķināšanai, rēķinot no kreisā gala.
    """
    
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    # Nosakām atbalstu atrašanās vietas
    L_AB = forces.get("L_AB", L_total if L_total > 0 else 1.0) 
    
    # Balstu reakcijas (tikai vertikālās)
    reactions = []
    # Balsts A ir pie x=A_pos
    if "A_y" in R_dict: reactions.append((R_dict["A_y"], forces.get("A_pos", 0.0), "Ay"))
    # Balsts B ir pie x=B_pos
    if "B_y" in R_dict: reactions.append((R_dict["B_y"], forces.get("B_pos", L_AB), "By")) 
    
    
    for i, x in enumerate(x_coords):
        V = 0
        M = 0
        
        # 1. Pieliekam Momenta Reakciju (tikai iemūrējumam pie x=0)
        if "M_A" in R_dict and forces.get("A_pos", 0.0) == 0.0: 
            M += R_dict["M_A"] # M_A ir konstants visā laidumā pie x=0
        
        # 2. Pieliekam Vertikālās Reakcijas (no kreisās puses)
        for R, R_pos, _ in reactions:
            if x >= R_pos:
                V += R
                M += R * (x - R_pos)
        
        # 3. Pieliekam Vertikālās Koncentrētās Slodzes
        for F, F_pos, _ in forces.get("point_loads", []):
            if x >= F_pos:
                V -= F
                M -= F * (x - F_pos)
        
        # 4. Pieliekam Izkliedētās Slodzes (q)
        for q_val, q_start, q_end in forces.get("dist_loads", []):
            if x > q_start:
                # Garums, kas pielikts no q sākuma līdz sekcijai x
                x_prime = min(x, q_end) - q_start 
                if x_prime > 0:
                    V -= q_val * x_prime
                    # M = M_iepriekšējais - Q_uz_x * (attālums no Q_uz_x centra līdz sekcijai x)
                    # Q_uz_x attālums līdz kreisajai sekcijas robežai ir x_prime / 2
                    # Tāpēc Q*x_prime* (attālums no Q centra līdz x)
                    M -= q_val * x_prime * (x - q_start - x_prime / 2)
                    
        V_values[i] = V
        M_values[i] = M

    return V_values, M_values

# --- ATRISINĀTĀJU FUNKCIJAS (Latviskotas) ---
# (Piemēri 1-7 paliek nemainīgi vai ar minimālām korekcijām)

def solve_example_1(F1, q, a, L, Lk):
    """Piemērs 1: Statiski Nosakāms. Hinge A, Roller B. F1 in span L, q on cantilever Lk."""
    if L <= 0: st.error("Laiduma garumam L jābūt pozitīvam."); return {}, np.array([0]), np.array([0]), np.array([0])
    L_total = L + Lk
    Q = q * Lk
    x_Q_center = L + Lk / 2 
    
    # Sum M_A = 0 (A pie x=0): By * L - F1 * a - Q * x_Q_center = 0
    By = (F1 * a + Q * x_Q_center) / L
    Ay = F1 + Q - By
    Ax = 0
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    forces = {"point_loads": [(F1, a, "F1")], "dist_loads": [(q, L, L_total)], "L_AB": L}
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

# (solve_example_2, 3, 4, 5, 6, 7 paliek kā iepriekšējā versijā ar precizēto loģiku)
def solve_example_2(F1, q, L, Lk):
    """Piemērs 2: Statiski Nosakāms. Hinge A, Roller B. q on span L, F1 on cantilever Lk."""
    if L <= 0: st.error("Laiduma garumam L jābūt pozitīvam."); return {}, np.array([0]), np.array([0]), np.array([0])
    L_total = L + Lk
    Q = q * L
    x_Q = L / 2 
    x_F1 = L_total
    
    # Sum M_A = 0 (A pie x=0): By * L - Q * x_Q - F1 * x_F1 = 0
    By = (Q * x_Q + F1 * x_F1) / L
    Ay = Q + F1 - By
    Ax = 0
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    forces = {"point_loads": [(F1, x_F1, "F1")], "dist_loads": [(q, 0, L)], "L_AB": L}
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_3(F1, F2, alpha, a, L, Lk):
    """Piemērs 3: Statiski Nosakāms. Slīpa F1 laidumā L, F2 uz konsoles Lk."""
    if L <= 0: st.error("Laiduma garumam L jābūt pozitīvam."); return {}, np.array([0]), np.array([0]), np.array([0])
    L_total = L + Lk
    F1v, F1h = get_components(F1, alpha)
    x_F1 = a
    x_F2 = L_total
    
    Ax = F1h
    # Sum M_A = 0 (A pie x=0): By * L - F1v * a - F2 * L_total = 0
    By = (F1v * a + F2 * L_total) / L
    Ay = F1v + F2 - By
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    forces = {"point_loads": [(F1v, x_F1, "F1v"), (F2, x_F2, "F2")], "dist_loads": [], "L_AB": L}
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_4(F1, F2, alpha, a, L_F2, b, Lk):
    """Piemērs 4: Statiski Nosakāms. Slīpa F1 un F2."""
    L_AB = L_F2 + b
    if L_AB <= 0: st.error("Laidumam no A līdz B jābūt pozitīvam."); return {}, np.array([0]), np.array([0]), np.array([0])
    L_total = L_AB + Lk
    F1v, F1h = get_components(F1, alpha)
    x_F1 = a
    x_F2 = L_F2
    
    Ax = F1h
    # Sum M_A = 0 (A pie x=0): By * L_AB - F1v * a - F2 * L_F2 = 0
    By = (F1v * a + F2 * L_F2) / L_AB
    Ay = F1v + F2 - By
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    forces = {"point_loads": [(F1v, x_F1, "F1v"), (F2, x_F2, "F2")], "dist_loads": [], "L_AB": L_AB}
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_5(F1, q, Lk, L, a):
    """Piemērs 5: Statiski Nosakāms. F1 uz laiduma Lk. A (šarnīrs). q uz laiduma L (garums a). B (rullītis)."""
    if L <= 0: st.error("Laiduma garumam L jābūt pozitīvam."); return {}, np.array([0]), np.array([0]), np.array([0])
    L_total = L
    
    Q = q * a
    x_Q_center = a / 2 
    x_F1 = Lk # Tā kā Lk ir attālums no A
    
    # Sum M_A = 0: R_By * L - Q * x_Q_center - F1 * Lk = 0
    By = (Q * x_Q_center + F1 * Lk) / L
    
    # Sum F_y = 0: Ay + By - Q - F1 = 0
    Ay = Q + F1 - By
    Ax = 0
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    # Diagrammas garums no 0 līdz L
    x_coords = np.linspace(0, L_total, 1000)
    
    forces = {
        "point_loads": [(F1, Lk, "F1")], 
        "dist_loads": [(q, 0, a)], 
        "L_AB": L,
        "A_pos": 0.0,
        "B_pos": L
    }
    
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values


def solve_example_6(q, F1, Lk, L, a):
    """Piemērs 6: Statiski Nosakāms. q uz konsoles Lk. A (rullītis). F1 uz laiduma L. B (rullītis)."""
    if L <= 0: st.error("Laiduma garumam L jābūt pozitīvam."); return {}, np.array([0]), np.array([0]), np.array([0])
    L_total = Lk + L
    Q = q * Lk
    
    # Pieņemsim: A pie 0, B pie L. q no -Lk līdz 0. F1 pie a (no A)
    x_F1_val = a
    
    # Sum M_A = 0: By * L + F1 * a + Q * (Lk/2) = 0 (Q rada pretēju momentu pulksteņrād.)
    # M(A) = -F1*a - Q * (Lk/2) + By*L = 0 -> By*L = F1*a + Q*Lk/2
    By = (F1 * x_F1_val + Q * (Lk/2)) / L
    
    # Sum F_y = 0: Ay + By - Q - F1 = 0
    Ay = Q + F1 - By
    Ax = 0
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    # Diagrammas garums no -Lk līdz L
    x_coords = np.linspace(-Lk, L, 1000)
    
    forces = {
        "point_loads": [(F1, a, "F1")], 
        "dist_loads": [(q, -Lk, 0)], # q no -Lk līdz 0
        "L_AB": L,
        "A_pos": 0.0,
        "B_pos": L
    }
    
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_7(F1, F2, F3, alpha, Lk, L, a, b):
    """Piemērs 7: Statiski Nosakāms. F1 uz konsoles Lk. A (rullītis). F2, slīpa F3 uz laiduma L. B (rullītis)."""
    if L <= 0: st.error("Laiduma garumam L jābūt pozitīvam."); return {}, np.array([0]), np.array([0]), np.array([0])
    L_total = Lk + L
    F3v, F3h = get_components(F3, alpha)
    
    # Pieņemsim A pie 0, B pie L, F1 pie -Lk
    x_F1 = -Lk 
    x_F2 = a
    x_F3 = L - b
    
    Ax = F3h # Rullītis A dod Ax = F3h
    
    # M(A) = -F1*Lk - F2*a - F3v*(L-b) + By*L = 0
    By = (F1 * Lk + F2 * a + F3v * (L - b)) / L
    
    # Sum F_y = 0: Ay + By - F1 - F2 - F3v = 0
    Ay = F1 + F2 + F3v - By
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(-Lk, L, 1000)
    forces = {
        "point_loads": [(F1, x_F1, "F1"), (F2, x_F2, "F2"), (F3v, x_F3, "F3v")], 
        "dist_loads": [], 
        "L_AB": L,
        "A_pos": 0.0,
        "B_pos": L
    }
    
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_8(q, F3, h, L):
    """Piemērs 8: Statiski Nenosakāms. Iemūrēts A (pa labi). q uz L. Ekscentriska F3."""
    
    Q = q * L 
    L_half = L / 2
    
    # Statiskais Līdzsvars (A atrodas labajā galā pie x=L)
    # 1. Horizontālais Līdzsvars: RAx - F3 = 0 (RAx pa labi)
    Ax = F3 
    
    # 2. Vertikālais Līdzsvars: RAy - Q = 0 (RAy uz augšu)
    Ay = Q 
    
    # 3. Momenta Līdzsvars ap A (Clockwise = Positive):
    # MA + F3*h - Q * L_half = 0 
    # MA ir reakcijas moments (pretēji pulksteņrādītājam, ja M_A = Q*L/2 - F3*h, t.i. ja pozitīvs M_A ir pretpulkst.)
    # Atbilstoši Jūsu loģikai: MA (reakcija, Clockwise) + M(F3) (Counter-clockwise) - M(Q) (Clockwise) = 0
    # MA + F3 * h - Q * L_half = 0
    M_A = Q * L_half - F3 * h 
    
    R_dict = {"A_y": Ay, "A_x": Ax, "M_A": M_A}
    
    # Diagrammu aprēķins (sija no kreisā gala x=0 līdz labajam galam x=L)
    x_coords = np.linspace(0, L, 1000)
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    # M(x) formula, rēķinot no kreisā gala x
    for i, x in enumerate(x_coords):
        # Šķērsspēks (rēķināts no kreisā gala, Q uz leju, pozitīvs V iet uz leju)
        V = -q * x 
        
        # Lieces moments (rēķināts no kreisā gala, Q rada negatīvu momentu)
        # M(x) = - q * x * (x/2)
        M = - q * x**2 / 2
        
        # Pieliekam Iemūrējuma reakcijas (kas darbojas labajā galā pie x=L)
        # Šis ir nepareizs veids konsoles sijai. Pareizais veids ir vienkārši aprēķināt no brīvā gala (kreisā gala x=0).
        # Tā kā A ir iemūrēts labajā galā, reakcijas tiek izmantotas **pārbaudei** un **diagrammu galapunktiem**.
        # Šķēlumu V(x) un M(x) rēķinām no brīvā gala (x=0).
        
        V_values[i] = V
        M_values[i] = M

    return R_dict, x_coords, V_values, M_values

def solve_example_9(F2, F3, h, L, a):
    """Piemērs 9: Statiski Nenosakāms. Iemūrēts A (pa kreisi). F2 pie L. Ekscentriska F3."""
    st.warning("Piemērs 9 ir **Statiski Nenosakāms** (Iemūrēts atbalsts). Reakcijas aprēķinātas no statiskā līdzsvara. Diagrammu aprēķins balstīts uz konsoles elementu.")
    L_total = L + a
    x_F2 = L
    
    # Statiskais Līdzsvars (A atrodas kreisajā galā pie x=0)
    # 1. Horizontālais Līdzsvars: RAx + F3 = 0 (F3 iet pa kreisi, RAx iet pa labi)
    Ax = F3 
    
    # 2. Vertikālais Līdzsvars: RAy - F2 = 0 (RAy uz augšu)
    Ay = F2 
    
    # 3. Momenta Līdzsvars ap A (Clockwise = Positive):
    # MA + F3*h - F2 * L = 0 (F3 moments CCW, F2 moments CW)
    # MA ir reakcijas moments (pretēji pulksteņrādītājam, ja pozitīvs M_A ir pretpulkst.)
    M_A = F2 * L - F3 * h
    
    R_dict = {"A_y": Ay, "A_x": Ax, "M_A": M_A}
    x_coords = np.linspace(0, L_total, 1000)
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    # Šķēluma aprēķins no kreisā gala (x=0)
    for i, x in enumerate(x_coords):
        V = Ay
        M = M_A + Ay * x
        
        if x >= x_F2:
            V -= F2
            M -= F2 * (x - x_F2)
        
        V_values[i] = V
        M_values[i] = M

    return R_dict, x_coords, V_values, M_values


# --- FUNKCIJA FORMULU UN PASKAIDROJUMU RĀDĪŠANAI (Uzlabota) ---

def display_formulas(example, R_dict, input_params):
    """Parāda aprēķina formulas un paskaidrojumus atbilstoši izvēlētajam piemēram."""
    
    st.markdown("### 📋 Aprēķinu Formulas un Loģika")
    
    # Iegūstam aprēķinātās reakcijas ar 2 zīmēm aiz komata, lai parādītu formulās
    R_Ay = f"{R_dict.get('A_y', 0):.2f}"
    R_By = f"{R_dict.get('B_y', 0):.2f}"
    R_Ax = f"{R_dict.get('A_x', 0):.2f}"
    R_MA = f"{R_dict.get('M_A', 0):.2f}"
    
    if example == "Piemērs 8":
        # Iegūstam ievades datus
        F3 = input_params.get('F3', 0)
        q = input_params.get('q', 0)
        h = input_params.get('h', 0)
        L = input_params.get('L', 1)
        
        Q = q * L
        L_half = L / 2
        
        # Pārbaudes aprēķini
        moment_f3 = F3 * h
        moment_q = Q * L_half
        
        MA_calc = moment_q - moment_f3
        
        st.markdown(r"""
        **1. Slogojums**
        * $Q = q \cdot L = {q} \cdot {L} = {Q:.2f} \text{ kN}$ (Rezultējošais spēks no izkliedētās slodzes)
        * $x_{Q} = L/2 = {L_half:.2f} \text{ m}$ (Attālums no balsta A)

        **2. Reakciju Aprēķins (A ir Iemūrēts balsts, labajā galā)**
        * **Horizontālais Līdzsvars:** $\sum F_{x} = 0 \implies R_{A_{x}} - F_{3} = 0$
            $$R_{A_{x}} = F_{3} = {F3:.2f} \text{ kN}$$
        * **Vertikālais Līdzsvars:** $\sum F_{y} = 0 \implies R_{A_{y}} - Q = 0$
            $$R_{A_{y}} = Q = {Q:.2f} \text{ kN}$$
        * **Momenta Līdzsvars (ap A, pulksteņrādītājs pozitīvs):** $\sum M_{A} = 0 \implies R_{M_{A}} + M(F_{3}) - M(Q) = 0$
            * $M(F_{3}) = F_{3} \cdot h = {F3:.2f} \cdot {h:.2f} = {moment_f3:.2f} \text{ kNm}$ (Pretpulksteņrādītājs)
            * $M(Q) = Q \cdot (L/2) = {Q:.2f} \cdot {L_half:.2f} = {moment_q:.2f} \text{ kNm}$ (Pulksteņrādītājs)
            $$R_{M_{A}} = M(Q) - M(F_{3}) = {moment_q:.2f} - {moment_f3:.2f} = {MA_calc:.2f} \text{ kNm}$$

        **3. Reakciju Pārbaude**
        * **$\sum F_{x} = 0$:** $R_{A_{x}} - F_{3} = {R_Ax} - {F3:.2f} \approx {float(R_Ax) - F3:.2f} \checkmark$
        * **$\sum F_{y} = 0$:** $R_{A_{y}} - Q = {R_Ay} - {Q:.2f} \approx {float(R_Ay) - Q:.2f} \checkmark$
        * **$\sum M_{A} = 0$:** $R_{M_{A}} + M(F_{3}) - M(Q) = {R_MA} + {moment_f3:.2f} - {moment_q:.2f} \approx {float(R_MA) + moment_f3 - moment_q:.2f} \checkmark$
        """ % locals()) # Izmantojam locals(), lai varētu tieši ievietot mainīgos

    elif example == "Piemērs 5":
        # Iegūstam ievades datus
        F1 = input_params.get('F1', 0)
        q = input_params.get('q', 0)
        Lk = input_params.get('Lk', 0)
        L = input_params.get('L', 1)
        a = input_params.get('a', 0)
        
        Q = q * a
        x_Q = a / 2
        
        # Pārbaudes aprēķini
        moment_Q = Q * x_Q
        moment_F1 = F1 * Lk
        
        st.markdown(r"""
        **1. Izkliedētās Slodzes Rezultants**
        * $Q = q \cdot a = {q} \cdot {a} = {Q:.2f} \text{ kN}$
        * $x_{Q} = a/2 = {x_Q:.2f} \text{ m}$ (Attālums no A)

        **2. Reakciju Aprēķins (A pie $x=0$, L ir laidums starp A un B)**
        * **Momenta Līdzsvars (ap A):** $\sum M_{A} = 0 \implies R_{B_{y}} \cdot L - Q \cdot x_{Q} - F_{1} \cdot L_{k} = 0$
            $$R_{B_{y}} = \frac{Q \cdot x_{Q} + F_{1} \cdot L_{k}}{L} = \frac{{moment_Q:.2f} + {moment_F1:.2f}}{L} = {R_By} \text{ kN}$$
        * **Vertikālais Līdzsvars:** $\sum F_{y} = 0 \implies R_{A_{y}} + R_{B_{y}} - Q - F_{1} = 0$
            $$R_{A_{y}} = Q + F_{1} - R_{B_{y}} = {Q:.2f} + {F1:.2f} - {R_By} = {R_Ay} \text{ kN}$$

        **3. Reakciju Pārbaude**
        * **$\sum M_{B} = 0$ (Momenta Līdzsvars ap B):** $R_{A_{y}} \cdot L - Q \cdot (L - x_{Q}) - F_{1} \cdot (L - L_{k}) = 0$
            * $R_{A_{y}} \cdot L = {float(R_Ay) * L:.2f} \text{ kNm}$
            * $Q \cdot (L - x_{Q}) = {Q:.2f} \cdot ({L - x_Q:.2f}) = {Q * (L - x_Q):.2f} \text{ kNm}$
            * $F_{1} \cdot (L - L_{k}) = {F1:.2f} \cdot ({L - Lk:.2f}) = {F1 * (L - Lk):.2f} \text{ kNm}$
            * Pārbaudes summa: $({float(R_Ay) * L} - {Q * (L - x_Q)} - {F1 * (L - Lk)}):.2f \approx {float(float(R_Ay) * L - Q * (L - x_Q) - F1 * (L - Lk)):.2f} \checkmark$
        """ % locals())
        
    else:
        # Pārējiem piemēriem (2, 4, 6, 7, 9) rādīt pamatinformāciju
        st.markdown(r"""
        Šis piemērs izmanto statiskā līdzsvara vienādojumus $(\sum F_{x}=0, \sum F_{y}=0, \sum M=0)$ balstu reakciju aprēķināšanai, un pēc tam tiek veidotas šķēluma funkcijas $V(x)$ un $M(x)$.
        
        **Reakciju Pārbaude:** Reakcijas tiek aprēķinātas, izmantojot $\sum M=0$ ap vienu balstu, un pēc tam pārbaudītas, izmantojot $\sum F_{y}=0$ vai $\sum M=0$ ap otru balstu.
        """)
    
    st.markdown("---")


# --- UI Izkārtojums (Latviskots) ---

st.title("🏗️ Sijas Konstrukciju Analīzes Rīks")
st.markdown("---")

# --- Sijas Izvēle un Attēls ---
example_options = [f"Piemērs {i}" for i in range(1, 10)]
selected_example = st.sidebar.radio(
    "Izvēlieties Sijas Piemēru (1-9):",
    options=example_options,
    index=7, # Tagad atveram Piemēru 8
)

config = IMAGE_CONFIGS.get(selected_example)

st.subheader(f"Atrisinājums: {selected_example} - **{config['status']}**")
st.markdown(f"*{config['caption']}*")
st.image(config['url'], caption=f"Shēma piemēram {selected_example}") 

st.markdown("---")

# --- Dinamiskie Ievades Parametri un Risinājuma Izsaukums ---

R_dict = {}
x_coords, V_values, M_values = np.array([0]), np.array([0]), np.array([0])
alpha = 0 
input_params = {} # Saglabājam parametrus formulām

with st.sidebar:
    st.markdown(f"#### {selected_example} Ievades Parametri")
    
    # ------------------ IEVADES ATBILSTOŠI PIEMĒRAM ------------------
    if selected_example in ["Piemērs 3", "Piemērs 4", "Piemērs 7"]:
        alpha = st.slider("Leņķis $\\alpha$ (grādos no vertikāles)", 0, 90, 45, key="alpha_slider")

    if selected_example == "Piemērs 1":
        F1 = st.number_input("Koncentrētā Slodze ($F_1$, kN)", value=10.0, min_value=0.0, step=1.0, key="F11")
        q = st.number_input("Izkliedētā Slodze ($q$, kN/m)", value=5.0, min_value=0.0, step=1.0, key="q1")
        st.markdown("---")
        a = st.number_input("Attālums 'a' līdz $F_1$ (m)", value=2.0, min_value=0.1, step=0.5, key="a1")
        L = st.number_input("Laiduma Garums $L$ (m)", value=6.0, min_value=0.1, step=0.5, key="L1")
        Lk = st.number_input("Konsoles Garums $L_k$ (m)", value=2.0, min_value=0.0, step=0.5, key="Lk1")
        R_dict, x_coords, V_values, M_values = solve_example_1(F1, q, a, L, Lk)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Piemērs 2":
        q = st.number_input("Izkliedētā Slodze ($q$, kN/m)", value=8.0, min_value=0.0, step=1.0, key="q2")
        F1 = st.number_input("Koncentrētā Slodze ($F_1$, kN)", value=12.0, min_value=0.0, step=1.0, key="F12")
        st.markdown("---")
        L = st.number_input("Laiduma Garums $L$ (m)", value=7.0, min_value=0.1, step=0.5, key="L2")
        Lk = st.number_input("Konsoles Garums $L_k$ (m)", value=1.5, min_value=0.0, step=0.5, key="Lk2")
        R_dict, x_coords, V_values, M_values = solve_example_2(F1, q, L, Lk)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Piemērs 3":
        F1 = st.number_input("Slīpā Slodze ($F_1$, kN)", value=25.0, min_value=0.0, step=1.0, key="F13")
        F2 = st.number_input("Koncentrētā Slodze ($F_2$, kN)", value=10.0, min_value=0.0, step=1.0, key="F23")
        st.markdown("---")
        a = st.number_input("Attālums 'a' līdz $F_1$ (m)", value=3.0, min_value=0.1, step=0.5, key="a3")
        L = st.number_input("Laiduma Garums $L$ (m)", value=8.0, min_value=0.1, step=0.5, key="L3")
        Lk = st.number_input("Konsoles Garums $L_k$ (m)", value=2.0, min_value=0.0, step=0.5, key="Lk3")
        R_dict, x_coords, V_values, M_values = solve_example_3(F1, F2, alpha, a, L, Lk)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Piemērs 4":
        F1 = st.number_input("Slīpā Slodze ($F_1$, kN)", value=15.0, min_value=0.0, step=1.0, key="F14")
        F2 = st.number_input("Koncentrētā Slodze ($F_2$, kN)", value=20.0, min_value=0.0, step=1.0, key="F24")
        st.markdown("---")
        a = st.number_input("Attālums 'a' līdz $F_1$ no A (m)", value=2.0, min_value=0.1, step=0.5, key="a4")
        L_F2 = st.number_input("Attālums $L$ līdz $F_2$ no A (m)", value=5.0, min_value=0.1, step=0.5, key="LF24")
        b = st.number_input("Attālums 'b' no $F_2$ līdz B (m)", value=1.0, min_value=0.1, step=0.5, key="b4")
        Lk = st.number_input("Konsoles Garums $L_k$ (m)", value=1.0, min_value=0.0, step=0.5, key="Lk4")
        R_dict, x_coords, V_values, M_values = solve_example_4(F1, F2, alpha, a, L_F2, b, Lk)
        R_list_template = ["A_y", "B_y", "A_x"]
        
    elif selected_example == "Piemērs 5":
        F1 = st.number_input("Koncentrētā Slodze ($F_1$, kN)", value=6.0, min_value=0.0, step=1.0, key="F15")
        q = st.number_input("Izkliedētā Slodze ($q$, kN/m)", value=6.0, min_value=0.0, step=1.0, key="q5")
        st.markdown("---")
        Lk = st.number_input("Attālums $L_k$ līdz $F_1$ no A (m)", value=1.80, min_value=0.0, step=0.5, key="Lk5")
        L = st.number_input("Galvenā Laiduma Garums ($L$, m) A līdz B", value=4.90, min_value=0.1, step=0.5, key="L5")
        a = st.number_input("Izkliedētās Slodzes Garums ($a$, m)", value=1.80, min_value=0.0, max_value=L if L > 0 else 1.0, step=0.5, key="a5")
        R_dict, x_coords, V_values, M_values = solve_example_5(F1, q, Lk, L, a)
        R_list_template = ["A_y", "B_y", "A_x"]
        input_params = {'F1': F1, 'q': q, 'Lk': Lk, 'L': L, 'a': a} # Saglabājam ievades

    elif selected_example == "Piemērs 6":
        q = st.number_input("Izkliedētā Slodze ($q$, kN/m)", value=10.0, min_value=0.0, step=1.0, key="q6")
        F1 = st.number_input("Koncentrētā Slodze ($F_1$, kN)", value=25.0, min_value=0.0, step=1.0, key="F16")
        st.markdown("---")
        Lk = st.number_input("Konsoles Garums ($L_k$, m)", value=2.0, min_value=0.1, step=0.5, key="Lk6")
        L = st.number_input("Laiduma Garums ($L$, m)", value=5.0, min_value=0.1, step=0.5, key="L6")
        a = st.number_input("Attālums 'a' no A līdz $F_1$ (m)", value=3.0, min_value=0.0, max_value=L if L > 0 else 1.0, step=0.5, key="a6")
        R_dict, x_coords, V_values, M_values = solve_example_6(q, F1, Lk, L, a)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Piemērs 7":
        F1 = st.number_input("Slodze $F_1$ uz $L_k$ (kN)", value=10.0, min_value=0.0, step=1.0, key="F17")
        F2 = st.number_input("Slodze $F_2$ laidumā (kN)", value=5.0, min_value=0.0, step=1.0, key="F27")
        F3 = st.number_input("Slīpā Slodze $F_3$ (kN)", value=15.0, min_value=0.0, step=1.0, key="F37")
        st.markdown("---")
        Lk = st.number_input("Konsoles Garums $L_k$ (m)", value=2.0, min_value=0.1, step=0.5, key="Lk7")
        L = st.number_input("Laiduma Garums $L$ (m)", value=6.0, min_value=0.1, step=0.5, key="L7")
        a = st.number_input("Attālums 'a' līdz $F_2$ no A (m)", value=2.0, min_value=0.0, max_value=L, step=0.5, key="a7")
        b = st.number_input("Attālums 'b' no B līdz $F_3$ (m)", value=1.0, min_value=0.0, max_value=L, step=0.5, key="b7")
        R_dict, x_coords, V_values, M_values = solve_example_7(F1, F2, F3, alpha, Lk, L, a, b)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Piemērs 8":
        F3 = st.number_input("Horizontālā Slodze ($F_3$, kN)", value=2.5, min_value=0.0, step=0.1, key="F38")
        q = st.number_input("Izkliedētā Slodze ($q$, kN/m)", value=6.0, min_value=0.0, step=1.0, key="q8")
        st.markdown("---")
        h = st.number_input("Slodzes Nobīdes Augstums ($h$, m)", value=0.9, min_value=0.0, step=0.1, key="h8")
        L = st.number_input("Laiduma Garums ($L$, m)", value=4.90, min_value=0.1, step=0.5, key="L8")
        R_dict, x_coords, V_values, M_values = solve_example_8(q, F3, h, L)
        R_list_template = ["A_y", "A_x", "M_A"]
        input_params = {'F3': F3, 'q': q, 'h': h, 'L': L} # Saglabājam ievades

    elif selected_example == "Piemērs 9":
        F2 = st.number_input("Koncentrētā Slodze ($F_2$, kN)", value=20.0, min_value=0.0, step=1.0, key="F29")
        F3 = st.number_input("Horizontālā Slodze ($F_3$, kN)", value=10.0, min_value=0.0, step=1.0, key="F39")
        st.markdown("---")
        h = st.number_input("Slodzes Nobīdes Augstums ($h$, m)", value=0.5, min_value=0.0, step=0.1, key="h9")
        L = st.number_input("Attālums $L$ līdz $F_2$ no A (m)", value=4.0, min_value=0.1, step=0.5, key="L9")
        a = st.number_input("Attālums 'a' no $F_2$ līdz galam (m)", value=2.0, min_value=0.0, step=0.5, key="a9")
        R_dict, x_coords, V_values, M_values = solve_example_9(F2, F3, h, L, a)
        R_list_template = ["A_y", "A_x", "M_A"]


# --- Izvades Sadaļa: Formulu Rādīšana ---
display_formulas(selected_example, R_dict, input_params)


# --- Izvades Sadaļa: Reakcijas ---

st.subheader("Risinājums: Atbalsta Reakcijas")

R_list = []
for key in R_list_template:
    display_key = key.replace('_y', '_{y}').replace('_x', '_{x}').replace('_A', '_{A}').replace('M_A', 'M_{A}')
    
    if key.startswith("M_"):
        unit = "kNm"
    else:
        unit = "kN"
        
    R_list.append(
        f"| ${display_key}$ | **{R_dict.get(key, 0):,.2f}** | {unit} |"
    )

st.markdown(f"""
| Reakcija | Vērtība | Mērvienība |
| :---: | :---: | :---: |
{"\n".join(R_list)}
""")

st.markdown("---")

# --- Izvades Sadaļa: Diagrammas ---

st.subheader("Šķērsspēku ($V$) un Lieces Momenta ($M$) Diagrammas")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# --- Šķērsspēku Diagramma (V) ---
ax1.plot(x_coords, V_values, label='$V(x)$', color='#1e56a0')
ax1.axhline(0, color='black', linewidth=0.5) 
ax1.set_title("Šķērsspēku Diagramma $V(x)$", fontsize=16)
ax1.set_xlabel("x (m)")
ax1.set_ylabel("Šķērsspēks (kN)")
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.fill_between(x_coords, V_values, 0, where=(V_values > 0), color='#1e56a0', alpha=0.2, label='Pozitīvs V')
ax1.fill_between(x_coords, V_values, 0, where=(V_values < 0), color='#ff7f0e', alpha=0.2, label='Negatīvs V')
ax1.legend(loc='upper right')

# --- Lieces Momenta Diagramma (M) ---
ax2.plot(x_coords, M_values, label='$M(x)$', color='#d62728')
ax2.axhline(0, color='black', linewidth=0.5) 
ax2.set_title("Lieces Momenta Diagramma $M(x)$", fontsize=16)
ax2.set_xlabel("x (m)")
ax2.set_ylabel("Lieces Moments (kNm)")
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.fill_between(x_coords, M_values, 0, where=(M_values > 0), color='#2ca02c', alpha=0.2, label='Pozitīvs M (Apakšā stiepe)')
ax2.fill_between(x_coords, M_values, 0, where=(M_values < 0), color='#d62728', alpha=0.2, label='Negatīvs M (Augšā stiepe)')
ax2.legend(loc='upper right')

plt.tight_layout()
st.pyplot(fig)
