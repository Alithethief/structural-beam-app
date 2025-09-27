import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Konfigurācija Streamlit lapai ---
st.set_page_config(layout="wide", page_title="Siju Analīzes Kalkulators (9 Piemēri)")

# --- Konstantes un Attēlu Konfigurācija ---
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

# --- DIAGRAMMU APRĒĶINĀŠANAS FUNKCIJA (Stat. Nosakāmām sijām) ---

def calculate_diagrams(L_total, x_coords, R_dict, forces):
    """Ģeneriska funkcija V un M diagrammu aprēķināšanai statiski nosakāmām sijām."""
    
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    # Reakcijas tiek pieliktas to koordinātās
    reactions = []
    if "A_y" in R_dict: reactions.append((R_dict["A_y"], 0.0, "Ay"))
    # Dažos piemēros B ir L, dažos L_AB
    L_B = forces.get("L_AB", L_total if L_total > 0 else 1.0) 
    if "B_y" in R_dict: reactions.append((R_dict["B_y"], L_B, "By")) 
    if "M_A" in R_dict: 
        # M_A ir jāpieskaita pie x=0
        pass 
    
    for i, x in enumerate(x_coords):
        V = 0
        M = 0
        
        # 1. Pieliekam Momenta Reakciju (ja ir iemūrēts)
        if "M_A" in R_dict: 
            M += R_dict["M_A"] # Tiek pielikts visā laidumā
        
        # 2. Pieliekam Reakcijas (no kreisās puses)
        for R, R_pos, R_name in reactions:
            if R_name != "MA": # Ignorējam momentu V aprēķinam
                if x >= R_pos:
                    V += R
                    M += R * (x - R_pos)
        
        # 3. Pieliekam Vertikālās Koncentrētās Slodzes
        for F, F_pos, F_type in forces.get("point_loads", []):
            if x >= F_pos:
                V -= F
                M -= F * (x - F_pos)
        
        # 4. Pieliekam Izkliedētās Slodzes (q)
        for q, q_start, q_end in forces.get("dist_loads", []):
            if x > q_start:
                # Garums, kas pielikts no q sākuma līdz sekcijai x
                x_prime = min(x, q_end) - q_start 
                if x_prime > 0:
                    V -= q * x_prime
                    # M = M_iepriekšējais - Q * (attālums no Q centra līdz sekcijai x)
                    M -= q * x_prime * (x - q_start - x_prime / 2)
                    
        V_values[i] = V
        M_values[i] = M

    return V_values, M_values

# --- ATRISINĀTĀJU FUNKCIJAS (Latviskotas) ---

def solve_example_1(F1, q, a, L, Lk):
    """Piemērs 1: Statiski Nosakāms. Hinge A, Roller B. F1 in span L, q on cantilever Lk."""
    if L <= 0: st.error("Laiduma garumam L jābūt pozitīvam."); return {}, np.array([0]), np.array([0]), np.array([0])
    L_total = L + Lk
    Q = q * Lk
    x_Q = L + Lk / 2 
    
    By = (F1 * a + Q * x_Q) / L
    Ay = F1 + Q - By
    Ax = 0
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    forces = {"point_loads": [(F1, a, "F1")], "dist_loads": [(q, L, L_total)]}
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_2(F1, q, L, Lk):
    """Piemērs 2: Statiski Nosakāms. Hinge A, Roller B. q on span L, F1 on cantilever Lk."""
    if L <= 0: st.error("Laiduma garumam L jābūt pozitīvam."); return {}, np.array([0]), np.array([0]), np.array([0])
    L_total = L + Lk
    Q = q * L
    x_Q = L / 2 
    x_F1 = L_total
    
    By = (Q * x_Q + F1 * x_F1) / L
    Ay = Q + F1 - By
    Ax = 0
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    forces = {"point_loads": [(F1, x_F1, "F1")], "dist_loads": [(q, 0, L)]}
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
    By = (F1v * a + F2 * x_F2) / L
    Ay = F1v + F2 - By
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    forces = {"point_loads": [(F1v, x_F1, "F1v"), (F2, x_F2, "F2")], "dist_loads": []}
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
    By = (F1v * a + F2 * L_F2) / L_AB
    Ay = F1v + F2 - By
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    forces = {"point_loads": [(F1v, x_F1, "F1v"), (F2, x_F2, "F2")], "dist_loads": [], "L_AB": L_AB}
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_5(F1, q, Lk, L, a):
    """Piemērs 5: Statiski Nosakāms. F1 uz konsoles Lk. A (šarnīrs). q uz laiduma L (garums a). B (rullītis)."""
    if L <= 0: st.error("Laiduma garumam L jābūt pozitīvam."); return {}, np.array([0]), np.array([0]), np.array([0])
    L_total = Lk + L
    Q = q * a
    
    # Sum M_A = 0 (A pie Lk): By * L + F1 * Lk - Q * (Lk + a / 2) = 0
    # Pēc shēmas F1 un q rotē vienā virzienā ap A (PV), By rotē pretēji (P)
    # F1 ir uz konsoles kreisajā pusē pie x=0
    # A ir pie x=Lk
    
    # Mainīsim koordinātu sākumu uz A (x=Lk) ērtai aprēķināšanai:
    # F1 ir pie -Lk, B ir pie 0 (vecajā sistēmā bija pie L)
    # Labāk palikt pie x=0 kreisajā galā (F1)
    
    # Sum M_A (A pie Lk) = 0: By * L + F1 * Lk - Q * (Lk + a / 2) = 0
    # Pārbaudīsim reakcijas virzienus. F1 un Q spiež uz leju.
    # F1 rada momentu ap A (Pulksteņrād.) -> M = F1 * Lk
    # Q rada momentu ap A (Pret pulkst.) -> M = Q * (a/2 + Lk)
    # By rada momentu ap A (Pulksteņrād.) -> M = By * L
    
    # Ļausim sijas kreisajam galam būt x=0, A pie x=Lk, B pie x=Lk+L
    x_A = Lk
    x_B = Lk + L
    x_Q = Lk + a / 2 # Q centra attālums no A (ja A ir pie Lk, tad a/2)
    
    # Sum M_A = 0 (A pie Lk): By * L - F1 * Lk - Q * (x_Q - Lk) = 0.
    # Q centrs ir pie Lk + a/2. Attālums no A ir a/2.
    
    # Pieņemsim, ka A ir pie x=0 (kā visur citur) un F1 ir kreisajā konsolē.
    # Lk ir konsoles garums. L ir laidums starp A un B.
    # F1 (pie -Lk), A (pie 0), B (pie L), q (no 0 līdz a)
    
    # Pēc shēmas: F1 ir pie -Lk. A ir pie 0. B ir pie L. q ir no 0 līdz a.
    
    # Pārrēķins pie A (x=0)
    Q = q * a
    x_Q_center = a / 2 
    x_F1 = -Lk # Tā kā rēķinām no kreisā gala, F1 ir pie -Lk
    
    # Sum M_A = 0: R_By * L - Q * x_Q_center + F1 * Lk = 0
    By = (Q * x_Q_center - F1 * Lk) / L
    
    # Sum F_y = 0: Ay + By - Q - F1 = 0
    Ay = Q + F1 - By
    Ax = 0
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    # Diagrammas garums no -Lk līdz L
    x_coords = np.linspace(-Lk, L, 1000)
    
    # Pārveidojam spēku koordinātes uz jauno sistēmu, kur sākums ir -Lk
    # point_loads: (F, F_pos, F_type)
    # dist_loads: (q, q_start, q_end)
    forces = {
        "point_loads": [(F1, -Lk, "F1")], 
        "dist_loads": [(q, 0, a)], 
        "L_AB": L
    }
    
    # Pielāgojam aprēķinu funkciju, lai atbalsti būtu pie 0 un L
    def calculate_diagrams_ex5(L_total, x_coords, R_dict, forces):
        V_values = np.zeros_like(x_coords)
        M_values = np.zeros_like(x_coords)
        
        # Atbalsti A pie 0, B pie L (no Lk)
        reactions = [(R_dict["A_y"], 0.0, "Ay"), (R_dict["B_y"], L, "By")] 
        
        for i, x in enumerate(x_coords):
            V = 0
            M = 0
            
            # 1. Pieliekam Reakcijas (no kreisās puses)
            for R, R_pos, R_name in reactions:
                if x >= R_pos:
                    V += R
                    M += R * (x - R_pos)
            
            # 2. Pieliekam Koncentrētās Slodzes
            for F, F_pos, F_type in forces.get("point_loads", []):
                if x >= F_pos:
                    V -= F
                    M -= F * (x - F_pos)
            
            # 3. Pieliekam Izkliedētās Slodzes (q)
            for q, q_start, q_end in forces.get("dist_loads", []):
                if x > q_start:
                    x_prime = min(x, q_end) - q_start 
                    if x_prime > 0:
                        V -= q * x_prime
                        M -= q * x_prime * (x - q_start - x_prime / 2)
                        
            V_values[i] = V
            M_values[i] = M
        return V_values, M_values

    V_values, M_values = calculate_diagrams_ex5(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values


def solve_example_6(q, F1, Lk, L, a):
    """Piemērs 6: Statiski Nosakāms. q uz konsoles Lk. A (rullītis). F1 uz laiduma L. B (rullītis)."""
    if L <= 0: st.error("Laiduma garumam L jābūt pozitīvam."); return {}, np.array([0]), np.array([0]), np.array([0])
    L_total = Lk + L
    Q = q * Lk
    x_Q = Lk / 2
    x_F1 = Lk + a
    
    # Atbalsti: A pie Lk, B pie Lk+L. Pārbaudām, vai attēls ir pareizi interpretēts.
    # Attēlā: A pie 0, B pie L, q uz kreisās konsoles -Lk. F1 laidumā L.
    
    # Pieņemsim: A pie 0, B pie L, q no -Lk līdz 0, F1 pie a (no A)
    # L ir laidums starp A un B
    
    # Atbilstoši shēmai (A pie 0, B pie L, q uz konsoles -Lk):
    x_F1_val = a
    x_Q_center = Lk / 2 
    
    # Sum M_A = 0: By * L + Q * x_Q_center - F1 * x_F1_val = 0
    By = (F1 * x_F1_val - Q * x_Q_center) / L
    
    # Sum F_y = 0: Ay + By - Q - F1 = 0
    Ay = Q + F1 - By
    Ax = 0
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    # Diagrammas garums no -Lk līdz L
    x_coords = np.linspace(-Lk, L, 1000)
    
    # Spēki un to koordinātes:
    forces = {
        "point_loads": [(F1, a, "F1")], 
        "dist_loads": [(q, -Lk, 0)], 
        "L_AB": L
    }
    
    # Izmantojam modificētu aprēķinu, kur atbalsti ir pie 0 un L, sākums pie -Lk
    def calculate_diagrams_ex6(L_total, x_coords, R_dict, forces):
        V_values = np.zeros_like(x_coords)
        M_values = np.zeros_like(x_coords)
        L = forces["L_AB"] # Laidums L
        reactions = [(R_dict["A_y"], 0.0, "Ay"), (R_dict["B_y"], L, "By")] 
        
        for i, x in enumerate(x_coords):
            V = 0
            M = 0
            
            # 1. Pieliekam Reakcijas (no kreisās puses)
            for R, R_pos, R_name in reactions:
                if x >= R_pos:
                    V += R
                    M += R * (x - R_pos)
            
            # 2. Pieliekam Koncentrētās Slodzes
            for F, F_pos, F_type in forces.get("point_loads", []):
                if x >= F_pos:
                    V -= F
                    M -= F * (x - F_pos)
            
            # 3. Pieliekam Izkliedētās Slodzes (q)
            for q, q_start, q_end in forces.get("dist_loads", []):
                if x > q_start:
                    x_prime = min(x, q_end) - q_start 
                    if x_prime > 0:
                        V -= q * x_prime
                        M -= q * x_prime * (x - q_start - x_prime / 2)
                        
            V_values[i] = V
            M_values[i] = M
        return V_values, M_values

    V_values, M_values = calculate_diagrams_ex6(L_total, x_coords, R_dict, forces)
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
    
    Ax = F3h
    # Sum M_A = 0: By * L + F1 * Lk - F2 * a - F3v * (L - b) = 0
    By = (F2 * a + F3v * (L - b) - F1 * Lk) / L
    
    # Sum F_y = 0: Ay + By - F1 - F2 - F3v = 0
    Ay = F1 + F2 + F3v - By
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(-Lk, L, 1000)
    forces = {"point_loads": [(F1, x_F1, "F1"), (F2, x_F2, "F2"), (F3v, x_F3, "F3v")], "dist_loads": [], "L_AB": L}
    
    # Atkal izmantojam modificētu aprēķinu ar A pie 0, B pie L
    V_values, M_values = solve_example_6(0, 0, 0, L, 0)[2:4] # Ņemam V un M (nulles vērtības)
    
    def calculate_diagrams_ex7(L_total, x_coords, R_dict, forces):
        V_values = np.zeros_like(x_coords)
        M_values = np.zeros_like(x_coords)
        L = forces["L_AB"]
        reactions = [(R_dict["A_y"], 0.0, "Ay"), (R_dict["B_y"], L, "By")] 
        
        for i, x in enumerate(x_coords):
            V = 0
            M = 0
            
            # 1. Pieliekam Reakcijas (no kreisās puses)
            for R, R_pos, R_name in reactions:
                if x >= R_pos:
                    V += R
                    M += R * (x - R_pos)
            
            # 2. Pieliekam Koncentrētās Slodzes
            for F, F_pos, F_type in forces.get("point_loads", []):
                if x >= F_pos:
                    V -= F
                    M -= F * (x - F_pos)
            
            V_values[i] = V
            M_values[i] = M
        return V_values, M_values
    
    V_values, M_values = calculate_diagrams_ex7(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_8(q, F3, h, L):
    """Piemērs 8: Statiski Nenosakāms. Iemūrēts A (pa labi). q uz L. Ekscentriska F3."""
    st.warning("Piemērs 8 ir **Statiski Nenosakāms** (Iemūrēts atbalsts). Reakcijas aprēķinātas no statiskā līdzsvara. Diagrammu aprēķins balstīts uz konsoles elementu.")
    
    Q = q * L 
    # Atbalsts A atrodas pie x=L (labais gals).
    Ax = F3 
    Ay = Q 
    
    # Momenta reakcija MA (Pieņemam, ka M_A ir pretēji pulksteņrādītājam (P))
    # Sum M_A = 0: R_M_A - F3 * h + Q * (L / 2) = 0
    M_A = F3 * h - Q * (L / 2)
    
    R_dict = {"A_y": Ay, "A_x": Ax, "M_A": M_A}
    
    x_coords = np.linspace(0, L, 1000)
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    # Šķēluma aprēķins no kreisā gala (x=0)
    for i, x in enumerate(x_coords):
        # Spēki: q no 0 līdz L (uz leju). Nulle.
        V = -q * x 
        # Moments: M = -q * x * (x / 2)
        M = -q * x**2 / 2
        
        # Pieliekam reakcijas no A (x=L) (Jāizmanto tikai, ja rēķinām no A virzienā uz x=0)
        # Ja rēķinām no kreisā gala (x=0): M(x) = M(x)_slodzes + M(x)_reakcijas
        # Šajā gadījumā iemūrējums A ir pie L. Tātad rēķināts kā vienkāršs konsoles gals.
        
        # Ja rēķinām no kreisā gala (x=0):
        # Šķērsspēks: V(x) = -q*x
        # Moments: M(x) = -q*x^2/2
        
        # Šis risinājums ignorē iemūrējuma A deformācijas komponenti un rēķina vienkāršu konsoli 
        # ar $R_{A_y}$ un $R_{M_A}$ pie L. Pārrakstīsim, ņemot vērā, ka šķēlumu ņemam no x=0.
        
        # M(x) formula, ja rēķina no labā gala A (pie L):
        # M(x) = -R_M_A + R_A_y * x' - q * x'^2 / 2, kur x' = L - x
        x_prime = L - x 
        V = -Ay + q * x_prime # V(x') no labās puses (+ uz augšu, - uz leju)
        M = -M_A + Ay * x_prime - q * x_prime**2 / 2 
        
        V_values[i] = V
        M_values[i] = M

    return R_dict, x_coords, V_values, M_values

def solve_example_9(F2, F3, h, L, a):
    """Piemērs 9: Statiski Nenosakāms. Iemūrēts A (pa kreisi). F2 pie L. Ekscentriska F3."""
    st.warning("Piemērs 9 ir **Statiski Nenosakāms** (Iemūrēts atbalsts). Reakcijas aprēķinātas no statiskā līdzsvara. Diagrammu aprēķins balstīts uz konsoles elementu.")
    L_total = L + a
    x_F2 = L
    
    Ax = F3 
    Ay = F2 
    
    # Momenta reakcija MA (Pieņemam, ka M_A ir pretēji pulksteņrādītājam (P))
    # Sum M_A = 0: R_M_A - F2 * L + F3 * h = 0
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


# --- FUNKCIJA FORMULU UN PASKAIDROJUMU RĀDĪŠANAI ---

def display_formulas(example, R_dict):
    """Parāda aprēķina formulas un paskaidrojumus atbilstoši izvēlētajam piemēram."""
    
    st.markdown("### 📋 Aprēķinu Formulas un Loģika")
    
    # Iegūstam aprēķinātās reakcijas ar 2 zīmēm aiz komata, lai parādītu formulās
    R_Ay = f"{R_dict.get('A_y', 0):.2f}"
    R_By = f"{R_dict.get('B_y', 0):.2f}"
    R_Ax = f"{R_dict.get('A_x', 0):.2f}"
    R_MA = f"{R_dict.get('M_A', 0):.2f}"
    
    if example == "Piemērs 1":
        st.markdown(f"""
        **1. Reakciju Aprēķins (A pie $x=0$)**
        * $Q = q \cdot L_{k}$ (Kopējā slodze uz konsoles)
        * **Momenta Līdzsvars:** $\sum M_{A} = 0 \implies R_{B_{y}} \cdot L - F_{1} \cdot a - Q \cdot (L + L_{k}/2) = 0$
            $$R_{B_{y}} = \\frac{{F_{1} \cdot a + Q \cdot (L + L_{k}/2)}}{{L}} = {R_By} \\text{{ kN}}$$
        * **Vertikālo Spēku Līdzsvars:** $\sum F_{y} = 0 \implies R_{A_{y}} + R_{B_{y}} - F_{1} - Q = 0$
            $$R_{A_{y}} = F_{1} + Q - R_{B_{y}} = {R_Ay} \\text{{ kN}}$$

        **2. Iekšējo Spēku Vienādojumi $V(x)$ un $M(x)$**
        * **Sekcija I ($0 \le x < a$):**
            $$V(x) = R_{A_{y}}$$
            $$M(x) = R_{A_{y}} \cdot x$$
        * **Sekcija II ($a \le x < L$):**
            $$V(x) = R_{A_{y}} - F_{1}$$
            $$M(x) = R_{A_{y}} \cdot x - F_{1} \cdot (x-a)$$
        * **Sekcija III ($L \le x \le L_{kop}$):**
            $$V(x) = R_{A_{y}} - F_{1} + R_{B_{y}} - q \cdot (x-L)$$
            $$M(x) = R_{A_{y}} \cdot x - F_{1} \cdot (x-a) + R_{B_{y}} \cdot (x-L) - \\frac{{q \cdot (x-L)^2}}{{2}}$$
        """)
        
    elif example == "Piemērs 3":
        st.markdown(f"""
        **1. Komponentes**
        * $F_{1v} = F_{1} \cdot \\cos(\\alpha)$; $F_{1h} = F_{1} \cdot \\sin(\\alpha)$ (Vertikālais leņķis)

        **2. Reakciju Aprēķins (A pie $x=0$)**
        * **Horizontālais Līdzsvars:** $\sum F_{x} = 0 \implies R_{A_{x}} + F_{1h} = 0$
            $$R_{A_{x}} = -F_{1h} = {R_Ax} \\text{{ kN}}$$
        * **Momenta Līdzsvars:** $\sum M_{A} = 0 \implies R_{B_{y}} \cdot L - F_{1v} \cdot a - F_{2} \cdot (L+L_{k}) = 0$
            $$R_{B_{y}} = \\frac{{F_{1v} \cdot a + F_{2} \cdot (L+L_{k})}}{{L}} = {R_By} \\text{{ kN}}$$
        * **Vertikālais Līdzsvars:** $\sum F_{y} = 0 \implies R_{A_{y}} + R_{B_{y}} - F_{1v} - F_{2} = 0$
            $$R_{A_{y}} = F_{1v} + F_{2} - R_{B_{y}} = {R_Ay} \\text{{ kN}}$$

        **3. Iekšējo Spēku Vienādojumi $V(x)$ un $M(x)$**
        * **Sekcija I ($0 \le x < a$):** $V(x) = R_{A_{y}}$; $M(x) = R_{A_{y}} \cdot x$
        * **Sekcija II ($a \le x < L$):** $V(x) = R_{A_{y}} - F_{1v}$; $M(x) = R_{A_{y}} \cdot x - F_{1v} \cdot (x-a)$
        * **Sekcija III ($L \le x \le L_{kop}$):** $V(x) = R_{A_{y}} - F_{1v} + R_{B_{y}}$; $M(x) = R_{A_{y}} \cdot x - F_{1v} \cdot (x-a) + R_{B_{y}} \cdot (x-L)$
        """)

    elif example == "Piemērs 8":
        st.markdown(f"""
        **1. Reakciju Aprēķins (Iemūrējums A pie $x=L$, rēķinot no kreisā gala)**
        * $Q = q \cdot L$ (Kopējā slodze)
        * **Horizontālais Līdzsvars:** $\sum F_{x} = 0 \implies R_{A_{x}} - F_{3} = 0$
            $$R_{A_{x}} = F_{3} = {R_Ax} \\text{{ kN}}$$
        * **Vertikālais Līdzsvars:** $\sum F_{y} = 0 \implies R_{A_{y}} - Q = 0$
            $$R_{A_{y}} = Q = {R_Ay} \\text{{ kN}}$$
        * **Momenta Līdzsvars:** $\sum M_{A} = 0 \implies R_{M_{A}} - F_{3} \cdot h + Q \cdot (L/2) = 0$
            $$R_{M_{A}} = F_{3} \cdot h - Q \cdot \\frac{{L}}{{2}} = {R_MA} \\text{{ kNm}}$$
            *(Pozitīvs $R_{M_{A}}$ ir pretēji pulksteņrādītājam.)*

        **2. Iekšējo Spēku Vienādojumi $V(x)$ un $M(x)$ (Rēķinot no labā gala A, ar $x' = L-x$)**
        * **Šķērsspēks:**
            $$V(x) = -R_{A_{y}} + q \cdot x'$$
        * **Lieces Moments:**
            $$M(x) = -R_{M_{A}} + R_{A_{y}} \cdot x' - \\frac{{q \cdot (x')^2}}{{2}}$$
        """)
        
    elif example == "Piemērs 9":
        st.markdown(f"""
        **1. Reakciju Aprēķins (Iemūrējums A pie $x=0$)**
        * **Horizontālais Līdzsvars:** $\sum F_{x} = 0 \implies R_{A_{x}} - F_{3} = 0$
            $$R_{A_{x}} = F_{3} = {R_Ax} \\text{{ kN}}$$
        * **Vertikālais Līdzsvars:** $\sum F_{y} = 0 \implies R_{A_{y}} - F_{2} = 0$
            $$R_{A_{y}} = F_{2} = {R_Ay} \\text{{ kN}}$$
        * **Momenta Līdzsvars:** $\sum M_{A} = 0 \implies R_{M_{A}} - F_{2} \cdot L + F_{3} \cdot h = 0$
            $$R_{M_{A}} = F_{2} \cdot L - F_{3} \cdot h = {R_MA} \\text{{ kNm}}$$
            *(Pozitīvs $R_{M_{A}}$ ir pretēji pulksteņrādītājam.)*

        **2. Iekšējo Spēku Vienādojumi $V(x)$ un $M(x)$ (Rēķinot no kreisā gala A)**
        * **Sekcija I ($0 \le x < L$):**
            $$V(x) = R_{A_{y}}$$
            $$M(x) = R_{M_{A}} + R_{A_{y}} \cdot x$$
        * **Sekcija II ($L \le x \le L_{kop}$):**
            $$V(x) = R_{A_{y}} - F_{2}$$
            $$M(x) = R_{M_{A}} + R_{A_{y}} \cdot x - F_{2} \cdot (x-L)$$
        """)

    else:
        # Pārējie piemēri (2, 4, 5, 6, 7)
        st.markdown("""
        Pārējiem statiski nosakāmiem piemēriem (2, 4, 5, 6, 7) tiek izmantoti tie paši pamatprincipi:
        1.  **Reakcijas:** Aprēķinātas no $\sum M = 0$ un $\sum F_{y} = 0$.
        2.  **Iekšējie spēki:** Tiek izveidotas šķēluma funkcijas $V(x)$ un $M(x)$ katram slodzes intervālam. $V(x)$ ir lineāra funkcija no $M(x)$ atvasinājuma.
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
    index=0,
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
        F1 = st.number_input("Koncentrētā Slodze ($F_1$, kN)", value=15.0, min_value=0.0, step=1.0, key="F15")
        q = st.number_input("Izkliedētā Slodze ($q$, kN/m)", value=8.0, min_value=0.0, step=1.0, key="q5")
        st.markdown("---")
        Lk = st.number_input("Konsoles Garums ($L_k$, m)", value=1.0, min_value=0.1, step=0.5, key="Lk5")
        L = st.number_input("Galvenā Laiduma Garums ($L$, m)", value=6.0, min_value=0.1, step=0.5, key="L5")
        a = st.number_input("Izkliedētās Slodzes Garums ($a$, m)", value=4.0, min_value=0.0, max_value=L if L > 0 else 1.0, step=0.5, key="a5")
        R_dict, x_coords, V_values, M_values = solve_example_5(F1, q, Lk, L, a)
        R_list_template = ["A_y", "B_y", "A_x"]

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
        F3 = st.number_input("Horizontālā Slodze ($F_3$, kN)", value=10.0, min_value=0.0, step=1.0, key="F38")
        q = st.number_input("Izkliedētā Slodze ($q$, kN/m)", value=5.0, min_value=0.0, step=1.0, key="q8")
        st.markdown("---")
        h = st.number_input("Slodzes Nobīdes Augstums ($h$, m)", value=0.5, min_value=0.0, step=0.1, key="h8")
        L = st.number_input("Laiduma Garums ($L$, m)", value=8.0, min_value=0.1, step=0.5, key="L8")
        R_dict, x_coords, V_values, M_values = solve_example_8(q, F3, h, L)
        R_list_template = ["A_y", "A_x", "M_A"]

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
display_formulas(selected_example, R_dict)


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
