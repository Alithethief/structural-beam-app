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

# --- Custom CSS for Better Styling ---
st.markdown("""
<style>
.formula-box {
    background-color: #f0f8ff;
    border-left: 5px solid #2E86AB;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.result-box {
    background-color: #f0fff0;
    border: 2px solid #32cd32;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# --- Ģeometrijas Palīgfunkcija ---
def get_components(F, alpha_deg):
    """Aprēķina vertikālos un horizontālos spēka komponentus, pieņemot, ka alfa ir no vertikāles."""
    alpha_rad = math.radians(alpha_deg)
    F_v = F * math.cos(alpha_rad)
    F_h = F * math.sin(alpha_rad)
    return F_v, F_h

# --- UZLABOTA DIAGRAMMU APRĒĶINĀŠANAS FUNKCIJA ---
def calculate_diagrams(L_total, x_coords, R_dict, forces):
    """
    Uzlabota funkcija V un M diagrammu aprēķināšanai ar precīzāku loģiku.
    """
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    # Balstu pozīcijas
    A_pos = forces.get("A_pos", 0.0)
    B_pos = forces.get("B_pos", forces.get("L_AB", L_total))
    
    for i, x in enumerate(x_coords):
        V = 0
        M = 0
        
        # 1. Pievienojam momenta reakciju (tikai iemūrējumam)
        if "M_A" in R_dict and A_pos == 0.0:
            M += R_dict["M_A"]
        
        # 2. Pievienojam balstu reakcijas (no kreisās puses)
        if "A_y" in R_dict and x >= A_pos:
            V += R_dict["A_y"]
            M += R_dict["A_y"] * (x - A_pos)
            
        if "B_y" in R_dict and x >= B_pos:
            V += R_dict["B_y"]
            M += R_dict["B_y"] * (x - B_pos)
        
        # 3. Pievienojam koncentrētās slodzes
        for F_val, F_pos, _ in forces.get("point_loads", []):
            if x > F_pos:
                V -= F_val
                M -= F_val * (x - F_pos)
        
        # 4. Pievienojam izkliedētās slodzes
        for q_val, q_start, q_end in forces.get("dist_loads", []):
            if x > q_start:
                active_length = min(x, q_end) - q_start
                if active_length > 0:
                    total_force = q_val * active_length
                    V -= total_force
                    center_distance = active_length / 2
                    M -= total_force * (x - q_start - center_distance)
        
        V_values[i] = V
        M_values[i] = M
    
    return V_values, M_values

# --- ATRISINĀTĀJU FUNKCIJAS ---
def solve_example_1(F1, q, a, L, Lk):
    """Piemērs 1: Šarnīrs A, Rullītis B. F1 laidumā L, q uz konsoles Lk."""
    if L <= 0: 
        st.error("Laiduma garumam L jābūt pozitīvam.")
        return {}, np.array([0]), np.array([0]), np.array([0])
    
    L_total = L + Lk
    Q = q * Lk
    x_Q_center = L + Lk / 2
    
    By = (F1 * a + Q * x_Q_center) / L
    Ay = F1 + Q - By
    Ax = 0
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    forces = {
        "point_loads": [(F1, a, "F1")], 
        "dist_loads": [(q, L, L_total)], 
        "L_AB": L,
        "A_pos": 0.0,
        "B_pos": L
    }
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_2(F1, q, L, Lk):
    """Piemērs 2: Šarnīrs A, Rullītis B. q uz laiduma L, F1 uz konsoles Lk."""
    if L <= 0:
        st.error("Laiduma garumam L jābūt pozitīvam.")
        return {}, np.array([0]), np.array([0]), np.array([0])
    
    L_total = L + Lk
    Q = q * L
    x_Q = L / 2
    x_F1 = L_total
    
    By = (Q * x_Q + F1 * x_F1) / L
    Ay = Q + F1 - By
    Ax = 0
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    forces = {
        "point_loads": [(F1, x_F1, "F1")], 
        "dist_loads": [(q, 0, L)], 
        "L_AB": L,
        "A_pos": 0.0,
        "B_pos": L
    }
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_3(F1, F2, alpha, a, L, Lk):
    """Piemērs 3: Slīpa F1 laidumā L, F2 uz konsoles Lk."""
    if L <= 0:
        st.error("Laiduma garumam L jābūt pozitīvam.")
        return {}, np.array([0]), np.array([0]), np.array([0])
    
    L_total = L + Lk
    F1v, F1h = get_components(F1, alpha)
    
    Ax = F1h
    By = (F1v * a + F2 * L_total) / L
    Ay = F1v + F2 - By
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    forces = {
        "point_loads": [(F1v, a, "F1v"), (F2, L_total, "F2")], 
        "dist_loads": [], 
        "L_AB": L,
        "A_pos": 0.0,
        "B_pos": L
    }
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_4(F1, F2, alpha, a, L_F2, b, Lk):
    """Piemērs 4: Slīpa F1 un F2."""
    L_AB = L_F2 + b
    if L_AB <= 0:
        st.error("Laidumam no A līdz B jābūt pozitīvam.")
        return {}, np.array([0]), np.array([0]), np.array([0])
    
    L_total = L_AB + Lk
    F1v, F1h = get_components(F1, alpha)
    
    Ax = F1h
    By = (F1v * a + F2 * L_F2) / L_AB
    Ay = F1v + F2 - By
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    forces = {
        "point_loads": [(F1v, a, "F1v"), (F2, L_F2, "F2")], 
        "dist_loads": [], 
        "L_AB": L_AB,
        "A_pos": 0.0,
        "B_pos": L_AB
    }
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_5(F1, q, Lk, L, a):
    """Piemērs 5: F1 uz konsoles Lk. Šarnīrs A. q uz laiduma L (garums a). Rullītis B."""
    if L <= 0:
        st.error("Laiduma garumam L jābūt pozitīvam.")
        return {}, np.array([0]), np.array([0]), np.array([0])
    
    L_total = L
    Q = q * a
    x_Q_center = a / 2
    
    By = (Q * x_Q_center + F1 * Lk) / L
    Ay = Q + F1 - By
    Ax = 0
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
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
    """Piemērs 6: q uz konsoles Lk. Rullītis A. F1 uz laiduma L. Rullītis B."""
    if L <= 0:
        st.error("Laiduma garumam L jābūt pozitīvam.")
        return {}, np.array([0]), np.array([0]), np.array([0])
    
    L_total = Lk + L
    Q = q * Lk
    
    By = (F1 * a + Q * (Lk/2)) / L
    Ay = Q + F1 - By
    Ax = 0
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(-Lk, L, 1000)
    forces = {
        "point_loads": [(F1, a, "F1")], 
        "dist_loads": [(q, -Lk, 0)], 
        "L_AB": L,
        "A_pos": 0.0,
        "B_pos": L
    }
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_7(F1, F2, F3, alpha, Lk, L, a, b):
    """Piemērs 7: F1 uz konsoles Lk. Rullītis A. F2, slīpa F3 uz laiduma L. Rullītis B."""
    if L <= 0:
        st.error("Laiduma garumam L jābūt pozitīvam.")
        return {}, np.array([0]), np.array([0]), np.array([0])
    
    L_total = Lk + L
    F3v, F3h = get_components(F3, alpha)
    
    Ax = F3h
    By = (F1 * Lk + F2 * a + F3v * (L - b)) / L
    Ay = F1 + F2 + F3v - By
    
    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(-Lk, L, 1000)
    forces = {
        "point_loads": [(F1, -Lk, "F1"), (F2, a, "F2"), (F3v, L-b, "F3v")], 
        "dist_loads": [], 
        "L_AB": L,
        "A_pos": 0.0,
        "B_pos": L
    }
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_8(q, F3, h, L):
    """Piemērs 8: LABOTS - Iemūrēts A (labajā galā). q uz L. Ekscentriska F3 kreisajā galā."""
    
    Q = q * L
    
    # Statiskais līdzsvars
    Ax = F3
    Ay = Q
    M_A = Q * (L/2) - F3 * h
    
    R_dict = {"A_y": Ay, "A_x": Ax, "M_A": M_A}
    x_coords = np.linspace(0, L, 1000)
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    for i, x in enumerate(x_coords):
        V_values[i] = -q * x
        M_values[i] = -q * x**2 / 2 + F3 * h
    
    return R_dict, x_coords, V_values, M_values

def solve_example_9(F2, F3, h, L, a):
    """Piemērs 9: LABOTS - Iemūrēts A (kreisajā galā). F2 pie L. Ekscentriska F3 labajā galā."""
    
    L_total = L + a
    
    # Statiskais līdzsvars
    Ax = F3
    Ay = F2
    M_A = F2 * L - F3 * h
    
    R_dict = {"A_y": Ay, "A_x": Ax, "M_A": M_A}
    x_coords = np.linspace(0, L_total, 1000)
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    for i, x in enumerate(x_coords):
        V = Ay
        M = M_A + Ay * x
        
        if x >= L:
            V -= F2
            M -= F2 * (x - L)
        
        V_values[i] = V
        M_values[i] = M
    
    return R_dict, x_coords, V_values, M_values

# --- FORMULU RĀDĪŠANAS FUNKCIJA ---
def display_formulas(example, R_dict, input_params):
    """Parāda aprēķina formulas ar paskaidrojumiem."""
    
    st.markdown("### 📋 Aprēķinu Formulas un Loģika")
    
    R_Ay = f"{R_dict.get('A_y', 0):.2f}"
    R_By = f"{R_dict.get('B_y', 0):.2f}"
    R_Ax = f"{R_dict.get('A_x', 0):.2f}"
    R_MA = f"{R_dict.get('M_A', 0):.2f}"
    
    if example == "Piemērs 8":
        F3 = input_params.get('F3', 0)
        q = input_params.get('q', 0)
        h = input_params.get('h', 0)
        L = input_params.get('L', 1)
        
        Q = q * L
        
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **Konsoles sija ar iemūrētu atbalstu labajā galā**
        
        **1. Slogojums:**
        - Izkliedētā slodze: q = {q} kN/m uz garumu L = {L} m
        - Rezultējošais spēks: Q = q × L = {q} × {L} = {Q:.2f} kN
        - Ekscentriskā slodze: F₃ = {F3} kN ar nobīdi h = {h} m
        
        **2. Reakciju aprēķins:**
        - **Horizontālais līdzsvars:** Rₐₓ = F₃ = {R_Ax} kN
        - **Vertikālais līdzsvars:** Rₐᵧ = Q = {R_Ay} kN
        - **Momenta līdzsvars:** Mₐ = Q×(L/2) - F₃×h = {Q:.2f}×{L/2:.2f} - {F3}×{h} = {R_MA} kNm
        
        **3. Diagrammu aprēķins (no brīvā gala x=0):**
        - **Šķērsspēks:** V(x) = -q×x
        - **Moments:** M(x) = -q×x²/2 + F₃×h
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    elif example == "Piemērs 5":
        F1 = input_params.get('F1', 0)
        q = input_params.get('q', 0)
        Lk = input_params.get('Lk', 0)
        L = input_params.get('L', 1)
        a = input_params.get('a', 0)
        
        Q = q * a
        x_Q = a / 2
        
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **Sija ar konsoli un izkliedētu slodzi**
        
        **1. Izkliedētās slodzes rezultants:**
        - Q = q × a = {q} × {a} = {Q:.2f} kN pie x = {x_Q:.2f} m
        
        **2. Momenta līdzsvars ap A:**
        - Rᵦᵧ × {L} = Q × {x_Q:.2f} + F₁ × {Lk}
        - Rᵦᵧ = ({Q:.2f} × {x_Q:.2f} + {F1} × {Lk}) / {L} = {R_By} kN
        
        **3. Vertikālais līdzsvars:**
        - Rₐᵧ = Q + F₁ - Rᵦᵧ = {Q:.2f} + {F1} - {R_By} = {R_Ay} kN
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.markdown("""
        **Šis piemērs izmanto statiskā līdzsvara vienādojumus:**
        
        1. **ΣFₓ = 0** (Horizontālais līdzsvars)
        2. **ΣFᵧ = 0** (Vertikālais līdzsvars)
        3. **ΣM = 0** (Momenta līdzsvars)
        
        Reakcijas tiek aprēķinātas, izmantojot šos vienādojumus un pēc tam pārbaudītas.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")

# --- UI IZKĀRTOJUMS ---
st.title("🏗️ Sijas Konstrukciju Analīzes Rīks")
st.markdown("---")

# Sijas izvēle
example_options = [f"Piemērs {i}" for i in range(1, 10)]
selected_example = st.sidebar.radio(
    "Izvēlieties Sijas Piemēru (1-9):",
    options=example_options,
    index=7,
)

config = IMAGE_CONFIGS.get(selected_example)

st.subheader(f"Atrisinājums: {selected_example} - **{config['status']}**")
st.markdown(f"*{config['caption']}*")
st.image(config['url'], caption=f"Shēma piemēram {selected_example}")

st.markdown("---")

# --- Dinamiskie Ievades Parametri ---
R_dict = {}
x_coords, V_values, M_values = np.array([0]), np.array([0]), np.array([0])
alpha = 0
input_params = {}

with st.sidebar:
    st.markdown(f"#### {selected_example} Ievades Parametri")
    
    if selected_example in ["Piemērs 3", "Piemērs 4", "Piemērs 7"]:
        alpha = st.slider("Leņķis α (grādos no vertikāles)", 0, 90, 45, key="alpha_slider")

    if selected_example == "Piemērs 1":
        F1 = st.number_input("Koncentrētā Slodze (F₁, kN)", value=10.0, min_value=0.0, step=1.0, key="F11")
        q = st.number_input("Izkliedētā Slodze (q, kN/m)", value=5.0, min_value=0.0, step=1.0, key="q1")
        st.markdown("---")
        a = st.number_input("Attālums 'a' līdz F₁ (m)", value=2.0, min_value=0.1, step=0.5, key="a1")
        L = st.number_input("Laiduma Garums L (m)", value=6.0, min_value=0.1, step=0.5, key="L1")
        Lk = st.number_input("Konsoles Garums Lk (m)", value=2.0, min_value=0.0, step=0.5, key="Lk1")
        R_dict, x_coords, V_values, M_values = solve_example_1(F1, q, a, L, Lk)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Piemērs 2":
        q = st.number_input("Izkliedētā Slodze (q, kN/m)", value=8.0, min_value=0.0, step=1.0, key="q2")
        F1 = st.number_input("Koncentrētā Slodze (F₁, kN)", value=12.0, min_value=0.0, step=1.0, key="F12")
        st.markdown("---")
        L = st.number_input("Laiduma Garums L (m)", value=7.0, min_value=0.1, step=0.5, key="L2")
        Lk = st.number_input("Konsoles Garums Lk (m)", value=1.5, min_value=0.0, step=0.5, key="Lk2")
        R_dict, x_coords, V_values, M_values = solve_example_2(F1, q, L, Lk)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Piemērs 3":
        F1 = st.number_input("Slīpā Slodze (F₁, kN)", value=25.0, min_value=0.0, step=1.0, key="F13")
        F2 = st.number_input("Koncentrētā Slodze (F₂, kN)", value=10.0, min_value=0.0, step=1.0, key="F23")
        st.markdown("---")
        a = st.number_input("Attālums 'a' līdz F₁ (m)", value=3.0, min_value=0.1, step=0.5, key="a3")
        L = st.number_input("Laiduma Garums L (m)", value=8.0, min_value=0.1, step=0.5, key="L3")
        Lk = st.number_input("Konsoles Garums Lk (m)", value=2.0, min_value=0.0, step=0.5, key="Lk3")
        R_dict, x_coords, V_values, M_values = solve_example_3(F1, F2, alpha, a, L, Lk)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Piemērs 4":
        F1 = st.number_input("Slīpā Slodze (F₁, kN)", value=15.0, min_value=0.0, step=1.0, key="F14")
        F2 = st.number_input("Koncentrētā Slodze (F₂, kN)", value=20.0, min_value=0.0, step=1.0, key="F24")
        st.markdown("---")
        a = st.number_input("Attālums 'a' līdz F₁ no A (m)", value=2.0, min_value=0.1, step=0.5, key="a4")
        L_F2 = st.number_input("Attālums L līdz F₂ no A (m)", value=5.0, min_value=0.1, step=0.5, key="LF24")
        b = st.number_input("Attālums 'b' no F₂ līdz B (m)", value=1.0, min_value=0.1, step=0.5, key="b4")
        Lk = st.number_input("Konsoles Garums Lk (m)", value=1.0, min_value=0.0, step=0.5, key="Lk4")
        R_dict, x_coords, V_values, M_values = solve_example_4(F1, F2, alpha, a, L_F2, b, Lk)
        R_list_template = ["A_y", "B_y", "A_x"]
        
    elif selected_example == "Piemērs 5":
        F1 = st.number_input("Koncentrētā Slodze (F₁, kN)", value=6.0, min_value=0.0, step=1.0, key="F15")
        q = st.number_input("Izkliedētā Slodze (q, kN/m)", value=6.0, min_value=0.0, step=1.0, key="q5")
        st.markdown("---")
        Lk = st.number_input("Attālums Lk līdz F₁ no A (m)", value=1.80, min_value=0.0, step=0.5, key="Lk5")
        L = st.number_input("Galvenā Laiduma Garums L (m) A līdz B", value=4.90, min_value=0.1, step=0.5, key="L5")
        a = st.number_input("Izkliedētās Slodzes Garums a (m)", value=1.80, min_value=0.0, step=0.5, key="a5")
        R_dict, x_coords, V_values, M_values = solve_example_5(F1, q, Lk, L, a)
        R_list_template = ["A_y", "B_y", "A_x"]
        input_params = {'F1': F1, 'q': q, 'Lk': Lk, 'L': L, 'a': a}

    elif selected_example == "Piemērs 6":
        q = st.number_input("Izkliedētā Slodze (q, kN/m)", value=10.0, min_value=0.0, step=1.0, key="q6")
        F1 = st.number_input("Koncentrētā Slodze (F₁, kN)", value=25.0, min_value=0.0, step=1.0, key="F16")
        st.markdown("---")
        Lk = st.number_input("Konsoles Garums Lk (m)", value=2.0, min_value=0.1, step=0.5, key="Lk6")
        L = st.number_input("Laiduma Garums L (m)", value=5.0, min_value=0.1, step=0.5, key="L6")
        a = st.number_input("Attālums 'a' no A līdz F₁ (m)", value=3.0, min_value=0.0, step=0.5, key="a6")
        R_dict, x_coords, V_values, M_values = solve_example_6(q, F1, Lk, L, a)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Piemērs 7":
        F1 = st.number_input("Slodze F₁ uz Lk (kN)", value=10.0, min_value=0.0, step=1.0, key="F17")
        F2 = st.number_input("Slodze F₂ laidumā (kN)", value=5.0, min_value=0.0, step=1.0, key="F27")
        F3 = st.number_input("Slīpā Slodze F₃ (kN)", value=15.0, min_value=0.0, step=1.0, key="F37")
        st.markdown("---")
        Lk = st.number_input("Konsoles Garums Lk (m)", value=2.0, min_value=0.1, step=0.5, key="Lk7")
        L = st.number_input("Laiduma Garums L (m)", value=6.0, min_value=0.1, step=0.5, key="L7")
        a = st.number_input("Attālums 'a' līdz F₂ no A (m)", value=2.0, min_value=0.0, step=0.5, key="a7")
        b = st.number_input("Attālums 'b' no B līdz F₃ (m)", value=1.0, min_value=0.0, step=0.5, key="b7")
        R_dict, x_coords, V_values, M_values = solve_example_7(F1, F2, F3, alpha, Lk, L, a, b)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Piemērs 8":
        F3 = st.number_input("Horizontālā Slodze (F₃, kN)", value=2.5, min_value=0.0, step=0.1, key="F38")
        q = st.number_input("Izkliedētā Slodze (q, kN/m)", value=6.0, min_value=0.0, step=1.0, key="q8")
        st.markdown("---")
        h = st.number_input("Slodzes Nobīdes Augstums h (m)", value=0.9, min_value=0.0, step=0.1, key="h8")
        L = st.number_input("Laiduma Garums L (m)", value=4.90, min_value=0.1, step=0.5, key="L8")
        R_dict, x_coords, V_values, M_values = solve_example_8(q, F3, h, L)
        R_list_template = ["A_y", "A_x", "M_A"]
        input_params = {'F3': F3, 'q': q, 'h': h, 'L': L}

    elif selected_example == "Piemērs 9":
        F2 = st.number_input("Koncentrētā Slodze (F₂, kN)", value=20.0, min_value=0.0, step=1.0, key="F29")
        F3 = st.number_input("Horizontālā Slodze (F₃, kN)", value=10.0, min_value=0.0, step=1.0, key="F39")
        st.markdown("---")
        h = st.number_input("Slodzes Nobīdes Augstums h (m)", value=0.5, min_value=0.0, step=0.1, key="h9")
        L = st.number_input("Attālums L līdz F₂ no A (m)", value=4.0, min_value=0.1, step=0.5, key="L9")
        a = st.number_input("Attālums 'a' no F₂ līdz galam (m)", value=2.0, min_value=0.0, step=0.5, key="a9")
        R_dict, x_coords, V_values, M_values = solve_example_9(F2, F3, h, L, a)
        R_list_template = ["A_y", "A_x", "M_A"]

# --- Izvades Sadaļa: Formulu Rādīšana ---
display_formulas(selected_example, R_dict, input_params)

# --- Izvades Sadaļa: Reakcijas ---
st.subheader("🎯 Risinājums: Atbalsta Reakcijas")

st.markdown('<div class="result-box">', unsafe_allow_html=True)
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
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# --- Izvades Sadaļa: Diagrammas ---
st.subheader("📊 Šķērsspēku (V) un Lieces Momenta (M) Diagrammas")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# --- Šķērsspēku Diagramma (V) ---
ax1.plot(x_coords, V_values, 'b-', linewidth=2.5, label='V(x)')
ax1.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)
ax1.fill_between(x_coords, V_values, 0, where=(V_values > 0), color='lightblue', alpha=0.6, label='Pozitīvs V')
ax1.fill_between(x_coords, V_values, 0, where=(V_values < 0), color='lightcoral', alpha=0.6, label='Negatīvs V')
ax1.set_title("Šķērsspēku Diagramma V(x)", fontsize=16, fontweight='bold')
ax1.set_xlabel("Attālums x (m)", fontsize=12)
ax1.set_ylabel("Šķērsspēks V (kN)", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Pievienot anotācijas maksimālajām/minimālajām vērtībām
max_V = np.max(V_values)
min_V = np.min(V_values)
if abs(max_V) > 0.01:
    max_idx = np.argmax(V_values)
    ax1.annotate(f'Max V = {max_V:.2f} kN', 
                 xy=(x_coords[max_idx], max_V), 
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
if abs(min_V) > 0.01:
    min_idx = np.argmin(V_values)
    ax1.annotate(f'Min V = {min_V:.2f} kN', 
                 xy=(x_coords[min_idx], min_V), 
                 xytext=(10, -20), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# --- Lieces Momenta Diagramma (M) ---
ax2.plot(x_coords, M_values, 'r-', linewidth=2.5, label='M(x)')
ax2.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)
ax2.fill_between(x_coords, M_values, 0, where=(M_values > 0), color='lightgreen', alpha=0.6, label='Pozitīvs M (Sagging)')
ax2.fill_between(x_coords, M_values, 0, where=(M_values < 0), color='lightpink', alpha=0.6, label='Negatīvs M (Hogging)')
ax2.set_title("Lieces Momenta Diagramma M(x)", fontsize=16, fontweight='bold')
ax2.set_xlabel("Attālums x (m)", fontsize=12)
ax2.set_ylabel("Lieces Moments M (kNm)", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Pievienot anotācijas maksimālajām/minimālajām vērtībām
max_M = np.max(M_values)
min_M = np.min(M_values)
if abs(max_M) > 0.01:
    max_idx = np.argmax(M_values)
    ax2.annotate(f'Max M = {max_M:.2f} kNm', 
                 xy=(x_coords[max_idx], max_M), 
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
if abs(min_M) > 0.01:
    min_idx = np.argmin(M_values)
    ax2.annotate(f'Min M = {min_M:.2f} kNm', 
                 xy=(x_coords[min_idx], min_M), 
                 xytext=(10, -20), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
st.pyplot(fig)

# --- Educational Section ---
st.markdown("---")
st.markdown("### 📚 Mācību Piezīmes")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 🔧 Līdzsvara Vienādojumi
    
    **Katrai sijai jāizpilda trīs līdzsvara nosacījumi:**
    
    1. **ΣFₓ = 0** (Horizontālo spēku līdzsvars)
    2. **ΣFᵧ = 0** (Vertikālo spēku līdzsvars)  
    3. **ΣM = 0** (Momenta līdzsvars ap jebkuru punktu)
    
    **💡 Padoms:** Izvēlieties momenta centru gudri, lai izslēgtu nezināmās reakcijas un vienkāršotu aprēķinus.
    """)

with col2:
    st.markdown("""
    #### 📊 Zīmju Konvencijas
    
    **Izmantotās standarta zīmju konvencijas:**
    
    - **Spēki:** ↑ Pozitīvi, ↓ Negatīvi
    - **Momenti:** ⟲ Pretpulksteņrādītājs Pozitīvs
    - **Šķērsspēks:** Uz augšu labajā pusē = Pozitīvs
    - **Lieces Moments:** Sagging (stiepe apakšā) = Pozitīvs
    
    **💡 Padoms:** Konsekventa zīmju konvencija ir ļoti svarīga pareiziem rezultātiem!
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 8px; margin-top: 2rem;'>
<h4>🎓 Praktizējieties!</h4>
<p>Izmēģiniet dažādas parametru kombinācijas, lai saprastu, kā slodzes ietekmē sijas uzvedību. 
Pievērsiet uzmanību, kā mainās reakcijas un kā diagrammas reaģē uz dažādiem slodžu nosacījumiem.</p>
</div>
""", unsafe_allow_html=True)
