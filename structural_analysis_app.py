import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Configuration for Streamlit Page ---
st.set_page_config(layout="wide", page_title="Multi-Case Beam Analysis Calculator (9 Examples)")

# --- Constants for Uploaded Images ---
IMAGE_CONFIGS = {
    "Example 1": {
        "caption": "Hinge A, Roller B. F1 in span L, q on cantilever Lk.",
        "url": "exsample_1.JPG",
        "status": "Determinate",
    },
    "Example 2": {
        "caption": "Hinge A, Roller B. q on span L, F1 on cantilever Lk.",
        "url": "exsample_2.JPG",
        "status": "Determinate",
    },
    "Example 3": {
        "caption": "Hinge A, Roller B. Inclined F1 in span L, F2 on cantilever Lk.",
        "url": "exsample_3.JPG",
        "status": "Determinate",
    },
    "Example 4": {
        "caption": "Hinge A, Roller B. Inclined F1 and F2 on span L+b.",
        "url": "exsample_4.JPG",
        "status": "Determinate",
    },
    "Example 5": {
        "caption": "Cantilever F1 on Lk. Hinge A. q on span L (length a). Roller B.",
        "url": "exsample_5.JPG",
        "status": "Determinate",
    },
    "Example 6": {
        "caption": "q on cantilever Lk. Roller A. F1 on span L. Roller B.",
        "url": "exsample_6.JPG",
        "status": "Determinate",
    },
    "Example 7": {
        "caption": "Cantilever F1 on Lk. Roller A. F2, Inclined F3 on span L. Roller B.",
        "url": "exsample_7.JPG",
        "status": "Determinate",
    },
    "Example 8": {
        "caption": "Fixed A (right). q on L. Eccentric F3 at left end.",
        "url": "exsample_8.JPG",
        "status": "Indeterminate (Fixed)",
    },
    "Example 9": {
        "caption": "Fixed A (left). F2 at L. Eccentric F3 at right end (L+a).",
        "url": "exsample_9.JPG",
        "status": "Indeterminate (Fixed)",
    },
}

# --- Geometry Helper ---
def get_components(F, alpha_deg):
    """Calculates vertical and horizontal force components, assuming alpha is from the vertical."""
    alpha_rad = math.radians(alpha_deg)
    # F_v = F * cos(alpha) (Since alpha is from the vertical axis)
    F_v = F * math.cos(alpha_rad)
    # F_h = F * sin(alpha)
    F_h = F * math.sin(alpha_rad)
    return F_v, F_h

# --- SOLVER FUNCTIONS (Determinate Beams) ---

def calculate_diagrams(L_total, x_coords, R_dict, forces):
    """Generic function to calculate V and M diagrams for determinate beams."""
    
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    # Reactions are applied at their specific coordinates
    reactions = []
    if "A_y" in R_dict: reactions.append((R_dict["A_y"], 0.0, "Ay"))
    if "B_y" in R_dict: reactions.append((R_dict["B_y"], L_total if "L" in forces else forces.get("L_AB", 0.0), "By"))
    if "M_A" in R_dict: reactions.append((R_dict["M_A"], 0.0, "MA"))
    
    # Apply vertical concentrated forces and equivalent forces from distributed loads
    for i, x in enumerate(x_coords):
        V = 0
        M = 0
        
        # 1. Apply Reactions
        for R, R_pos, R_name in reactions:
            if R_name != "MA": # Skip moment reaction for V calculation
                # For fixed beams, A is at x=0. For overhanging, A is at x=0
                if x >= R_pos:
                    V += R
                    M += R * (x - R_pos)
            else: # Apply moment reaction at x=0
                M += R
        
        # 2. Apply Vertical Concentrated Loads
        for F, F_pos, F_type in forces.get("point_loads", []):
            if x >= F_pos:
                V -= F
                M -= F * (x - F_pos)
        
        # 3. Apply Distributed Loads (q)
        for q, q_start, q_end in forces.get("dist_loads", []):
            if x > q_start:
                x_prime = min(x, q_end) - q_start # Length of q applied up to x
                if x_prime > 0:
                    V -= q * x_prime
                    M -= q * x_prime * (x - q_start - x_prime / 2)
                    
        V_values[i] = V
        M_values[i] = M

    return V_values, M_values

def solve_example_1(F1, q, a, L, Lk):
    """Example 1: Hinge A, Roller B. F1 in span L, q on cantilever Lk. Determinate."""
    
    if L <= 0: st.error("Span length L must be positive."); return {}, np.array([0]), np.array([0]), np.array([0])
        
    L_total = L + Lk
    x_F1 = a
    
    # Equivalent point load for q
    Q = q * Lk
    x_Q = L + Lk / 2 
    
    # 1. Reactions
    # Sum M_A = 0: By * L - F1 * a - Q * x_Q = 0 (F1 and Q cause negative moment (CW) around A)
    By = (F1 * a + Q * x_Q) / L
    
    # Sum F_y = 0: Ay + By - F1 - Q = 0
    Ay = F1 + Q - By
    Ax = 0

    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    
    forces = {
        "point_loads": [(F1, x_F1, "F1")],
        "dist_loads": [(q, L, L_total)],
    }
    
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_2(F1, q, L, Lk):
    """Example 2: Hinge A, Roller B. q on span L, F1 on cantilever Lk. Determinate."""

    if L <= 0: st.error("Span length L must be positive."); return {}, np.array([0]), np.array([0]), np.array([0])

    L_total = L + Lk
    
    # Equivalent point load for q
    Q = q * L
    x_Q = L / 2 
    x_F1 = L_total
    
    # 1. Reactions
    # Sum M_A = 0: By * L - Q * x_Q - F1 * x_F1 = 0 
    By = (Q * x_Q + F1 * x_F1) / L
    
    # Sum F_y = 0: Ay + By - Q - F1 = 0
    Ay = Q + F1 - By
    Ax = 0

    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    
    forces = {
        "point_loads": [(F1, x_F1, "F1")],
        "dist_loads": [(q, 0, L)],
    }
    
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_3(F1, F2, alpha, a, L, Lk):
    """Example 3: Hinge A, Roller B. Inclined F1 in span L, F2 on cantilever Lk. Determinate."""
    
    if L <= 0: st.error("Span length L must be positive."); return {}, np.array([0]), np.array([0]), np.array([0])
        
    L_total = L + Lk
    
    # F1 components (alpha from vertical)
    F1v, F1h = get_components(F1, alpha)
    x_F1 = a
    x_F2 = L_total
    
    # 1. Reactions
    # Sum F_x = 0: Ax + F1h = 0 (assuming F1h acts left based on diagram)
    Ax = F1h
    
    # Sum M_A = 0: By * L - F1v * a - F2 * x_F2 = 0
    By = (F1v * a + F2 * x_F2) / L
    
    # Sum F_y = 0: Ay + By - F1v - F2 = 0
    Ay = F1v + F2 - By

    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    
    forces = {
        "point_loads": [(F1v, x_F1, "F1v"), (F2, x_F2, "F2")],
        "dist_loads": [],
    }
    
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_4(F1, F2, alpha, a, L_F2, b, Lk):
    """Example 4: Hinge A, Roller B. Inclined F1 at x=a. F2 at x=L_F2. Span A->B = L_F2+b. Determinate."""
    
    L_AB = L_F2 + b
    if L_AB <= 0: st.error("Span A to B must be positive."); return {}, np.array([0]), np.array([0]), np.array([0])
    
    L_total = L_AB + Lk
    
    # F1 components (alpha from vertical)
    F1v, F1h = get_components(F1, alpha)
    x_F1 = a
    x_F2 = L_F2
    x_B = L_AB
    
    # 1. Reactions
    # Sum F_x = 0: Ax + F1h = 0 
    Ax = F1h
    
    # Sum M_A = 0: By * L_AB - F1v * a - F2 * L_F2 = 0
    By = (F1v * a + F2 * L_F2) / L_AB
    
    # Sum F_y = 0: Ay + By - F1v - F2 = 0
    Ay = F1v + F2 - By

    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    
    forces = {
        "point_loads": [(F1v, x_F1, "F1v"), (F2, x_F2, "F2")],
        "dist_loads": [],
    }
    
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_5(F1, q, Lk, L, a):
    """Example 5: Cantilever F1 on Lk. Hinge A. q on span L (length a). Roller B. Determinate."""
    
    if L <= 0: st.error("Span length L must be positive."); return {}, np.array([0]), np.array([0]), np.array([0])
        
    L_total = Lk + L
    
    # Equivalent point load for q
    Q = q * a
    x_Q = Lk + a / 2
    x_F1 = 0
    
    # 1. Reactions
    # Sum M_A = 0 (A at Lk): By * L - F1 * Lk - Q * (a/2) = 0
    By = (F1 * Lk + Q * (a / 2)) / L
    
    # Sum F_y = 0: Ay + By - F1 - Q = 0
    Ay = F1 + Q - By
    Ax = 0

    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    
    forces = {
        "point_loads": [(F1, x_F1, "F1")],
        "dist_loads": [(q, Lk, Lk + a)],
    }
    
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_6(q, F1, Lk, L, a):
    """Example 6: q on cantilever Lk. Roller A. F1 on span L. Roller B. Determinate."""
    
    if L <= 0: st.error("Span length L must be positive."); return {}, np.array([0]), np.array([0]), np.array([0])
        
    L_total = Lk + L
    
    # Equivalent point load for q
    Q = q * Lk
    x_Q = Lk / 2
    x_F1 = Lk + a
    
    # 1. Reactions
    # Sum M_A = 0 (A at Lk): By * L + Q * (Lk / 2) - F1 * a = 0
    By = (F1 * a - Q * (Lk / 2)) / L
    
    # Sum F_y = 0: Ay + By - Q - F1 = 0
    Ay = Q + F1 - By
    Ax = 0

    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    
    forces = {
        "point_loads": [(F1, x_F1, "F1")],
        "dist_loads": [(q, 0, Lk)],
    }
    
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

def solve_example_7(F1, F2, F3, alpha, Lk, L, a, b):
    """Example 7: Cantilever F1 on Lk. Roller A. F2, Inclined F3 on span L. Roller B. Determinate."""
    
    if L <= 0: st.error("Span length L must be positive."); return {}, np.array([0]), np.array([0]), np.array([0])
        
    L_total = Lk + L
    
    # F3 components (alpha from vertical)
    F3v, F3h = get_components(F3, alpha)
    
    x_F1 = Lk # F1 is at the end of the cantilever
    x_F2 = Lk + a
    x_F3 = Lk + L - b
    
    # 1. Reactions (A at Lk)
    # Sum F_x = 0: Ax + F3h = 0 
    Ax = F3h
    
    # Sum M_A = 0: By * L - F1 * Lk - F2 * a - F3v * (L - b) = 0
    By = (F1 * Lk + F2 * a + F3v * (L - b)) / L
    
    # Sum F_y = 0: Ay + By - F1 - F2 - F3v = 0
    Ay = F1 + F2 + F3v - By

    R_dict = {"A_y": Ay, "B_y": By, "A_x": Ax}
    x_coords = np.linspace(0, L_total, 1000)
    
    forces = {
        "point_loads": [(F1, x_F1, "F1"), (F2, x_F2, "F2"), (F3v, x_F3, "F3v")],
        "dist_loads": [],
    }
    
    V_values, M_values = calculate_diagrams(L_total, x_coords, R_dict, forces)
    return R_dict, x_coords, V_values, M_values

# --- SOLVER FUNCTIONS (Indeterminate Beams - Static Equilibrium Only) ---

def solve_example_8(q, F3, h, L):
    """Example 8: Fixed A (right). q on L. Eccentric F3 at left end. Indeterminate (Fixed)."""
    
    st.warning("Example 8 is **Statically Indeterminate** (Fixed support). Reactions are calculated by static equilibrium, but diagram values are approximations.")
    
    # Reactions at A (x=L)
    Q = q * L # Total distributed load
    Ax = -F3 # F3 to the right, Ax to the left
    Ay = Q # Ay upward
    
    # Moment at A (reaction to external moments)
    # F3 * h (CW, negative) | Q * L/2 (CCW, positive)
    M_A = Q * (L / 2) - F3 * h
    
    R_dict = {"A_y": Ay, "A_x": Ax, "M_A": M_A}
    x_coords = np.linspace(0, L, 1000)
    
    # Diagram calculation (V and M from x=0, which is the left end)
    
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    for i, x in enumerate(x_coords):
        # Calculate V and M from the right (A)
        x_prime = L - x # distance from A (right end)
        
        # Shear V(x) (from right side: -Ay + q * x_prime)
        V = -Ay + q * x_prime
        
        # Moment M(x) (from right side: M_A + Ay * x_prime - q * x_prime^2 / 2)
        M = M_A + Ay * x_prime - q * x_prime**2 / 2
        
        # Adjust M for eccentric F3 moment at the left end
        # The F3*h couple is a pure moment acting everywhere.
        # This simplification ignores the exact M contribution of F3, so M is approximated.
        # For a full cantilever: M(x) = -F3 * h - q*x^2/2 (if x=0 is fixed)
        
        V_values[i] = V
        M_values[i] = M

    return R_dict, x_coords, V_values, M_values

def solve_example_9(F2, F3, h, L, a):
    """Example 9: Fixed A (left). F2 at L. Eccentric F3 at right end (L+a). Indeterminate (Fixed)."""
    
    st.warning("Example 9 is **Statically Indeterminate** (Fixed support). Reactions are calculated by static equilibrium, but diagram values are approximations.")
    
    L_total = L + a
    x_F2 = L
    
    # 1. Reactions at A (x=0)
    Ax = F3 # F3 to the left, Ax to the right
    Ay = F2 # Ay upward
    
    # Moment at A (reaction to external moments)
    # F2 * L (CW, negative) | F3 * h (CCW, positive)
    M_A = F2 * L - F3 * h
    
    R_dict = {"A_y": Ay, "A_x": Ax, "M_A": M_A}
    x_coords = np.linspace(0, L_total, 1000)
    
    # Diagram calculation (V and M from x=0, fixed end)
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    for i, x in enumerate(x_coords):
        V = Ay
        M = M_A + Ay * x
        
        if x >= x_F2:
            V -= F2
            M -= F2 * (x - x_F2)
        
        V_values[i] = V
        M_values[i] = M

    return R_dict, x_coords, V_values, M_values

# --- UI Layout ---

st.title("🏗️ Structural Beam Analysis Tool")
st.markdown("---")

# --- Beam Selector and Image ---
selected_example = st.sidebar.radio(
    "Select Beam Example to Solve (1-9):",
    options=[f"Example {i}" for i in range(1, 10)],
    index=0, # Default to Example 1
)

config = IMAGE_CONFIGS.get(selected_example)

st.subheader(f"Solving: {selected_example} - **{config['status']}**")
st.markdown(f"*{config['caption']}*")
st.image(config['url'], caption=f"Scheme for {selected_example}") 

st.markdown("---")

# --- Dynamic Input Parameters and Solution Call ---

R_dict = {}
x_coords, V_values, M_values = np.array([0]), np.array([0]), np.array([0])
alpha = 0 # Default angle

with st.sidebar:
    st.markdown(f"#### {selected_example} Inputs")
    
    if selected_example in ["Example 3", "Example 4", "Example 7"]:
        alpha = st.slider("Angle $\\alpha$ (degrees from vertical)", 0, 90, 45, key="alpha_slider")

    if selected_example == "Example 1":
        F1 = st.number_input("Point Load ($F_1$, kN)", value=10.0, min_value=0.0, step=1.0, key="F11")
        q = st.number_input("Distributed Load ($q$, kN/m)", value=5.0, min_value=0.0, step=1.0, key="q1")
        st.markdown("---")
        a = st.number_input("Distance 'a' to $F_1$ (m)", value=2.0, min_value=0.1, step=0.5, key="a1")
        L = st.number_input("Span Length $L$ (m)", value=6.0, min_value=0.1, step=0.5, key="L1")
        Lk = st.number_input("Cantilever Length $L_k$ (m)", value=2.0, min_value=0.0, step=0.5, key="Lk1")
        R_dict, x_coords, V_values, M_values = solve_example_1(F1, q, a, L, Lk)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Example 2":
        q = st.number_input("Distributed Load ($q$, kN/m)", value=8.0, min_value=0.0, step=1.0, key="q2")
        F1 = st.number_input("Point Load ($F_1$, kN)", value=12.0, min_value=0.0, step=1.0, key="F12")
        st.markdown("---")
        L = st.number_input("Span Length $L$ (m)", value=7.0, min_value=0.1, step=0.5, key="L2")
        Lk = st.number_input("Cantilever Length $L_k$ (m)", value=1.5, min_value=0.0, step=0.5, key="Lk2")
        R_dict, x_coords, V_values, M_values = solve_example_2(F1, q, L, Lk)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Example 3":
        F1 = st.number_input("Inclined Load ($F_1$, kN)", value=25.0, min_value=0.0, step=1.0, key="F13")
        F2 = st.number_input("Point Load ($F_2$, kN)", value=10.0, min_value=0.0, step=1.0, key="F23")
        st.markdown("---")
        a = st.number_input("Distance 'a' to $F_1$ (m)", value=3.0, min_value=0.1, step=0.5, key="a3")
        L = st.number_input("Span Length $L$ (m)", value=8.0, min_value=0.1, step=0.5, key="L3")
        Lk = st.number_input("Cantilever Length $L_k$ (m)", value=2.0, min_value=0.0, step=0.5, key="Lk3")
        R_dict, x_coords, V_values, M_values = solve_example_3(F1, F2, alpha, a, L, Lk)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Example 4":
        F1 = st.number_input("Inclined Load ($F_1$, kN)", value=15.0, min_value=0.0, step=1.0, key="F14")
        F2 = st.number_input("Point Load ($F_2$, kN)", value=20.0, min_value=0.0, step=1.0, key="F24")
        st.markdown("---")
        a = st.number_input("Dist. 'a' to $F_1$ from A (m)", value=2.0, min_value=0.1, step=0.5, key="a4")
        L_F2 = st.number_input("Dist. $L$ to $F_2$ from A (m)", value=5.0, min_value=0.1, step=0.5, key="LF24")
        b = st.number_input("Dist. 'b' from $F_2$ to B (m)", value=1.0, min_value=0.1, step=0.5, key="b4")
        Lk = st.number_input("Cantilever Length $L_k$ (m)", value=1.0, min_value=0.0, step=0.5, key="Lk4")
        R_dict, x_coords, V_values, M_values = solve_example_4(F1, F2, alpha, a, L_F2, b, Lk)
        R_list_template = ["A_y", "B_y", "A_x"]
        
    elif selected_example == "Example 5":
        F1 = st.number_input("Point Load ($F_1$, kN)", value=15.0, min_value=0.0, step=1.0, key="F15")
        q = st.number_input("Distributed Load ($q$, kN/m)", value=8.0, min_value=0.0, step=1.0, key="q5")
        st.markdown("---")
        Lk = st.number_input("Cantilever Length ($L_k$, m)", value=1.0, min_value=0.1, step=0.5, key="Lk5")
        L = st.number_input("Main Span Length ($L$, m)", value=6.0, min_value=0.1, step=0.5, key="L5")
        a = st.number_input("Distributed Load Length ($a$, m)", value=4.0, min_value=0.0, max_value=L if L > 0 else 1.0, step=0.5, key="a5")
        R_dict, x_coords, V_values, M_values = solve_example_5(F1, q, Lk, L, a)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Example 6":
        q = st.number_input("Distributed Load ($q$, kN/m)", value=10.0, min_value=0.0, step=1.0, key="q6")
        F1 = st.number_input("Point Load ($F_1$, kN)", value=25.0, min_value=0.0, step=1.0, key="F16")
        st.markdown("---")
        Lk = st.number_input("Cantilever Length ($L_k$, m)", value=2.0, min_value=0.1, step=0.5, key="Lk6")
        L = st.number_input("Span Length ($L$, m)", value=5.0, min_value=0.1, step=0.5, key="L6")
        a = st.number_input("Distance 'a' from A to $F_1$ (m)", value=3.0, min_value=0.0, max_value=L if L > 0 else 1.0, step=0.5, key="a6")
        R_dict, x_coords, V_values, M_values = solve_example_6(q, F1, Lk, L, a)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Example 7":
        F1 = st.number_input("Load $F_1$ on $L_k$ (kN)", value=10.0, min_value=0.0, step=1.0, key="F17")
        F2 = st.number_input("Load $F_2$ on span (kN)", value=5.0, min_value=0.0, step=1.0, key="F27")
        F3 = st.number_input("Inclined Load $F_3$ (kN)", value=15.0, min_value=0.0, step=1.0, key="F37")
        st.markdown("---")
        Lk = st.number_input("Cantilever Length $L_k$ (m)", value=2.0, min_value=0.1, step=0.5, key="Lk7")
        L = st.number_input("Span Length $L$ (m)", value=6.0, min_value=0.1, step=0.5, key="L7")
        a = st.number_input("Dist. 'a' to $F_2$ from A (m)", value=2.0, min_value=0.0, max_value=L, step=0.5, key="a7")
        b = st.number_input("Dist. 'b' from B to $F_3$ (m)", value=1.0, min_value=0.0, max_value=L, step=0.5, key="b7")
        R_dict, x_coords, V_values, M_values = solve_example_7(F1, F2, F3, alpha, Lk, L, a, b)
        R_list_template = ["A_y", "B_y", "A_x"]

    elif selected_example == "Example 8":
        F3 = st.number_input("Horizontal Load ($F_3$, kN)", value=10.0, min_value=0.0, step=1.0, key="F38")
        q = st.number_input("Distributed Load ($q$, kN/m)", value=5.0, min_value=0.0, step=1.0, key="q8")
        st.markdown("---")
        h = st.number_input("Load Offset Height ($h$, m)", value=0.5, min_value=0.0, step=0.1, key="h8")
        L = st.number_input("Span Length ($L$, m)", value=8.0, min_value=0.1, step=0.5, key="L8")
        R_dict, x_coords, V_values, M_values = solve_example_8(q, F3, h, L)
        R_list_template = ["A_y", "A_x", "M_A"]

    elif selected_example == "Example 9":
        F2 = st.number_input("Point Load ($F_2$, kN)", value=20.0, min_value=0.0, step=1.0, key="F29")
        F3 = st.number_input("Horizontal Load ($F_3$, kN)", value=10.0, min_value=0.0, step=1.0, key="F39")
        st.markdown("---")
        h = st.number_input("Load Offset Height ($h$, m)", value=0.5, min_value=0.0, step=0.1, key="h9")
        L = st.number_input("Distance $L$ to $F_2$ from A (m)", value=4.0, min_value=0.1, step=0.5, key="L9")
        a = st.number_input("Distance 'a' from $F_2$ to end (m)", value=2.0, min_value=0.0, step=0.5, key="a9")
        R_dict, x_coords, V_values, M_values = solve_example_9(F2, F3, h, L, a)
        R_list_template = ["A_y", "A_x", "M_A"]

# --- Output Section ---

st.subheader("Solution: Support Reactions")

R_list = []
for key in R_list_template:
    # Formatting for display in LaTeX
    display_key = key.replace('_y', '_{y}').replace('_x', '_{x}').replace('_A', '_{A}').replace('M_A', 'M_{A}')
    
    if key.startswith("M_"):
        unit = "kNm"
    else:
        unit = "kN"
        
    R_list.append(
        f"| ${display_key}$ | **{R_dict.get(key, 0):,.2f}** | {unit} |"
    )

st.markdown(f"""
| Reaction | Value | Unit |
| :---: | :---: | :---: |
{"\n".join(R_list)}
""")

st.markdown("---")

st.subheader("Shear Force ($V$) and Bending Moment ($M$) Diagrams")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# --- Shear Force Diagram (V) ---
ax1.plot(x_coords, V_values, label='$V(x)$', color='#1e56a0')
ax1.axhline(0, color='black', linewidth=0.5) 
ax1.set_title("Shear Force Diagram $V(x)$", fontsize=16)
ax1.set_xlabel("x (m)")
ax1.set_ylabel("Shear Force (kN)")
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.fill_between(x_coords, V_values, 0, where=(V_values > 0), color='#1e56a0', alpha=0.2)
ax1.fill_between(x_coords, V_values, 0, where=(V_values < 0), color='#ff7f0e', alpha=0.2)
ax1.legend(loc='upper right')

# --- Bending Moment Diagram (M) ---
ax2.plot(x_coords, M_values, label='$M(x)$', color='#d62728')
ax2.axhline(0, color='black', linewidth=0.5) 
ax2.set_title("Bending Moment Diagram $M(x)$", fontsize=16)
ax2.set_xlabel("x (m)")
ax2.set_ylabel("Bending Moment (kNm)")
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.fill_between(x_coords, M_values, 0, where=(M_values > 0), color='#2ca02c', alpha=0.2)
ax2.fill_between(x_coords, M_values, 0, where=(M_values < 0), color='#d62728', alpha=0.2)
ax2.legend(loc='upper right')

plt.tight_layout()
st.pyplot(fig)
