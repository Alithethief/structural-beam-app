import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration for Streamlit Page ---
st.set_page_config(layout="wide", page_title="Multi-Case Beam Analysis Calculator")

# --- Constants for Uploaded Images (Used for Visualizing the Schemes) ---
# NOTE: The Streamlit environment must have access to these files (e.g., if uploaded to a 'data' folder in GitHub)
# For the Canvas environment, we use the uploaded file names directly.
IMAGE_CONFIGS = {
    "Example 5": {
        "caption": "Beam with cantilever and distributed load (Statically Determinate)",
        # Use the actual uploaded file name
        "url": "uploaded:slodzes.PNG-14cfd484-2caa-473a-b60b-f8c352dab8d9", 
    },
    "Example 8": {
        "caption": "Fixed-end cantilever beam with horizontal and distributed loads (Statically Indeterminate)",
        # Use the actual uploaded file name
        "url": "uploaded:Slodzes_02.PNG-88cbea13-0d91-4c44-bb2c-cd6e06eaa148",
    },
    # We keep the old examples for structural completeness, but focus on 5 and 8
    "Example 6": {
        "caption": "Cantilevered beam supported by two rollers (Statically Determinate)",
        "url": "https://placehold.co/700x150/0F79BD/FFFFFF?text=EXAMPLE+6:+Cantilever+Left+of+A,+Point+Load+F1+in+Span+AB",
    },
    "Example 1": {
        "caption": "Two-span continuous beam (Statically Indeterminate)",
        "url": "https://placehold.co/700x150/5C6BC0/FFFFFF?text=EXAMPLE+1:+Two-Span+Continuous+Beam",
    }
}

# --- FUNCTIONS FOR EACH EXAMPLE ---

def solve_example_1(F1, a, b):
    # Placeholder for Statically Indeterminate Beam (Three Supports)
    L_AB = a + b 
    L_BC = a     
    L = L_AB + L_BC
    
    st.warning("Example 1 is **Statically Indeterminate**. The results below are simplified placeholder values.")
    
    R_A = F1 * 0.4
    R_B = F1 * 1.3
    R_C = F1 * 0.3
    
    x_coords = np.linspace(0, L, 1000)
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    V_values = np.concatenate([
        np.linspace(R_A, R_A - F1, 500), 
        np.linspace(-R_C, R_C, 500)      
    ])[:1000] 

    M_values = np.concatenate([
        np.linspace(0, -F1 * L_AB * 0.1, 500),  
        np.linspace(-F1 * L_AB * 0.1, 0, 500)
    ])[:1000]

    return {"A": R_A, "B": R_B, "C": R_C}, x_coords, V_values, M_values

def solve_example_5(q, F1, Lk, L, a):
    """
    Solves Statically DETERMINATE Beam (Example 5): Cantilever (Lk) -> A (Hinge) -> Span (L) -> B (Roller).
    Loads: Point load F1 at left end (x=0), Distributed load 'q' over distance 'a' starting at A.
    """
    
    if L <= 0:
        st.error("Span length L must be greater than zero for Example 5.")
        return {"A_y": 0, "B_y": 0, "A_x": 0}, np.array([0]), np.array([0]), np.array([0])
        
    # The hinge at A provides horizontal and vertical reactions (Ax, Ay).
    # Since only vertical loads F1 and q exist, Ax = 0.
    Ax = 0
    
    # Total distributed load force and its distance from A
    Q = q * a
    x_Q = a / 2
    
    # 1. Calculate Support Reactions (Ay, By)
    
    # Sum of Moments about A (Sigma M_A = 0). Clockwise is positive.
    # By * L = (F1 * Lk) + (Q * x_Q)
    By = (F1 * Lk + Q * x_Q) / L
    
    # Sum of Vertical Forces (Sigma F_y = 0). Upward is positive.
    # Ay = F1 + Q - By
    Ay = F1 + Q - By

    # 2. Calculate Shear (V) and Moment (M) Functions
    Lt = Lk + L
    x_coords = np.linspace(0, Lt, 1000)
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)

    x_A = Lk
    x_q_end = Lk + a
    
    for i, x in enumerate(x_coords):
        V = 0
        M = 0
        
        # Section I: 0 <= x < Lk (Cantilever)
        if x < x_A:
            # Shear: Constant F1
            V = F1 
            # Moment: Linear -F1 * x (assuming positive moment causes tension on bottom)
            M = -F1 * x
            
        # Section II: Lk <= x < Lk + a (Between A and end of q)
        elif x < x_q_end:
            x_prime = x - x_A # Distance from A
            q_x = x_prime     # Length of distributed load currently active
            
            # V = F1 + Ay - q * x_prime
            V = F1 + Ay - q * x_prime
            
            # M = -F1 * x + Ay * x_prime - (q * x_prime) * (x_prime / 2)
            M = -F1 * x + Ay * x_prime - q * x_prime**2 / 2
            
        # Section III: Lk + a <= x <= Lt (Between end of q and B)
        else:
            x_prime = x - x_A # Distance from A
            
            # V = F1 + Ay - Q
            V = F1 + Ay - Q 
            
            # M = -F1 * x + Ay * x_prime - Q * (x_prime - x_Q) 
            # Note: x_prime - x_Q is the distance from the center of Q to the cut section
            M = -F1 * x + Ay * x_prime - Q * (x_prime - x_Q)
            
        V_values[i] = V
        M_values[i] = M

    return {"A_y": Ay, "B_y": By, "A_x": Ax}, x_coords, V_values, M_values

def solve_example_6(q, F1, Lk, L, a):
    # This remains the same as before.
    if L <= 0:
        st.error("Span length L must be greater than zero for Example 6.")
        return {"A": 0, "B": 0}, np.array([0]), np.array([0]), np.array([0])

    # 1. Calculate Support Reactions (Ay, By)
    By = (q * Lk * (Lk / 2) + F1 * (L - a)) / L
    Ay = q * Lk + F1 - By

    # 2. Calculate Shear (V) and Moment (M) Functions
    Lt = Lk + L
    x_coords = np.linspace(0, Lt, 1000)
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)

    x_A = Lk
    x_F1 = Lk + (L - a)
    
    for i, x in enumerate(x_coords):
        V = 0
        M = 0
        
        # Section I: 0 <= x < Lk (Cantilever)
        if x < x_A:
            V = -q * x
            M = -q * x**2 / 2
            
        # Section II: Lk <= x < x_F1 (Between A and F1)
        elif x < x_F1:
            V = -q * Lk + Ay
            M = -q * Lk * (x - Lk/2) + Ay * (x - Lk)
            
        # Section III: x_F1 <= x <= Lt (Between F1 and B)
        else:
            V = -q * Lk + Ay - F1
            M = -q * Lk * (x - Lk/2) + Ay * (x - Lk) - F1 * (x - x_F1)
            
        V_values[i] = V
        M_values[i] = M

    return {"A": Ay, "B": By}, x_coords, V_values, M_values

def solve_example_8(q, F3, h, L):
    """
    Solves Statically INDETERMINATE Beam (Example 8): Fixed support at A (x=L).
    Loads: Distributed load 'q' over span L, Horizontal point load 'F3' at x=0, distance 'h' above the beam.
    """
    if L <= 0:
        st.error("Span length L must be greater than zero for Example 8.")
        return {"A_y": 0, "A_x": 0, "M_A": 0}, np.array([0]), np.array([0]), np.array([0])

    st.warning("Example 8 is **Statically Indeterminate** (Fixed support). The solution uses known standard formulas for this case.")
    
    # 1. Calculate Support Reactions at A (x=L)
    Q = q * L # Total distributed load
    
    # Sum of Horizontal Forces (Ax)
    Ax = F3
    
    # Sum of Vertical Forces (Ay)
    Ay = Q
    
    # Sum of Moments (MA) - Clockwise Moment is negative, F3 causes a clockwise moment.
    # MA = (Q * L/2) - (F3 * h)
    # We define M_A as a resisting moment (positive tension on bottom)
    M_A = (q * L * L / 2) - (F3 * h) 
    
    # 2. Calculate Shear (V) and Moment (M) Functions (measured from left, x=0)
    x_coords = np.linspace(0, L, 1000)
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    # V(x) = F3 (Horizontal is ignored in V for vertical cut)
    # V(x) = q*L - q*x (If solving from the left, which is easier for plotting)
    # Since the fixed support is at A (x=L) and the loads are q (over L) and F3 (at x=0),
    # it's simpler to calculate V and M from the left end (x=0).
    
    for i, x in enumerate(x_coords):
        # Shear V(x) (Vertical component only): Constant 0, then Ay at x=L, then q*x is removed.
        # Since the vertical load is only q, and Ay = q*L, V must go from q*L to 0.
        V = Ay - q * x # V starts at Ay (at x=L) and decreases linearly.
        
        # Moment M(x):
        # M(x) = M_A + Ay*(L-x) - q*(L-x)^2 / 2 - F3*h (This is complex)
        
        # Simpler way (solving from the left, x=0):
        # M(x) = (-F3*h) + q*x^2/2 - Ay*x (where Ay is the reaction *at x=L*)
        # M(x) = -F3*h + q*x^2/2 (since reaction Ay is only at x=L)
        
        # Correctly solving from the left end (x=0) to fixed end (x=L):
        # V(x) = F3 (Horizontal, ignored)
        # Vertical Shear: V(x) = -q * x (V_left = 0, V_right = -q*L)
        V = -q * x
        
        # Moment M(x):
        # M(x) = (Moment due to F3 at x) + (Moment due to q at x)
        # M(x) = (F3 * h) - (q * x * x / 2)
        M = F3 * h - q * x**2 / 2
            
        V_values[i] = V
        M_values[i] = M

    return {"A_y": Ay, "A_x": Ax, "M_A": M_A}, x_coords, V_values, M_values


# --- UI Layout ---

st.title("🏗️ Structural Beam Analysis Tool")
st.markdown("---")

# --- Beam Selector and Image ---
selected_example = st.sidebar.radio(
    "Select Beam Example to Solve:",
    options=["Example 5", "Example 8", "Example 6", "Example 1"],
    index=0,
)

config = IMAGE_CONFIGS.get(selected_example, IMAGE_CONFIGS["Example 5"])

st.subheader(f"Solving: {selected_example}")
st.markdown(f"*{config['caption']}*")
# Display the visual representation of the solved beam
st.image(config['url'], caption=config['caption']) 

st.markdown("---")

# --- Dynamic Input Parameters and Solution Call ---

R_dict = {}
x_coords, V_values, M_values = np.array([0]), np.array([0]), np.array([0])

if selected_example == "Example 5":
    with st.sidebar:
        st.markdown("#### Example 5 Inputs")
        F1 = st.number_input("Point Load ($F_1$, kN)", value=15.0, min_value=0.0, step=1.0, key="F15")
        q = st.number_input("Distributed Load ($q$, kN/m)", value=8.0, min_value=0.0, step=1.0, key="q5")
        st.markdown("---")
        Lk = st.number_input("Cantilever Length ($L_k$, m)", value=1.0, min_value=0.1, step=0.5, key="Lk5")
        L = st.number_input("Main Span Length ($L$, m)", value=6.0, min_value=0.1, step=0.5, key="L5")
        a = st.number_input("Distributed Load Length ($a$, m)", value=4.0, min_value=0.0, max_value=L if L > 0 else 1.0, step=0.5, key="a5")
        
    R_dict, x_coords, V_values, M_values = solve_example_5(q, F1, Lk, L, a)

elif selected_example == "Example 8":
    with st.sidebar:
        st.markdown("#### Example 8 Inputs")
        F3 = st.number_input("Horizontal Load ($F_3$, kN)", value=10.0, min_value=0.0, step=1.0, key="F38")
        q = st.number_input("Distributed Load ($q$, kN/m)", value=5.0, min_value=0.0, step=1.0, key="q8")
        st.markdown("---")
        h = st.number_input("Load Offset Height ($h$, m)", value=0.5, min_value=0.0, step=0.1, key="h8")
        L = st.number_input("Span Length ($L$, m)", value=8.0, min_value=0.1, step=0.5, key="L8")
        
    R_dict, x_coords, V_values, M_values = solve_example_8(q, F3, h, L)

elif selected_example == "Example 6":
    with st.sidebar:
        st.markdown("#### Example 6 Inputs")
        q = st.number_input("Distributed Load ($q$, kN/m)", value=10.0, min_value=0.0, step=1.0, key="q6")
        F1 = st.number_input("Point Load ($F_1$, kN)", value=25.0, min_value=0.0, step=1.0, key="F16")
        st.markdown("---")
        Lk = st.number_input("Cantilever Length ($L_k$, m)", value=2.0, min_value=0.1, step=0.5, key="Lk6")
        L = st.number_input("Span Length ($L$, m)", value=5.0, min_value=0.1, step=0.5, key="L6")
        a = st.number_input("Distance 'a' from B to $F_1$ (m)", value=1.0, min_value=0.0, max_value=L if L > 0 else 1.0, step=0.5, key="a6")
        
    R_dict, x_coords, V_values, M_values = solve_example_6(q, F1, Lk, L, a)

elif selected_example == "Example 1":
    with st.sidebar:
        st.markdown("#### Example 1 Inputs")
        F1 = st.number_input("Point Load ($F_1$, kN)", value=10.0, min_value=0.0, step=1.0, key="F11")
        st.markdown("---")
        a = st.number_input("Distance 'a' (m) - Span BC length", value=4.0, min_value=0.1, step=0.5, key="a1")
        b = st.number_input("Distance 'b' (m) - F1 offset in Span AB", value=2.0, min_value=0.1, step=0.5, key="b1")
        
    # Call the solution function (which provides a simplified/placeholder output)
    R_dict, x_coords, V_values, M_values = solve_example_1(F1, a, b)


# --- Output Section ---

st.subheader("Solution: Support Reactions")

if selected_example == "Example 5":
    R_list = [
        f"| $A_y$ | **{R_dict.get('A_y', 0):,.2f}** |",
        f"| $B_y$ | **{R_dict.get('B_y', 0):,.2f}** |",
        f"| $A_x$ | **{R_dict.get('A_x', 0):,.2f}** |"
    ]
elif selected_example == "Example 8":
    R_list = [
        f"| $A_y$ | **{R_dict.get('A_y', 0):,.2f}** |",
        f"| $A_x$ | **{R_dict.get('A_x', 0):,.2f}** |",
        f"| $M_A$ (Moment) | **{R_dict.get('M_A', 0):,.2f}** |"
    ]
elif selected_example == "Example 1":
    st.markdown("""
        **Note on Indeterminacy:** For this two-span beam, the reactions below are based on a simplified model (see methodology below). 
        True reactions in indeterminate beams require solving a system of equations, but this provides a visual estimate.
    """)
    R_list = [
        f"| $A_y$ | **{R_dict.get('A', 0):,.2f}** |",
        f"| $B_y$ | **{R_dict.get('B', 0):,.2f}** |",
        f"| $C_y$ | **{R_dict.get('C', 0):,.2f}** |"
    ]
else: # Example 6
    R_list = [
        f"| $A_y$ | **{R_dict.get('A', 0):,.2f}** |",
        f"| $B_y$ | **{R_dict.get('B', 0):,.2f}** |"
    ]
    
st.markdown(f"""
| Reaction | Value (kN) |
| :---: | :---: |
{"\n".join(R_list)}
""")

st.markdown("---")

st.subheader("Shear Force (V) and Bending Moment (M) Diagrams")

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

# --- Explanation Section ---
st.markdown("---")
st.subheader("Solving Methodology")

if selected_example == "Example 5":
    st.markdown("""
        **Beam Type:** Statically Determinate (Hinge $A$ and Roller $B$).
        **Method:** Solution based on static equilibrium ($\Sigma M=0$, $\Sigma F_y=0$).
        1.  **Moment Equilibrium ($\Sigma M_A = 0$):** Used to calculate the reaction $B_y$.
        2.  **Vertical Force Equilibrium ($\Sigma F_y = 0$):** Used to calculate the reaction $A_y$.
        3.  **Horizontal Force Equilibrium ($\Sigma F_x = 0$):** Used to calculate the reaction $A_x$ (which is $0$ in this case).
        The Shear ($V$) and Moment ($M$) diagrams are constructed by analyzing the internal forces in three segments.
    """)
elif selected_example == "Example 8":
    st.markdown("""
        **Beam Type:** Statically Indeterminate (Fixed support at $A$).
        **Method:** The fixed support ($A$) introduces three unknown reactions ($A_x$, $A_y$, $M_A$). The solution uses standard formulas for a cantilever with a distributed load and an end moment ($F_3 \times h$).
        The Moment ($M$) and Shear ($V$) functions are derived from the resulting forces.
    """)
elif selected_example == "Example 1":
    st.markdown("""
        **Beam Type:** Statically Indeterminate (three supports, $A$, $B$, $C$).
        **Method:** For this complex case, the current script uses simplified **placeholder** values for reactions and diagrams to demonstrate the app's structure.
        **For accurate results:** Solving a statically indeterminate continuous beam requires advanced structural analysis methods (e.g., the Three Moment Equation or Moment Distribution Method) to determine the moment at the middle support ($B$).
    """)
elif selected_example == "Example 6":
    st.markdown("""
        **Beam Type:** Statically Determinate (two supports, $A$ and $B$).
        **Method:** The solution is based on the two fundamental equations of static equilibrium:
        1.  **Moment Equilibrium ($\Sigma M_A = 0$):** Used to calculate the reaction $B_y$.
        2.  **Vertical Force Equilibrium ($\Sigma F_y = 0$):** Used to calculate the reaction $A_y$.
        The Shear ($V$) and Moment ($M$) diagrams are then constructed by analyzing the internal forces in three separate segments of the beam.
    """)
