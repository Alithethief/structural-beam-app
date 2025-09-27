import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration for Streamlit Page ---
# Sets a wider layout for better visualization of plots
st.set_page_config(layout="wide", page_title="Beam Analysis Calculator (Case 6)")

def calculate_reactions_and_diagrams(q, F1, Lk, L, a):
    """
    Calculates support reactions (Ay, By) and the Shear (V) and Moment (M) 
    diagrams for the Statically Determinate Beam in Example 6 of the PDF.

    The beam is: Cantilever (Lk) -> Support A (Roller) -> Span (L) -> Support B (Roller).
    Loads: Distributed load 'q' on Lk, Point load 'F1' at Lk + (L-a) from left end.
    
    x=0 is the far left end of the beam.
    Support A is at x = Lk.
    Support B is at x = Lk + L.
    Force F1 is at x = Lk + L - a.
    Total length = Lk + L.
    """
    
    # 1. Check for valid lengths
    if L <= 0:
        st.error("Span length L must be greater than zero.")
        return 0, 0, np.array([0]), np.array([0]), np.array([0])

    # 2. Calculate Support Reactions (Ay, By)
    
    # Sum of Moments about A (Sigma M_A = 0). Clockwise is positive.
    # By * L = (q * Lk^2 / 2) + F1 * (L - a)
    try:
        # Calculate By
        By = (q * Lk**2 / 2 + F1 * (L - a)) / L
        
        # Sum of Vertical Forces (Sigma F_y = 0). Upward is positive.
        # -q * Lk + Ay + By - F1 = 0 => Ay = q * Lk + F1 - By
        Ay = q * Lk + F1 - By
        
    except ZeroDivisionError:
        st.error("Length L cannot be zero for reaction calculation.")
        return 0, 0, np.array([0]), np.array([0]), np.array([0])

    # 3. Calculate Shear (V) and Moment (M) Functions
    
    Lt = Lk + L  # Total length
    x_coords = np.linspace(0, Lt, 1000)
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)

    # Key points:
    x_A = Lk
    x_F1 = Lk + (L - a)
    
    for i, x in enumerate(x_coords):
        V = 0
        M = 0
        
        # Section I: 0 <= x < Lk (Cantilevered part with distributed load q)
        if x < x_A:
            V = -q * x
            M = -q * x**2 / 2
            
        # Section II: Lk <= x < x_F1 (Between A and F1)
        elif x < x_F1:
            # Shear: Net force from q + Ay
            V = -q * Lk + Ay
            
            # Moment: Distributed load moment + Ay moment
            M = -q * Lk * (x - Lk/2) + Ay * (x - Lk)
            
        # Section III: x_F1 <= x <= Lt (Between F1 and B)
        else:
            # Shear: Net force from q + Ay - F1
            V = -q * Lk + Ay - F1
            
            # Moment: Distributed load moment + Ay moment - F1 moment
            M = -q * Lk * (x - Lk/2) + Ay * (x - Lk) - F1 * (x - x_F1)
            
        V_values[i] = V
        M_values[i] = M

    return Ay, By, x_coords, V_values, M_values

# --- UI Layout ---

st.title("🏗️ Structural Beam Analysis App")
st.markdown("---")
st.header("Static Analysis of a Cantilevered Beam (Example 6)")

st.info("This application calculates the support reactions and plots the Shear Force (V) and Bending Moment (M) diagrams for the beam configuration shown in Example 6 of your provided document.")

# Display Beam Configuration Image Placeholder
st.markdown("### Beam Configuration")
st.image("https://placehold.co/700x150/0F79BD/FFFFFF?text=Beam+Configuration+Example+6:+Cantilever+Left+of+A,+Point+Load+F1+in+Span+AB", caption="Statically Determinate Beam: Cantilevered Left, Supported at A and B")

# Input Parameters in Sidebar
with st.sidebar:
    st.subheader("Input Parameters")
    
    st.markdown("#### Loads")
    q = st.number_input("Distributed Load ($q$, kN/m)", value=10.0, min_value=0.0, step=1.0)
    F1 = st.number_input("Point Load ($F_1$, kN)", value=25.0, min_value=0.0, step=1.0)
    
    st.markdown("#### Dimensions (m)")
    Lk = st.number_input("Cantilever Length ($L_k$, m)", value=2.0, min_value=0.1, step=0.5)
    L = st.number_input("Span Length ($L$, m)", value=5.0, min_value=0.1, step=0.5)
    a = st.number_input("Distance 'a' from B to $F_1$ (m)", value=1.0, min_value=0.0, max_value=L, step=0.5)
    
    st.markdown("---")
    st.markdown("**Instructions:** Adjust the values above and the diagrams will update instantly.")


# --- Calculation and Output ---

Ay, By, x_coords, V_values, M_values = calculate_reactions_and_diagrams(q, F1, Lk, L, a)

st.markdown("---")

st.subheader("Solution: Support Reactions")

st.markdown(f"""
The reactions are calculated using the equations of static equilibrium ($\Sigma F_y = 0$, $\Sigma M = 0$).

### Results:
| Reaction | Value (kN) |
| :---: | :---: |
| $A_y$ | **{Ay:,.2f}** |
| $B_y$ | **{By:,.2f}** |
""")

st.markdown("---")

st.subheader("Shear Force (V) and Bending Moment (M) Diagrams")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
Lt = Lk + L

# --- Shear Force Diagram (V) ---
ax1.plot(x_coords, V_values, label='$V(x)$', color='#1e56a0')
ax1.axhline(0, color='black', linewidth=0.5) 
ax1.axvline(Lk, color='gray', linestyle='--', linewidth=0.8, label='Support A') 
ax1.axvline(Lt, color='gray', linestyle='--', linewidth=0.8, label='Support B') 
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
ax2.axvline(Lk, color='gray', linestyle='--', linewidth=0.8, label='Support A') 
ax2.axvline(Lt, color='gray', linestyle='--', linewidth=0.8, label='Support B') 
ax2.set_title("Bending Moment Diagram $M(x)$", fontsize=16)
ax2.set_xlabel("x (m)")
ax2.set_ylabel("Bending Moment (kNm)")
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.fill_between(x_coords, M_values, 0, where=(M_values > 0), color='#2ca02c', alpha=0.2)
ax2.fill_between(x_coords, M_values, 0, where=(M_values < 0), color='#d62728', alpha=0.2)
ax2.legend(loc='upper right')


# Adjust layout and display plots
plt.tight_layout()
st.pyplot(fig)

# --- Explanation Section ---
st.markdown("---")
st.subheader("Solving Methodology (How the Code Works)")
st.markdown("""
To find the internal forces (Shear and Moment) in the beam:

1.  **Reactions ($\mathbf{A_y, B_y}$):** Calculated using static equilibrium equations:
    * **Moment Equilibrium ($\Sigma M_A = 0$):** This solves for $B_y$.
    * **Vertical Force Equilibrium ($\Sigma F_y = 0$):** This solves for $A_y$.
2.  **Sectional Analysis:** The beam is analytically divided into three segments:
    * **Segment I** ($0 < x < L_k$): Only the distributed load $q$ acts here. $V(x)$ is linear, and $M(x)$ is parabolic.
    * **Segment II** ($L_k < x < L_k + L - a$): Reactions $A_y$ and the distributed load $q$ are considered.
    * **Segment III** ($L_k + L - a < x < L_k + L$): Reactions $A_y, B_y$, distributed load $q$, and point load $F_1$ are all included in the force/moment calculations.
3.  **Plotting:** The code uses `numpy` to generate 1000 points along the beam's length and applies the correct $V(x)$ and $M(x)$ formula to each point, then uses `matplotlib` to plot the results clearly.
""")
