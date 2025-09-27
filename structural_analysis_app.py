import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Page Configuration ---
st.set_page_config(
    layout="wide", 
    page_title="Interactive Beam Analysis Tutorial",
    page_icon="🏗️"
)

# --- Custom CSS for Better Styling ---
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 700;
}
.step-header {
    background: linear-gradient(90deg, #2E86AB, #A23B72);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: 600;
    margin: 1rem 0;
}
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
.warning-box {
    background-color: #fff5ee;
    border-left: 5px solid #ff6347;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def get_force_components(F, alpha_deg):
    """Calculate vertical and horizontal components of an angled force."""
    alpha_rad = math.radians(alpha_deg)
    F_vertical = F * math.cos(alpha_rad)
    F_horizontal = F * math.sin(alpha_rad)
    return F_vertical, F_horizontal

def create_educational_diagram(x_coords, V_values, M_values, example_title):
    """Create educational diagrams with detailed annotations."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Shear Force Diagram
    ax1.plot(x_coords, V_values, 'b-', linewidth=3, label='V(x)')
    ax1.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)
    ax1.fill_between(x_coords, V_values, 0, where=(V_values > 0), 
                     color='lightblue', alpha=0.6, label='Positive V')
    ax1.fill_between(x_coords, V_values, 0, where=(V_values < 0), 
                     color='lightcoral', alpha=0.6, label='Negative V')
    ax1.set_title(f'Shear Force Diagram - {example_title}', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Distance x (m)', fontsize=12)
    ax1.set_ylabel('Shear Force V (kN)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add annotations for max/min values
    max_V = np.max(V_values)
    min_V = np.min(V_values)
    if abs(max_V) > 0.01:
        max_idx = np.argmax(V_values)
        ax1.annotate(f'Max V = {max_V:.2f} kN', 
                     xy=(x_coords[max_idx], max_V), 
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Bending Moment Diagram
    ax2.plot(x_coords, M_values, 'r-', linewidth=3, label='M(x)')
    ax2.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)
    ax2.fill_between(x_coords, M_values, 0, where=(M_values > 0), 
                     color='lightgreen', alpha=0.6, label='Positive M (Sagging)')
    ax2.fill_between(x_coords, M_values, 0, where=(M_values < 0), 
                     color='lightpink', alpha=0.6, label='Negative M (Hogging)')
    ax2.set_title('Bending Moment Diagram', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Distance x (m)', fontsize=12)
    ax2.set_ylabel('Bending Moment M (kNm)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add annotations for max/min values
    max_M = np.max(M_values)
    min_M = np.min(M_values)
    if abs(max_M) > 0.01:
        max_idx = np.argmax(M_values)
        ax2.annotate(f'Max M = {max_M:.2f} kNm', 
                     xy=(x_coords[max_idx], max_M), 
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    return fig

# --- Educational Solver Functions ---
def solve_simply_supported_beam_educational(F1, q, a, L, Lk):
    """
    Educational solver for simply supported beam with cantilever.
    Returns reactions, coordinates, forces, and step-by-step explanation.
    """
    if L <= 0:
        st.error("⚠️ Span length L must be positive!")
        return {}, np.array([0]), np.array([0]), np.array([0]), []
    
    # Step-by-step solution
    steps = []
    
    # Step 1: Identify loads and geometry
    L_total = L + Lk
    Q_distributed = q * Lk  # Total force from distributed load
    Q_position = L + Lk / 2  # Position of resultant force
    
    steps.append({
        "title": "Step 1: Analyze Loading and Geometry",
        "content": f"""
        - Total beam length: {L_total} m
        - Point load F₁ = {F1} kN at distance a = {a} m from A
        - Distributed load q = {q} kN/m over cantilever length {Lk} m
        - Resultant of distributed load: Q = q × Lk = {q} × {Lk} = {Q_distributed:.2f} kN
        - Position of resultant: x = {L} + {Lk}/2 = {Q_position:.2f} m from A
        """
    })
    
    # Step 2: Apply equilibrium equations
    # Sum of moments about A = 0
    By = (F1 * a + Q_distributed * Q_position) / L
    
    steps.append({
        "title": "Step 2: Apply Moment Equilibrium (∑M_A = 0)",
        "content": f"""
        Taking moments about support A (clockwise positive):
        
        **B_y × {L} - F₁ × {a} - Q × {Q_position:.2f} = 0**
        
        B_y × {L} = {F1} × {a} + {Q_distributed:.2f} × {Q_position:.2f}
        B_y × {L} = {F1 * a:.2f} + {Q_distributed * Q_position:.2f} = {F1 * a + Q_distributed * Q_position:.2f}
        
        **B_y = {By:.2f} kN ↑**
        """
    })
    
    # Sum of vertical forces = 0
    Ay = F1 + Q_distributed - By
    
    steps.append({
        "title": "Step 3: Apply Vertical Force Equilibrium (∑F_y = 0)",
        "content": f"""
        Sum of vertical forces = 0:
        
        **A_y + B_y - F₁ - Q = 0**
        
        A_y = F₁ + Q - B_y
        A_y = {F1} + {Q_distributed:.2f} - {By:.2f} = **{Ay:.2f} kN ↑**
        
        Horizontal equilibrium: A_x = 0 (no horizontal forces)
        """
    })
    
    # Step 4: Verification
    moment_check = By * L - F1 * a - Q_distributed * Q_position
    force_check = Ay + By - F1 - Q_distributed
    
    steps.append({
        "title": "Step 4: Verification",
        "content": f"""
        **Moment Check:** {By:.2f} × {L} - {F1} × {a} - {Q_distributed:.2f} × {Q_position:.2f} = {moment_check:.3f} ✓
        **Force Check:** {Ay:.2f} + {By:.2f} - {F1} - {Q_distributed:.2f} = {force_check:.3f} ✓
        """
    })
    
    # Calculate diagrams
    R_dict = {"A_y": Ay, "B_y": By, "A_x": 0}
    x_coords = np.linspace(0, L_total, 1000)
    
    # Shear force and bending moment calculation
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    for i, x in enumerate(x_coords):
        V = 0
        M = 0
        
        # Add reaction at A
        if x >= 0:
            V += Ay
            M += Ay * x
        
        # Subtract point load F1
        if x > a:
            V -= F1
            M -= F1 * (x - a)
        
        # Add reaction at B
        if x >= L:
            V += By
            M += By * (x - L)
        
        # Subtract distributed load
        if x > L:
            distributed_length = min(x - L, Lk)
            distributed_force = q * distributed_length
            V -= distributed_force
            M -= distributed_force * (distributed_length / 2)
        
        V_values[i] = V
        M_values[i] = M
    
    return R_dict, x_coords, V_values, M_values, steps

def solve_cantilever_beam_educational(q, F3, h, L):
    """Educational solver for cantilever beam with distributed and eccentric loads."""
    
    steps = []
    
    # Step 1: Problem setup
    Q = q * L  # Total distributed load
    
    steps.append({
        "title": "Step 1: Cantilever Beam Analysis Setup",
        "content": f"""
        **Cantilever beam fixed at right end (x = {L} m)**
        
        - Distributed load: q = {q} kN/m over length L = {L} m
        - Total vertical load: Q = q × L = {q} × {L} = {Q:.2f} kN
        - Eccentric horizontal load: F₃ = {F3} kN at height h = {h} m
        - Fixed support provides: vertical reaction, horizontal reaction, and moment reaction
        """
    })
    
    # Step 2: Equilibrium equations
    Ax = F3  # Horizontal equilibrium
    Ay = Q   # Vertical equilibrium
    M_A = Q * (L/2) - F3 * h  # Moment equilibrium about fixed support
    
    steps.append({
        "title": "Step 2: Static Equilibrium at Fixed Support",
        "content": f"""
        **Horizontal Equilibrium (∑F_x = 0):**
        A_x - F₃ = 0  →  **A_x = {Ax:.2f} kN →**
        
        **Vertical Equilibrium (∑F_y = 0):**
        A_y - Q = 0  →  **A_y = {Ay:.2f} kN ↑**
        
        **Moment Equilibrium about A (∑M_A = 0):**
        M_A + Q × (L/2) - F₃ × h = 0
        M_A = F₃ × h - Q × (L/2) = {F3} × {h} - {Q:.2f} × {L/2:.2f}
        **M_A = {M_A:.2f} kNm**
        """
    })
    
    # Step 3: Shear and moment functions
    steps.append({
        "title": "Step 3: Shear Force and Bending Moment Functions",
        "content": f"""
        **For cantilever beam (measured from free end at x = 0):**
        
        **Shear Force:** V(x) = -q × x
        - At x = 0: V = 0
        - At x = {L}: V = -{Q:.2f} kN
        
        **Bending Moment:** M(x) = -q × x²/2 + F₃ × h
        - At x = 0: M = {F3 * h:.2f} kNm (due to eccentric load)
        - At x = {L}: M = -{Q * L / 2:.2f} + {F3 * h:.2f} = {-Q * L / 2 + F3 * h:.2f} kNm
        """
    })
    
    # Calculate diagrams
    R_dict = {"A_y": Ay, "A_x": Ax, "M_A": M_A}
    x_coords = np.linspace(0, L, 1000)
    V_values = np.zeros_like(x_coords)
    M_values = np.zeros_like(x_coords)
    
    for i, x in enumerate(x_coords):
        V_values[i] = -q * x
        M_values[i] = -q * x**2 / 2 + F3 * h
    
    return R_dict, x_coords, V_values, M_values, steps

# --- Main Application ---
st.markdown('<h1 class="main-header">🏗️ Interactive Beam Analysis Tutorial</h1>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem; font-size: 1.2rem; color: #666;'>
Learn structural analysis through interactive examples with step-by-step solutions
</div>
""", unsafe_allow_html=True)

# Sidebar for problem selection
with st.sidebar:
    st.markdown("## 📚 Select Learning Module")
    
    problem_type = st.selectbox(
        "Choose beam type:",
        ["Simply Supported with Cantilever", "Fixed Cantilever Beam"],
        help="Select the type of beam problem you want to learn about"
    )

# Main content area
col1, col2 = st.columns([1, 1])

if problem_type == "Simply Supported with Cantilever":
    with col1:
        st.markdown('<div class="step-header">📊 Input Parameters</div>', unsafe_allow_html=True)
        
        F1 = st.slider("Point Load F₁ (kN)", 0.0, 50.0, 15.0, 2.5,
                      help="Concentrated load applied on the main span")
        q = st.slider("Distributed Load q (kN/m)", 0.0, 20.0, 8.0, 1.0,
                     help="Uniformly distributed load on the cantilever")
        a = st.slider("Distance to F₁ from A (m)", 0.5, 8.0, 3.0, 0.5,
                     help="Position of point load from left support")
        L = st.slider("Main Span Length L (m)", 2.0, 12.0, 6.0, 0.5,
                     help="Length between supports A and B")
        Lk = st.slider("Cantilever Length Lk (m)", 0.5, 4.0, 2.0, 0.25,
                      help="Length of cantilever extension")
        
        # Solve the problem
        R_dict, x_coords, V_values, M_values, steps = solve_simply_supported_beam_educational(
            F1, q, a, L, Lk)
        
        # Display results
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Calculated Reactions")
        st.write(f"**A_y = {R_dict.get('A_y', 0):.2f} kN ↑**")
        st.write(f"**B_y = {R_dict.get('B_y', 0):.2f} kN ↑**")
        st.write(f"**A_x = {R_dict.get('A_x', 0):.2f} kN →**")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="step-header">📈 Beam Diagrams</div>', unsafe_allow_html=True)
        
        if len(x_coords) > 1:
            fig = create_educational_diagram(x_coords, V_values, M_values, 
                                           "Simply Supported Beam with Cantilever")
            st.pyplot(fig)

elif problem_type == "Fixed Cantilever Beam":
    with col1:
        st.markdown('<div class="step-header">📊 Input Parameters</div>', unsafe_allow_html=True)
        
        q = st.slider("Distributed Load q (kN/m)", 0.0, 15.0, 6.0, 0.5,
                     help="Uniformly distributed vertical load")
        F3 = st.slider("Horizontal Load F₃ (kN)", 0.0, 10.0, 2.5, 0.25,
                      help="Horizontal eccentric load at free end")
        h = st.slider("Eccentricity h (m)", 0.0, 2.0, 0.9, 0.1,
                     help="Vertical distance of horizontal load from beam axis")
        L = st.slider("Beam Length L (m)", 2.0, 8.0, 4.9, 0.1,
                     help="Total length of cantilever beam")
        
        # Solve the problem
        R_dict, x_coords, V_values, M_values, steps = solve_cantilever_beam_educational(
            q, F3, h, L)
        
        # Display results
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Calculated Reactions")
        st.write(f"**A_y = {R_dict.get('A_y', 0):.2f} kN ↑**")
        st.write(f"**A_x = {R_dict.get('A_x', 0):.2f} kN →**")
        st.write(f"**M_A = {R_dict.get('M_A', 0):.2f} kNm**")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="step-header">📈 Beam Diagrams</div>', unsafe_allow_html=True)
        
        if len(x_coords) > 1:
            fig = create_educational_diagram(x_coords, V_values, M_values, 
                                           "Fixed Cantilever Beam")
            st.pyplot(fig)

# Step-by-step solution section
st.markdown("---")
st.markdown('<div class="step-header">📖 Step-by-Step Solution</div>', unsafe_allow_html=True)

for i, step in enumerate(steps):
    with st.expander(f"{step['title']}", expanded=(i == 0)):
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.markdown(step['content'])
        st.markdown('</div>', unsafe_allow_html=True)

# Educational notes
st.markdown("---")
st.markdown('<div class="step-header">📚 Key Learning Points</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🔧 Equilibrium Equations
    
    **For any beam, three equilibrium conditions must be satisfied:**
    
    1. **∑F_x = 0** (Horizontal force equilibrium)
    2. **∑F_y = 0** (Vertical force equilibrium)  
    3. **∑M = 0** (Moment equilibrium about any point)
    
    **💡 Tip:** Always choose the moment center wisely to eliminate unknown reactions and simplify calculations.
    """)

with col2:
    st.markdown("""
    ### 📊 Sign Conventions
    
    **Standard sign conventions used:**
    
    - **Forces:** ↑ Positive, ↓ Negative
    - **Moments:** ⟲ Counter-clockwise Positive
    - **Shear Force:** Upward on right face = Positive
    - **Bending Moment:** Sagging (tension bottom) = Positive
    
    **💡 Tip:** Consistent sign convention is crucial for correct results!
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 8px; margin-top: 2rem;'>
<h4>🎓 Practice Makes Perfect!</h4>
<p>Try different parameter combinations to understand how loads affect beam behavior. 
Pay attention to how reactions change and how the diagrams respond to different loading conditions.</p>
</div>
""", unsafe_allow_html=True)
