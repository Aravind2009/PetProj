"""Aerodynamics Explorer — an offline, simplified educational Streamlit app."""
import difflib
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np
import streamlit as st

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# GEMINI_API_KEY must be set in the environment to enable AI explanations.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


st.set_page_config(page_title="Aerodynamics Explorer", page_icon="✈️", layout="wide")

NAVY, BLUE, SKY, ORANGE, MINT, INK = "#102A43", "#1677B8", "#DFF3FF", "#F58B3B", "#DFF5E8", "#334E68"
TOPICS = {
    "lift": {"icon": "🪶", "title": "Lift", "beginner": "Lift is the upward aerodynamic force that can support an aircraft's weight. A wing creates a pressure pattern and turns air downward; the reaction gives the wing an upward force.", "intermediate": "A wing at positive angle of attack establishes circulation and a pressure difference. Its overall flow field leaves with downward momentum (downwash), so the aircraft receives an upward force.", "advanced": "For a finite wing, lift is the integrated pressure and shear force normal to the freestream. Bernoulli is consistent with the pressure field, but it is not a stand-alone cause of lift; circulation, boundary conditions, and momentum balance matter.", "equation": r"L = \tfrac12\rho V^2 S C_L", "example": "During takeoff, pilots increase speed and often deploy flaps to increase CL, letting the wing make required lift at a lower speed."},
    "drag": {"icon": "💨", "title": "Drag", "beginner": "Drag is air resistance. It points opposite the aircraft's motion, so engines must supply thrust to maintain speed.", "intermediate": "Parasite drag includes skin-friction and pressure/form drag. Induced drag is the price of making lift on a finite wing, created with downwash and wingtip vortices.", "advanced": "A simple drag polar is CD = CD0 + k CL². Near cruise, designers balance parasite drag, which tends to grow with speed squared, against induced drag, which falls as speed increases for a given lift.", "equation": r"D = \tfrac12\rho V^2 S C_D", "example": "Long, high-aspect-ratio glider wings reduce induced drag and improve glide ratio."},
    "bernoulli": {"icon": "🌬️", "title": "Bernoulli and pressure", "beginner": "In a smooth, steady flow, faster air is often associated with lower static pressure. This helps describe the pressure pattern around a wing.", "intermediate": "Bernoulli relates pressure, velocity, and elevation along a streamline for an ideal flow. A wing's shape and angle cause air to accelerate in some regions and create a pressure distribution.", "advanced": "Real flows include viscosity, separation, and energy losses, so Bernoulli has limits. It must be used with continuity, momentum conservation, and the full flow field—not as the complete explanation for lift.", "equation": r"p + \tfrac12\rho V^2 + \rho g h = \mathrm{constant}\;\; (ideal\ flow)", "example": "A pitot-static system uses total and static pressure to estimate airspeed."},
    "angle of attack": {"icon": "📐", "title": "Angle of attack", "beginner": "Angle of attack is the angle between a wing's chord line and the relative wind. A little more angle usually makes more lift, up to a limit.", "intermediate": "Increasing angle of attack raises CL approximately linearly at first. It also raises drag and can eventually make the boundary layer separate.", "advanced": "The critical angle depends on airfoil shape, Reynolds number, surface condition, and unsteady effects. Stall is an airflow-separation event, not simply a low-speed event.", "equation": r"C_L \approx C_{L0} + a\alpha\;\; (pre\text{-}stall)", "example": "A pilot pitches up to raise angle of attack during flare, but avoids exceeding the critical angle."},
    "stall": {"icon": "⚠️", "title": "Stall and separation", "beginner": "A stall occurs when airflow separates from much of a wing. Lift falls and drag rises sharply; it can happen at many speeds if angle of attack is too high.", "intermediate": "An adverse pressure gradient slows air near the surface. If the boundary layer loses too much energy, it reverses locally and separates from the wing.", "advanced": "Stall behavior can be gradual or abrupt and changes with Reynolds number, roughness, sweep, control deflection, and unsteady motion. Real aircraft use margins, warning systems, and recovery procedures.", "equation": r"\alpha > \alpha_{critical} \Rightarrow \mathrm{separation\ likely}", "example": "Lowering the nose reduces angle of attack and is the first step of a basic stall recovery."},
    "boundary layer": {"icon": "🌊", "title": "Boundary layer and Reynolds number", "beginner": "Air touching a wing is slowed by friction. The thin region where speed changes from zero at the surface to the outside-flow speed is the boundary layer.", "intermediate": "A laminar boundary layer is orderly; a turbulent one mixes more strongly and has more skin friction, but can resist separation better. Reynolds number helps compare the influence of inertia and viscosity.", "advanced": "Transition and separation depend on pressure gradient, surface roughness, free-stream turbulence, and geometry. Reynolds number alone does not uniquely predict the flow state.", "equation": r"Re = \frac{\rho V c}{\mu}", "example": "Small model aircraft often operate at lower Reynolds numbers than full-size aircraft, so their airfoils behave differently."},
    "wing geometry": {"icon": "🪽", "title": "Wings, vortices, and control", "beginner": "Wing area helps determine how much lift is available. Ailerons roll the aircraft, the elevator pitches it, and the rudder yaws it.", "intermediate": "A finite wing has tip vortices and downwash, producing induced drag. Greater aspect ratio generally improves efficiency at lift-making conditions.", "advanced": "Sweep can delay compressibility effects, while flaps and slats reshape the high-lift system. Static stability depends on the relationship between center of gravity, aerodynamic center, and tail forces.", "equation": r"AR = \frac{b^2}{S},\quad C_{D_i} \approx \frac{C_L^2}{\pi e AR}", "example": "Airliners use flaps for takeoff and landing so they can create more lift at lower speeds."},
    "compressibility": {"icon": "🚀", "title": "Mach and compressibility", "beginner": "At high speed, air can compress noticeably. Mach number compares speed to the local speed of sound.", "intermediate": "Subsonic flow is usually below Mach 0.8. Near Mach 1, local supersonic pockets and shock waves can increase drag; swept wings help manage this regime.", "advanced": "Shock waves are thin regions of abrupt pressure, temperature, and velocity change. Transonic aerodynamics needs more detailed methods than this app's low-speed models.", "equation": r"M = \frac{V}{a}", "example": "A typical jet cruises near Mach 0.78–0.85 to balance time, fuel use, and transonic drag."},
}

# Curated offline reference. These are grouped deliberately: many terms are
# linked parts of one physical model rather than isolated facts.
REFERENCE_SECTIONS = {
    "Core physical forces & moments": {
        "terms": "Lift · Drag · Thrust · Weight · Pitching moment · Rolling moment · Yawing moment · Center of pressure · Aerodynamic center",
        "theory": "An aircraft in flight experiences lift, drag, thrust, and weight. Lift and drag are aerodynamic force components defined relative to the local relative wind; thrust and weight come from propulsion and gravity. Because forces act at locations, they also make moments: elevator-related pitching moment about the lateral axis, aileron-related rolling moment about the longitudinal axis, and rudder-related yawing moment about the vertical axis. The center of pressure is the location of the resultant aerodynamic force and can move with angle of attack. The aerodynamic center is a more useful reference location where pitching moment changes little with angle of attack (near the quarter-chord for a subsonic thin airfoil).",
    },
    "Fluid properties & conservation laws": {
        "terms": "Air density · Static pressure · Dynamic pressure · Total (stagnation) pressure · Temperature · Viscosity · Speed of sound · Continuity · Momentum · Energy",
        "theory": "Density affects how much force a given airflow can produce. Static pressure is the thermodynamic pressure of the air; dynamic pressure q = ½ρV² represents the kinetic-energy-per-volume scale used in aircraft loads. In an ideal, lossless deceleration, total (stagnation) pressure combines static and dynamic pressure. Continuity expresses conservation of mass: for steady incompressible flow, area times velocity is constant along a streamtube. Momentum conservation connects a wing's downward deflection of air to its upward force. Energy conservation motivates Bernoulli only under restrictive assumptions; viscosity, heat transfer, shocks, and separation add losses or invalidate simple forms.",
    },
    "Fundamental principles": {
        "terms": "Bernoulli's principle · Newton's laws · Venturi effect · Kutta–Joukowski theorem · Circulation · D'Alembert's paradox · Potential flow",
        "theory": "Bernoulli describes a pressure–speed relationship along an ideal streamline; it is one view of a wing's pressure field, not the whole cause of lift. Newton's laws explain the same lift through the net downward momentum imparted to air. Circulation is a mathematical measure of flow around an airfoil; the Kutta–Joukowski result connects circulation to lift in two-dimensional ideal flow. The Kutta condition selects the physically observed circulation at a sharp trailing edge. Potential flow neglects viscosity and predicts zero drag on a body (D'Alembert's paradox), showing why real drag requires viscous boundary-layer and separation physics. A Venturi is a constriction where a flow may speed up as area decreases; it is not a universal 'suction' explanation.",
    },
    "Wing & airfoil geometry": {
        "terms": "Chord line · Camber · Leading edge · Trailing edge · Angle of attack · Angle of incidence · Aspect ratio · Span · Taper ratio · Sweep · Dihedral/anhedral · Wing loading",
        "theory": "The chord line joins an airfoil's leading and trailing edges. Camber is its curvature; a cambered airfoil can create positive lift at zero geometric angle of attack, while a symmetric airfoil is often used where predictable inverted behavior matters. Angle of attack is measured from chord to relative wind, whereas incidence is the fixed mounting angle relative to the fuselage. Aspect ratio is span squared divided by area; higher values generally reduce induced drag. Taper changes chord along the span, sweep helps with high-speed compressibility, and dihedral promotes roll stability while anhedral reduces it. Wing loading is aircraft weight divided by wing area and helps set characteristic takeoff and stall speeds.",
    },
    "Boundary layer & viscous flow": {
        "terms": "Boundary-layer thickness · Skin-friction drag · Laminar flow · Turbulent flow · Separation · Stall · Reynolds number · Transition point · Viscous dissipation",
        "theory": "The no-slip condition makes air velocity zero at a surface. The resulting boundary layer grows downstream and produces skin-friction drag. Laminar flow has smoother layered motion; turbulent flow has mixing and usually higher skin friction, yet it can bring high-momentum air toward the surface and delay separation. Transition is the change between the two states and depends on Reynolds number, roughness, pressure gradients, and disturbances. Separation occurs when near-surface flow cannot overcome an adverse pressure gradient; widespread separation produces aerodynamic stall. Viscous dissipation converts organized flow energy into heat, so real flows have losses absent from ideal equations.",
    },
    "Subsonic vortex dynamics": {
        "terms": "Downwash · Upwash · Wingtip vortices · Induced drag · Ground effect · Downstream wake",
        "theory": "A finite wing has higher pressure below and lower pressure above, so air curls around its tips and forms trailing vortices. Their induced velocity tilts the local relative wind downward behind the wing (downwash); an observer ahead may see upwash. Tilting the lift vector rearward creates induced drag, especially at high lift and low speed. The wake carries momentum and vorticity downstream. Near the ground, the vortex/downwash pattern is constrained; induced drag can decrease and lift characteristics change—this is ground effect.",
    },
    "Compressible & high-speed flow": {
        "terms": "Mach number · Subsonic · Transonic · Supersonic · Hypersonic · Isentropic flow · Sound speed · Normal/oblique shocks · Expansion fans · Wave drag · Sonic boom · Critical Mach number",
        "theory": "Mach number is speed divided by local speed of sound, which changes mainly with temperature. Compressibility effects become important well before Mach 1. Subsonic flow is commonly below about M 0.8; transonic flow has both subsonic and supersonic regions and may form shocks. A normal shock is perpendicular to the flow and produces a large irreversible loss; an oblique shock is angled and can be weaker. An expansion fan turns supersonic flow outward and accelerates it. Isentropic relations only apply where compression/expansion is smooth and reversible, not across shocks. Wave drag rises near the critical Mach number; shock patterns can contribute to a sonic boom. Hypersonic flow adds strong temperature and real-gas effects beyond this app's low-speed models.",
    },
    "Stability, control & flight dynamics": {
        "terms": "Static stability · Dynamic stability · Longitudinal/lateral/vertical axes · Trim · Dihedral effect · Adverse yaw · Dutch roll · Phugoid",
        "theory": "Static stability asks whether a small disturbance initially produces a restoring tendency; dynamic stability asks how the resulting motion evolves in time. Pitch, roll, and yaw are rotations about the lateral, longitudinal, and vertical axes. Trim is an equilibrium condition with net forces and moments balanced for a chosen flight state. Dihedral produces a roll-restoring tendency in sideslip, while aileron deflection can create adverse yaw from unequal drag. Dutch roll is a coupled yaw–roll oscillation; a phugoid is a slow exchange between airspeed and altitude. Real stability depends on mass distribution, tails, sweep, damping, and control-system design.",
    },
    "Experimental & analysis tools": {
        "terms": "Wind tunnels · CFD · Navier–Stokes equations · Euler equations · Prandtl lifting-line theory · Schlieren photography",
        "theory": "Wind tunnels measure forces, pressures, and flow structures on models, with careful attention to Reynolds-number and blockage differences. Computational fluid dynamics (CFD) solves approximations to governing equations on a mesh. Navier–Stokes equations include viscosity; Euler equations neglect it and cannot predict viscous drag or separation alone. Prandtl lifting-line theory is a useful finite-wing, low-speed estimate of induced drag and downwash, but it is not a full nonlinear stall model. Schlieren photography visualizes density gradients, especially useful for shock waves and compressible flow.",
    },
}

EXTRA_KEYWORDS = {
    "thrust":"lift", "weight":"lift", "pitching moment":"lift", "rolling moment":"wing geometry", "yawing moment":"wing geometry", "center of pressure":"lift", "aerodynamic center":"lift",
    "static pressure":"bernoulli", "total pressure":"bernoulli", "stagnation":"bernoulli", "temperature":"compressibility", "viscosity":"boundary layer", "speed of sound":"compressibility", "energy":"bernoulli",
    "venturi":"bernoulli", "kutta":"lift", "circulation":"lift", "d'alembert":"drag", "potential flow":"drag",
    "chord":"angle of attack", "camber":"wing geometry", "leading edge":"wing geometry", "trailing edge":"wing geometry", "incidence":"angle of attack", "span":"wing geometry", "taper":"wing geometry", "sweep":"compressibility", "dihedral":"wing geometry", "wing loading":"wing geometry",
    "skin friction":"boundary layer", "laminar":"boundary layer", "turbulent":"boundary layer", "transition":"boundary layer", "dissipation":"boundary layer",
    "downwash":"wing geometry", "upwash":"wing geometry", "wingtip":"wing geometry", "ground effect":"wing geometry", "wake":"wing geometry",
    "subsonic":"compressibility", "transonic":"compressibility", "supersonic":"compressibility", "hypersonic":"compressibility", "isentropic":"compressibility", "normal shock":"compressibility", "oblique shock":"compressibility", "expansion fan":"compressibility", "wave drag":"compressibility", "sonic boom":"compressibility", "critical mach":"compressibility",
    "static stability":"wing geometry", "dynamic stability":"wing geometry", "trim":"wing geometry", "adverse yaw":"wing geometry", "dutch roll":"wing geometry", "phugoid":"wing geometry",
    "wind tunnel":"wing geometry", "cfd":"wing geometry", "navier":"boundary layer", "euler equations":"bernoulli", "lifting-line":"wing geometry", "schlieren":"compressibility",
}


def explain(query, level):
    text = query.lower().strip()
    aliases = {"reynolds": "boundary layer", "airfoil": "wing geometry", "vortex": "wing geometry", "induced drag": "wing geometry", "flap": "wing geometry", "slat": "wing geometry", "aileron": "wing geometry", "elevator": "wing geometry", "rudder": "wing geometry", "mach": "compressibility", "shock": "compressibility", "density": "bernoulli", "dynamic pressure": "bernoulli", "continuity": "bernoulli", "newton": "lift", "momentum": "lift", "relative wind": "angle of attack", **EXTRA_KEYWORDS}
    if text in TOPICS: return text, None
    for word, topic in aliases.items():
        if word in text: return topic, None
    matches = difflib.get_close_matches(text, list(TOPICS) + list(aliases), n=1, cutoff=0.45)
    if matches:
        return aliases.get(matches[0], matches[0]), f"Showing the closest built-in topic for **{matches[0]}**."
    return None, None


def aero_values(rho, speed, area, cl, cd0, ar):
    q = 0.5 * rho * speed**2
    induced_cd = cl**2 / (math.pi * 0.82 * ar)
    cd = cd0 + induced_cd
    return q, q * area * cl, q * area * cd, cd, induced_cd


def ai_explanation(question, level):
    """Return a concise educational answer, without storing the response."""
    if not GEMINI_API_KEY or genai is None:
        return None
    instructions = (
        "You are a careful aerodynamics tutor. Answer the user's question in "
        f"a {level} level. Use clear SI units where relevant. Explain physical "
        "intuition, define important terms, and include an equation only when it "
        "helps. State that simplified equations are approximations when appropriate. "
        "Do not give aircraft operating, flight-safety, certification, or design advice. "
        "Do not claim Bernoulli alone explains lift. Keep the answer under 500 words."
    )
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=instructions,
            max_output_tokens=1_200,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    )
    return response.text


def chart_style(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor("#FBFDFF"); ax.set_title(title, loc="left", color=NAVY, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=.18); ax.spines[["top", "right"]].set_visible(False)


def airfoil_points(angle=0, camber=.04):
    x = np.linspace(0, 1, 220); t = .12 * np.sin(np.pi*x)**.8; yc = camber * np.sin(np.pi*x)
    theta = np.deg2rad(angle)
    rotate = lambda y: (x*np.cos(theta)-y*np.sin(theta), x*np.sin(theta)+y*np.cos(theta))
    return x, rotate(yc+t), rotate(yc-t)


def plot_airfoil_flow(aoa, speed, separated=False):
    fig, ax = plt.subplots(figsize=(8, 4.5)); x, upper, lower = airfoil_points(aoa)
    for y0 in np.linspace(-.42, .42, 9):
        xx = np.linspace(-.4, 1.35, 100)
        bump = (.11 * np.exp(-((xx-.45)/.35)**2) * (1 if y0 > 0 else -1) * min(aoa/14, 1))
        if separated and y0 > 0: bump *= np.where(xx > .55, .2, 1)
        yy = y0 + bump
        ax.plot(xx, yy, color=BLUE if y0 != 0 else "#79BFE1", alpha=.62, lw=1.25)
        ax.annotate("", (xx[-1], yy[-1]), (xx[-2], yy[-2]), arrowprops={"arrowstyle":"->", "color":BLUE, "alpha":.5})
    ax.fill_between(x, lower[1], upper[1], color="#E6F4FA", zorder=3); ax.plot(*upper, color=NAVY, lw=2.4, zorder=4); ax.plot(*lower, color=NAVY, lw=2.4, zorder=4)
    ax.annotate("Lift", (.55, .65), (.55, .15), color=BLUE, ha="center", arrowprops={"arrowstyle":"->", "lw":2.5,"color":BLUE})
    ax.annotate("Drag", (1.25, -.05), (.72, -.05), color=ORANGE, va="center", arrowprops={"arrowstyle":"->","lw":2.5,"color":ORANGE})
    if separated: ax.text(.67, .43, "Illustrative\nseparation", color=ORANGE, weight="bold")
    ax.set_xlim(-.45, 1.45); ax.set_ylim(-.65, .85); ax.axis("off"); ax.set_title(f"Simplified flow at {aoa:.0f}° angle of attack • {speed:.0f} m/s", loc="left", color=NAVY, fontweight="bold"); fig.tight_layout(); return fig


def plot_polar(aoa, cl, cd0, ar):
    angles = np.linspace(-6, 22, 160); cls = np.where(angles <= 15, .1*(angles+2), 1.7-.13*(angles-15)); cds = cd0 + cls**2/(math.pi*.82*ar)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6)); a.plot(angles, cls, color=BLUE, lw=2.7, label="Lift coefficient"); a.axvline(15, ls="--", color=ORANGE, label="Approx. stall"); a.scatter([aoa], [cl], color=ORANGE, zorder=4); chart_style(a,"Lift curve (simplified)","Angle of attack (°)","CL (–)"); a.legend(frameon=False, fontsize=8)
    b.plot(cds, cls, color=NAVY, lw=2.7); b.scatter([cd0+cl**2/(math.pi*.82*ar)], [cl], color=ORANGE, zorder=4); chart_style(b,"Drag polar", "CD (–)","CL (–)"); fig.tight_layout(); return fig


def plot_boundary(reynolds):
    x = np.linspace(0, 1, 200); lam = .04+.21*np.sqrt(x); turb = .07+.34*np.sqrt(x)
    fig, ax = plt.subplots(figsize=(8, 3.8)); ax.fill_between(x, 0, lam, color=SKY, label="Laminar: thinner, orderly"); ax.fill_between(x, 0, turb, color=ORANGE, alpha=.20, label="Turbulent: fuller, mixed")
    ax.plot(x, lam, color=BLUE, lw=2); ax.plot(x, turb, color=ORANGE, lw=2); ax.plot([0,1],[0,0],color=NAVY,lw=3,label="Wing surface")
    chart_style(ax, f"Illustrative boundary layers • Re = {reynolds:,.0f}", "Distance along surface (m, normalized)", "Boundary-layer thickness (normalized)"); ax.legend(frameon=False, ncol=3, fontsize=8); fig.tight_layout(); return fig


def plot_controls(elevator, aileron, rudder):
    fig, ax = plt.subplots(figsize=(8, 4.3)); ax.add_patch(Polygon([(0,0),(4,.55),(6,0),(4,-.55)], closed=True, fc=SKY, ec=NAVY, lw=2)); ax.plot([3.7,4.6],[.52,.52+elevator*.02],color=ORANGE,lw=5,label="Elevator")
    ax.plot([2,2],[0,1.8],color=NAVY,lw=3); ax.plot([.7,3.3],[.9,.9+aileron*.025],color=BLUE,lw=6,label="Aileron pair")
    ax.plot([4.55,4.8],[.12,.12+rudder*.025],color=MINT,lw=6,label="Rudder")
    ax.annotate("Pitch",(5.6,1.45),(4.3,.8),arrowprops={"arrowstyle":"->","color":ORANGE},color=ORANGE); ax.annotate("Roll",(.2,1.65),(1.4,1.05),arrowprops={"arrowstyle":"->","color":BLUE},color=BLUE); ax.annotate("Yaw",(6.2,-.8),(4.8,.1),arrowprops={"arrowstyle":"->","color":"#23855A"},color="#23855A")
    ax.set_xlim(-.2,6.8); ax.set_ylim(-1.2,2.1); ax.axis("off"); ax.legend(frameon=False,loc="lower left"); ax.set_title("Simplified control-surface directions",loc="left",color=NAVY,fontweight="bold"); fig.tight_layout(); return fig


def plot_drag_force():
    fig, ax = plt.subplots(figsize=(8, 4)); ax.add_patch(plt.Circle((0, 0), .52, fc=SKY, ec=NAVY, lw=2))
    ax.annotate("Motion / thrust", (1.45, 0), (.55, 0), color=BLUE, va="center", arrowprops={"arrowstyle":"->", "color":BLUE, "lw":3})
    ax.annotate("Drag", (-1.35, 0), (-.55, 0), color=ORANGE, va="center", arrowprops={"arrowstyle":"->", "color":ORANGE, "lw":3})
    ax.text(0,0,"Aircraft\nshape",ha="center",va="center",color=NAVY); ax.set_xlim(-1.8,1.8); ax.set_ylim(-.9,.9); ax.axis("off"); ax.set_title("Drag opposes motion",loc="left",color=NAVY,fontweight="bold"); fig.tight_layout(); return fig


def plot_bernoulli_relation():
    velocity=np.linspace(5,100,150); pressure=.5*1.225*(100**2-velocity**2)
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(velocity,pressure,color=BLUE,lw=3,label="Idealized static-pressure change"); ax.fill_between(velocity,pressure,color=SKY); chart_style(ax,"Faster flow and lower static pressure (idealized)","Flow speed (m/s)","Relative pressure change (Pa)"); ax.legend(frameon=False); fig.tight_layout(); return fig


def topic_card(topic, level):
    item = TOPICS[topic]
    st.subheader(f"{item['icon']} {item['title']}")
    st.info(item[level])
    st.latex(item["equation"])
    st.caption("Practical example: " + item["example"])
    related = [name for name, section in REFERENCE_SECTIONS.items() if topic.split()[0] in section["theory"].lower()]
    if related:
        with st.expander("Go deeper: connected theory"):
            st.write(REFERENCE_SECTIONS[related[0]]["theory"])


st.markdown("""<style>
.block-container {max-width: 1400px; padding-top: 1.6rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#F2FAFF 0%,#FFFFFF 100%);}
h1,h2,h3 {color:#102A43;} .hero {padding:1.65rem 2rem; border-radius:18px; background:linear-gradient(115deg,#102A43,#1677B8); color:white; margin-bottom:1.25rem; box-shadow:0 10px 30px #102a4320;}
.hero h1 {color:white; margin:0;} .hero p {margin:.35rem 0 0; color:#DFF3FF; font-size:1.08rem;}
.stButton>button {border-radius:9px; border:1px solid #B9DFF2; color:#102A43; font-weight:600;}
[data-testid="stMetric"] {background:#F7FBFD; border:1px solid #DDECF3; border-radius:12px; padding:.6rem;}
</style>""", unsafe_allow_html=True)

for key, default in {"search": "", "topic": None, "reset_nonce": 0}.items(): st.session_state.setdefault(key, default)

with st.sidebar:
    st.title("✈️ Flight desk")
    st.caption("An offline, simplified visual learning tool.")
    level = st.radio("Explanation level", ["beginner", "intermediate", "advanced"], format_func=str.title, help="Changes the depth of written explanations.")
    st.divider(); st.subheader("Quick topics")
    for name, item in TOPICS.items():
        if st.button(f"{item['icon']} {item['title']}", key=f"pick_{name}", use_container_width=True):
            st.session_state.topic = name; st.session_state.search = name
    st.divider(); st.caption("SI units are used throughout. Models are illustrative, not flight-planning tools.")

st.markdown("""<div class="hero"><h1>✈️ Aerodynamics Explorer</h1><p>See the forces, test the equations, and connect the diagrams to real aircraft behavior.</p></div>""", unsafe_allow_html=True)
learn_tab, lab_tab, flow_tab, controls_tab, reference_tab, ask_tab = st.tabs(["📚 Learn", "🧪 Flight lab", "🌊 Flow lab", "🎛️ Controls & wing", "📖 Theory reference", "💬 Explain any concept"])

with learn_tab:
    query = st.text_input("Search a concept or ask a short question", value=st.session_state.search, placeholder="Try: why does a wing lift?", help="Offline keyword matching; no data leaves this app.")
    c1, c2 = st.columns([1, 4]);
    with c1: search_clicked = st.button("Explain", use_container_width=True)
    with c2:
        if st.button("Reset selection", use_container_width=False): st.session_state.topic=None; st.session_state.search=""
    if search_clicked: st.session_state.topic, suggestion = explain(query, level); st.session_state.search=query
    else: suggestion = None
    topic = st.session_state.topic
    if not topic:
        st.info("👋 Start with **lift**, or select a topic from the sidebar. Each lab is interactive and uses SI units.")
        st.columns(3)[0].metric("Topics", len(TOPICS)); st.columns(3)[1].metric("Equations", "5+"); st.columns(3)[2].metric("Mode", level.title())
    elif topic:
        if suggestion: st.info(suggestion)
        left, right = st.columns([.9, 1.35], gap="large")
        with left: topic_card(topic, level); st.warning("Simplified educational model: real aerodynamic behavior depends on geometry, turbulence, altitude, and operating conditions.")
        with right:
            if topic in {"lift", "angle of attack", "stall"}: st.pyplot(plot_airfoil_flow(8, 45, topic == "stall"), use_container_width=True)
            elif topic == "drag": st.pyplot(plot_drag_force(), use_container_width=True)
            elif topic == "bernoulli": st.pyplot(plot_bernoulli_relation(), use_container_width=True)
            elif topic == "boundary layer": st.pyplot(plot_boundary(600_000), use_container_width=True)
            elif topic == "wing geometry": st.pyplot(plot_polar(6,.8,.022,8), use_container_width=True)
            elif topic == "compressibility":
                mach=np.linspace(0,1.3,100); fig,ax=plt.subplots(figsize=(8,4)); ax.plot(mach,1+np.maximum(0,mach-.75)**2*18,color=ORANGE,lw=3,label="Illustrative wave-drag rise"); chart_style(ax,"Compressibility illustration","Mach number (–)","Relative drag (–)"); ax.legend(frameon=False); fig.tight_layout(); st.pyplot(fig,use_container_width=True)
            else: st.pyplot(plot_polar(6,.8,.022,8), use_container_width=True)
            st.caption("Diagram is illustrative: it shows relationships, not a CFD or wind-tunnel result.")

with lab_tab:
    st.subheader("🧪 Lift, drag, and performance lab"); st.write("**Objective:** change one flight condition at a time and observe why forces change.")
    if st.button("↺ Reset flight lab", help="Restore the lab's original example values."):
        for key, value in {"lab_rho": 1.225, "lab_speed": 55.0, "lab_area": 16.0, "lab_aoa": 6, "lab_cd0": .025, "lab_ar": 8.0}.items(): st.session_state[key] = value
    a,b,c,d = st.columns(4)
    with a: rho=st.number_input("Air density, ρ (kg/m³)",.3,1.5,1.225,.025,key="lab_rho",help="Lower density approximates higher altitude.")
    with b: speed=st.number_input("Airspeed, V (m/s)",1.0,250.0,55.0,1.0,key="lab_speed")
    with c: area=st.number_input("Wing area, S (m²)",1.0,500.0,16.0,1.0,key="lab_area")
    with d: aoa=st.slider("Angle of attack, α (°)",-6,22,6,key="lab_aoa")
    e,f,g=st.columns(3)
    with e: cd0=st.number_input("Zero-lift drag coefficient, CD₀ (–)",.005,.15,.025,.001,key="lab_cd0",format="%.3f")
    with f: ar=st.number_input("Aspect ratio, AR (–)",2.0,25.0,8.0,.5,key="lab_ar")
    with g: cl=max(-.4, min(1.7, .1*(aoa+2) if aoa <= 15 else 1.7-.13*(aoa-15))); st.metric("Estimated CL (–)",f"{cl:.2f}",help="A deliberately simplified lift curve with a 15° stall point.")
    q,lift,drag,cd,induced=aero_values(rho,speed,area,cl,cd0,ar)
    m1,m2,m3,m4=st.columns(4); m1.metric("Dynamic pressure q",f"{q:,.0f} Pa"); m2.metric("Lift L",f"{lift:,.0f} N"); m3.metric("Drag D",f"{drag:,.0f} N"); m4.metric("L/D",f"{lift/drag:.1f}" if drag>0 else "—")
    st.latex(r"q=\tfrac12\rho V^2\qquad L=qSC_L\qquad D=qSC_D")
    st.caption("ρ = density, V = airspeed, S = wing area, CL/CD = coefficients. Speed has a squared effect: doubling V roughly quadruples q, lift, and drag if coefficients stay fixed.")
    x,y=st.columns([1.2,1]);
    with x: st.pyplot(plot_airfoil_flow(aoa,speed,aoa>15),use_container_width=True)
    with y: st.pyplot(plot_polar(aoa,cl,cd0,ar),use_container_width=True)
    st.info("Observation: increasing angle of attack raises lift initially, but the simplified model marks separation/stall beyond 15°. Larger aspect ratio lowers the induced-drag part of CD.")

with flow_tab:
    st.subheader("🌊 Boundary layer, pressure, and flow regime"); st.write("**Objective:** compare orderly and mixed near-surface flow, then connect speed and chord to Reynolds number.")
    a,b,c=st.columns(3)
    with a: flow_speed=st.number_input("Flow speed (m/s)",.1,200.,35.,1.)
    with b: chord=st.number_input("Chord length (m)",.02,10.,1.2,.02)
    with c: viscosity=st.number_input("Dynamic viscosity, μ (Pa·s)",1e-6,1e-3,1.81e-5,format="%.2e")
    re=rho*flow_speed*chord/viscosity
    st.metric("Reynolds number, Re (–)",f"{re:,.0f}"); st.latex(r"Re=\frac{\rho Vc}{\mu}")
    l,r=st.columns([1.2,1]);
    with l: st.pyplot(plot_boundary(re),use_container_width=True)
    with r:
        v=np.linspace(5,flow_speed*1.35,100); p=.5*rho*((flow_speed*1.35)**2-v**2)
        fig,ax=plt.subplots(figsize=(6,3.8)); ax.plot(v,p,color=BLUE,lw=2.8); chart_style(ax,"Idealized pressure–velocity relation","Velocity (m/s)","Relative pressure change (Pa)"); fig.tight_layout(); st.pyplot(fig,use_container_width=True)
    st.info("A turbulent boundary layer has more skin friction but may stay attached longer. The pressure chart uses an idealized Bernoulli relationship; viscosity and losses in real air make the flow more complex.")

with controls_tab:
    st.subheader("🎛️ Wing geometry and aircraft controls"); st.write("**Objective:** see which surface primarily creates pitch, roll, or yaw, and why wing geometry affects efficiency.")
    a,b,c=st.columns(3)
    with a: elevator=st.slider("Elevator deflection (°)",-25,25,0,help="Primary pitch control.")
    with b: aileron=st.slider("Aileron differential (°)",-25,25,0,help="Primary roll control.")
    with c: rudder=st.slider("Rudder deflection (°)",-25,25,0,help="Primary yaw control.")
    st.pyplot(plot_controls(elevator,aileron,rudder),use_container_width=True)
    x,y=st.columns(2)
    with x: st.success("**Wingtip vortices & induced drag:** pressure leaks around a finite wing's tips, creating vortices and downwash. Higher aspect ratio generally reduces induced drag.")
    with y: st.info("**High lift:** flaps increase camber (and often area), raising CL for takeoff/landing. Slats help delay leading-edge separation. These diagrams do not predict a specific aircraft's moments.")

with reference_tab:
    st.subheader("📖 Aerodynamics theory reference")
    st.caption("A connected reference for the main terms behind the visual labs. Open a section for definitions, physical intuition, and limits of the simplified models.")
    chosen_section = st.selectbox("Choose a theory group", list(REFERENCE_SECTIONS), help="The grouped format keeps closely related ideas together.")
    section = REFERENCE_SECTIONS[chosen_section]
    st.markdown(f"**Concepts covered:** {section['terms']}")
    st.success(section["theory"])
    st.info("Scientific note: these explanations use standard low-speed or ideal-flow concepts where stated. Real aircraft aerodynamics also depends on three-dimensional geometry, viscosity, turbulence, compressibility, and unsteady effects.")

with ask_tab:
    st.subheader("💬 Explain any concept")
    api_ready = bool(GEMINI_API_KEY) and genai is not None
    if api_ready:
        st.success("✨ AI mode is ready. Your key is read securely from the environment; it is never shown in this app.")
    elif GEMINI_API_KEY and genai is None:
        st.warning("A Gemini API key was found, but the google-genai package is not installed. Run: `py -m pip install -r requirements.txt`")
    else:
        st.caption("Offline mode: add `GEMINI_API_KEY` to enable AI answers for topics beyond the built-in reference.")
    question=st.text_input("Your topic or question",placeholder="Why does a wing stall? What is Mach number?")
    if st.button("Get explanation",key="ask"):
        if not question.strip():
            st.warning("Enter a topic or question first.")
        elif api_ready:
            try:
                with st.spinner("Preparing an explanation..."):
                    answer = ai_explanation(question, level)
                st.markdown(answer)
                st.caption("AI-generated educational explanation. Check primary aerospace references for design, operating, or safety decisions.")
            except Exception as exc:
                st.error("The Gemini request could not be completed. Showing the offline reference instead.")
                st.caption(f"Gemini error: {type(exc).__name__}: {exc}")
                topic, note = explain(question, level)
                if topic:
                    if note: st.info(note)
                    topic_card(topic, level)
                else: st.warning("That topic is not yet in the built-in knowledge base. Try lift, drag, Bernoulli, angle of attack, stall, boundary layer, Reynolds number, wingtip vortices, controls, or Mach number.")
        else:
            topic,note=explain(question,level)
            if topic:
                if note: st.info(note)
                topic_card(topic,level)
                st.caption("Try next: use the Flight lab for lift, drag, or stall; use Flow lab for Reynolds number and boundary layers.")
            else: st.warning("That topic is not yet in the built-in knowledge base. Add an API key for an AI answer, or try lift, drag, Bernoulli, angle of attack, stall, boundary layer, Reynolds number, wingtip vortices, controls, or Mach number.")
    st.caption("Offline fallback remains available. AI mode sends only the question you submit to the API.")
