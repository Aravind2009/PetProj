# Aerodynamics Explorer

A polished, offline Streamlit learning app for beginner-to-advanced explanations of aerodynamics. It includes interactive calculation and diagram labs using simplified, physically motivated models.

## Install and run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If Python's `py` launcher is not available, use the Python executable installed on your machine instead. The app opens in a local browser window.

## How to use it

- Choose a reading depth (Beginner, Intermediate, or Advanced) in the sidebar.
- Use **Learn** or the sidebar quick topics for an explanation and a supporting diagram.
- Use **Flight lab** to change air density (altitude proxy), airspeed, wing area, angle of attack, zero-lift drag, and aspect ratio; results use SI units.
- Use **Flow lab** to compare boundary layers and calculate Reynolds number.
- Use **Controls & wing** to explore simplified elevator, aileron, and rudder effects.
- Use **Explain any concept** for offline keyword-based explanations, including fuzzy matching.
- Use **Theory reference** for grouped coverage of forces and moments; fluid properties and conservation laws; Bernoulli, Newton, circulation and potential flow; airfoil/wing geometry; viscous flow; vortices; compressible flow and shocks; stability and flight dynamics; and wind-tunnel, CFD, and analytical tools.

## Requirements and privacy

`requirements.txt` contains Streamlit, matplotlib, NumPy, and the optional Google Gemini SDK. No API key, database, or internet connection is required for offline mode.

## Optional AI explanations

The app works offline by default. To enable AI-generated answers in **Explain any concept**, create a Gemini API key, then set it only in the terminal session that starts Streamlit:

```powershell
$env:GEMINI_API_KEY = "your-new-key-here"
py -m pip install -r requirements.txt
streamlit run app.py
```

Never place a key in `app.py`, commit it to Git, or share it in screenshots. If the key is missing, the app keeps using its built-in offline reference. Gemini receives only the submitted question and the app's system instruction; usage may incur API charges.

## Important limitations

This is an educational visualizer, not a flight-performance, design, or safety tool. The airfoil flow, stall point, drag polar, pressure relation, boundary layers, control effects, and compressibility charts are illustrative simplified models. Real aircraft behavior depends on 3D geometry, surface condition, turbulence, Reynolds and Mach number, atmospheric conditions, and unsteady flow. Bernoulli's principle alone does not explain lift: the full pressure field, circulation, and the downward change in air momentum are all relevant.
