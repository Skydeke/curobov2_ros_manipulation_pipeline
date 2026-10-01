# A single world-collision contact: robot link vs scene obstacle.
string state_label  # "start", "goal", or segment description
string link         # Robot link involved
string obstacle     # Scene obstacle name
float64 depth       # Penetration depth (m); positive = past surface
