from pathlib import Path
import matplotlib.pyplot as plt
_style_path = Path(__file__).parent / "visuals_style.mplstyle"
plt.style.use(str(_style_path))
