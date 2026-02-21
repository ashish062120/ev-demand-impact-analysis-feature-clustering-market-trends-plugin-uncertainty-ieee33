import subprocess
import sys

SCRIPTS = [
    "01_feature_importance.py",
    "02_kmeans_clustering.py",
    "03_ieee33_evdemand_section.py",
    "04_scenario_impact_analysis.py",
]

def run():
    py = sys.executable
    for s in SCRIPTS:
        print(f"\n--- Running {s} ---")
        r = subprocess.run([py, s], check=False)
        if r.returncode != 0:
            print(f"Stopped: {s} failed.")
            break

if __name__ == "__main__":
    run()