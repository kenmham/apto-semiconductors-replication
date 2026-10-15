from pathlib import Path
import os,sys
R=Path(__file__).resolve().parents[1];O=R/'outputs/shoulder-bootstrap';O.mkdir(parents=True,exist_ok=True)
import valley_depth as V
V.FEATURES=str(R/'data/feature_selection.csv')

V._plot=lambda df: None
os.chdir(O);V.main()
