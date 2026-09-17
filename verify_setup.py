"""Verify that all core dependencies import correctly and report versions."""

import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
import streamlit
import xgboost as xgb

print(f"Python      : {sys.version.split()[0]}")
print(f"pandas      : {pd.__version__}")
print(f"numpy       : {np.__version__}")
print(f"matplotlib  : {matplotlib.__version__}")
print(f"seaborn     : {sns.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"xgboost     : {xgb.__version__}")
print(f"shap        : {shap.__version__}")
print(f"joblib      : {joblib.__version__}")
print(f"streamlit   : {streamlit.__version__}") 
