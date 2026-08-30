# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, mean_absolute_error, r2_score, confusion_matrix, roc_curve
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import optuna
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

feature_cols = [
    'lnX', 'prod_z', 'inv_r_avg', 'pH', 'E_В_м', 'T_C',
    'gel_желатин', 'gel_агароза', 'gel_силикагель',
    'ionic_strength', 'D_ratio', 'pore_size', 'Da', 'Pe',
    'logKsp', 'S_saturation',
    'lnX_pH', 'lnX_ionic', 'pH_E', 'pH_logKsp', 'lnX_S'
]

D_water_25 = {
    'Ag+': 1.65e-9, 'Cu2+': 0.72e-9, 'Co2+': 0.73e-9, 'Ni2+': 0.68e-9,
    'Mg2+': 0.71e-9, 'Ca2+': 0.79e-9, 'Zn2+': 0.70e-9, 'Cd2+': 0.72e-9,
    'Pb2+': 0.94e-9, 'Hg2+': 0.85e-9, 'Mn2+': 0.69e-9, 'Fe2+': 0.72e-9,
    'Fe3+': 0.60e-9, 'Al3+': 0.54e-9, 'Cr3+': 0.59e-9, 'La3+': 0.62e-9,
    'NH4+': 1.96e-9, 'Sr2+': 0.79e-9, 'Ba2+': 0.85e-9,
    'OH-': 5.27e-9, 'Cl-': 2.03e-9, 'Br-': 2.08e-9, 'I-': 2.04e-9,
    'SCN-': 1.33e-9, 'CrO4_2-': 1.07e-9, 'Cr2O7_2-': 1.00e-9,
    'SO4_2-': 1.06e-9, 'CO3_2-': 0.92e-9, 'C2O4_2-': 0.83e-9,
    'PO4_3-': 0.61e-9, 'HPO4_2-': 0.77e-9, 'H2PO4-': 0.87e-9,
    'Fe(CN)6_4-': 0.76e-9, 'Fe(CN)6_3-': 0.87e-9,
    'SiO3_2-': 0.73e-9, 'MoO4_2-': 0.91e-9, 'WO4_2-': 0.88e-9,
    'F-': 1.47e-9, 'тартрат-': 0.80e-9, 'O2-': 2.0e-9, 'S2-': 1.8e-9
}
ion_radii = {
    'Ag+': 1.15, 'Cu2+': 0.73, 'Co2+': 0.745, 'Ni2+': 0.69, 'Mg2+': 0.72,
    'Ca2+': 1.00, 'Zn2+': 0.74, 'Cd2+': 0.95, 'Pb2+': 1.19, 'Hg2+': 1.02,
    'Mn2+': 0.83, 'Fe2+': 0.78, 'Fe3+': 0.645, 'Al3+': 0.535, 'Cr3+': 0.615,
    'La3+': 1.032, 'NH4+': 1.48, 'Sr2+': 1.18, 'Ba2+': 1.35,
    'OH-': 1.40, 'Cl-': 1.81, 'Br-': 1.96, 'I-': 2.20, 'SCN-': 2.15,
    'CrO4_2-': 2.40, 'Cr2O7_2-': 2.50, 'SO4_2-': 2.30, 'CO3_2-': 1.78,
    'C2O4_2-': 2.12, 'PO4_3-': 2.38, 'HPO4_2-': 2.20, 'H2PO4-': 2.00,
    'Fe(CN)6_4-': 4.00, 'Fe(CN)6_3-': 4.00, 'SiO3_2-': 2.60,
    'MoO4_2-': 2.54, 'WO4_2-': 2.60, 'F-': 1.33, 'тартрат-': 2.50,
    'O2-': 1.40, 'S2-': 1.84
}
ion_charges = {
    'Ag+': 1, 'Cu2+': 2, 'Co2+': 2, 'Ni2+': 2, 'Mg2+': 2, 'Ca2+': 2,
    'Zn2+': 2, 'Cd2+': 2, 'Pb2+': 2, 'Hg2+': 2, 'Mn2+': 2, 'Fe2+': 2,
    'Fe3+': 3, 'Al3+': 3, 'Cr3+': 3, 'La3+': 3, 'NH4+': 1, 'Sr2+': 2, 'Ba2+': 2,
    'OH-': 1, 'Cl-': 1, 'Br-': 1, 'I-': 1, 'SCN-': 1, 'CrO4_2-': 2,
    'Cr2O7_2-': 2, 'SO4_2-': 2, 'CO3_2-': 2, 'C2O4_2-': 2, 'PO4_3-': 3,
    'HPO4_2-': 2, 'H2PO4-': 1, 'Fe(CN)6_4-': 4, 'Fe(CN)6_3-': 3,
    'SiO3_2-': 2, 'MoO4_2-': 2, 'WO4_2-': 2, 'F-': 1, 'тартрат-': 1,
    'O2-': 2, 'S2-': 2
}
Ksp_dict = {
    'Ag2Cr2O7': 2.0e-12, 'Ag2CrO4': 1.12e-12, 'AgCl': 1.77e-10,
    'AgBr': 5.35e-13, 'AgI': 8.51e-17, 'AgSCN': 1.0e-12,
    'CuCrO4': 1.0e-10, 'CuCr2O7': 1.0e-12, 'PbCrO4': 2.8e-13,
    'PbCr2O7': 1.0e-12, 'BaCrO4': 1.17e-10, 'SrCrO4': 3.6e-5,
    'CdCrO4': 1.0e-10, 'ZnCrO4': 1.0e-10, 'HgCrO4': 1.0e-10,
    'Co(OH)2': 1.6e-15, 'Mg(OH)2': 5.61e-12, 'Ca(OH)2': 5.5e-6,
    'Ni(OH)2': 5.48e-16, 'Cu(OH)2': 2.2e-20, 'Fe(OH)3': 2.79e-39,
    'Fe(OH)2': 4.87e-17, 'Al(OH)3': 3.0e-34, 'Cr(OH)3': 6.3e-31,
    'Mn(OH)2': 1.9e-13, 'Zn(OH)2': 3.0e-17, 'Cd(OH)2': 7.2e-15,
    'Pb(OH)2': 1.43e-20, 'CaHPO4': 1.0e-7, 'Ca3(PO4)2': 1.0e-29,
    'Ca5(PO4)3OH': 1.0e-58, 'F-apatite': 1.0e-60,
    'PbI2': 9.8e-9, 'HgI2': 3.2e-29, 'CuI': 5.06e-12, 'CdI2': 5.0e-5,
    'PbBr2': 6.6e-6, 'HgBr2': 6.2e-19, 'CaC2O4': 2.3e-9, 'BaC2O4': 1.6e-6,
    'CdC2O4': 1.0e-8, 'CaCO3': 3.36e-9, 'BaCO3': 5.1e-9, 'SrCO3': 5.6e-10,
    'SrSO4': 3.44e-7, 'CaSO4': 2.4e-5, 'CuS': 6.0e-37, 'CdS': 8.0e-27,
    'Ag2MoO4': 2.8e-12, 'Ag2WO4': 5.5e-12, 'CaWO4': 1.0e-10,
    'La2(MoO4)3': 1.0e-20, 'MnO2': 1.0e-13, 'CaSiO3': 1.0e-9,
    'CuSiO3': 1.0e-10, 'Fe4[Fe(CN)6]3': 1.0e-40, 'Cu2[Fe(CN)6]': 1.0e-16,
    'Pb(SCN)2': 1.0e-8, 'Ag2CrO4+Ag4[Fe(CN)6]': 1.0e-20,
    'PbCrO4+Pb2[Fe(CN)6]': 1.0e-20,
}
stoichiometry = {
    'Ag2Cr2O7': (2,1), 'Ag2CrO4': (2,1), 'CuCrO4': (1,1), 'CuCr2O7': (1,2),
    'PbCrO4': (1,1), 'PbCr2O7': (1,2), 'BaCrO4': (1,1), 'SrCrO4': (1,1),
    'CdCrO4': (1,1), 'ZnCrO4': (1,1), 'HgCrO4': (1,1), 'Cu+Zn/CrO4': (1,1),
    'Cu+Pb/CrO4': (1,1), 'Co(OH)2': (1,2), 'Mg(OH)2': (1,2), 'Ca(OH)2': (1,2),
    'Ca(OH)2/CaCO3': (1,1), 'Ca3(PO4)2': (3,2), 'Ca5(PO4)3OH': (5,3),
    'CaHPO4': (1,1), 'Ca8(HPO4)2(PO4)4': (8,6), 'Ni(OH)2': (1,2),
    'Ni+Co(OH)2': (1,2), 'Cu(OH)2': (1,2), 'Fe(OH)3': (1,3), 'Fe(OH)2': (1,2),
    'Al(OH)3': (1,3), 'Cr(OH)3': (1,3), 'Mn(OH)2': (1,2), 'Zn(OH)2': (1,2),
    'Cd(OH)2': (1,2), 'Pb(OH)2': (1,2), 'Co+Mg(OH)2': (1,2), 'Co+Ni(OH)2': (1,2),
    'Co+Cu(OH)2': (1,2), 'Mg+Ca/PO4': (1,1), 'F-apatite': (5,3),
    'PbI2': (1,2), 'AgCl': (1,1), 'AgBr': (1,1), 'AgI': (1,1),
    'HgI2': (1,2), 'CuI': (1,1), 'CdI2': (1,2), 'PbBr2': (1,2),
    'HgBr2': (1,2), 'AgSCN': (1,1), 'Pb(SCN)2': (1,2), 'PbI2+HgI2': (1,2),
    'CuS': (1,1), 'CdS': (1,1), 'CaC2O4': (1,1), 'BaC2O4': (1,1),
    'CdC2O4': (1,1), 'CaCO3': (1,1), 'BaCO3': (1,1), 'SrCO3': (1,1),
    'SrSO4': (1,1), 'CaSO4': (1,1), 'РЗЭ тартраты': (1,1),
    'Fe4[Fe(CN)6]3': (4,3), 'Cu2[Fe(CN)6]': (2,1), 'NH4Cl': (1,1),
    'Ag2CrO4+Ag4[Fe(CN)6]': (2,1), 'PbCrO4+Pb2[Fe(CN)6]': (1,1),
    'асфальтены': (1,1), 'CaSiO3': (1,1), 'CuSiO3': (1,1),
    'MnO2': (1,2), 'Ag2MoO4': (2,1), 'Ag2WO4': (2,1), 'CaWO4': (1,1),
    'La2(MoO4)3': (2,3)
}
mapping = {
    'Ag2Cr2O7': ('Ag+','Cr2O7_2-'), 'Ag2CrO4': ('Ag+','CrO4_2-'),
    'CuCrO4': ('Cu2+','CrO4_2-'), 'CuCr2O7': ('Cu2+','Cr2O7_2-'),
    'PbCrO4': ('Pb2+','CrO4_2-'), 'PbCr2O7': ('Pb2+','Cr2O7_2-'),
    'BaCrO4': ('Ba2+','CrO4_2-'), 'SrCrO4': ('Sr2+','CrO4_2-'),
    'CdCrO4': ('Cd2+','CrO4_2-'), 'ZnCrO4': ('Zn2+','CrO4_2-'),
    'HgCrO4': ('Hg2+','CrO4_2-'), 'Cu+Zn/CrO4': ('Cu2+','CrO4_2-'),
    'Cu+Pb/CrO4': ('Cu2+','CrO4_2-'), 'Co(OH)2': ('Co2+','OH-'),
    'Mg(OH)2': ('Mg2+','OH-'), 'Ca(OH)2': ('Ca2+','OH-'),
    'Ca(OH)2/CaCO3': ('Ca2+','CO3_2-'), 'Ca3(PO4)2': ('Ca2+','PO4_3-'),
    'Ca5(PO4)3OH': ('Ca2+','PO4_3-'), 'CaHPO4': ('Ca2+','HPO4_2-'),
    'Ca8(HPO4)2(PO4)4': ('Ca2+','PO4_3-'), 'Ni(OH)2': ('Ni2+','OH-'),
    'Ni+Co(OH)2': ('Ni2+','OH-'), 'Cu(OH)2': ('Cu2+','OH-'),
    'Fe(OH)3': ('Fe3+','OH-'), 'Fe(OH)2': ('Fe2+','OH-'),
    'Al(OH)3': ('Al3+','OH-'), 'Cr(OH)3': ('Cr3+','OH-'),
    'Mn(OH)2': ('Mn2+','OH-'), 'Zn(OH)2': ('Zn2+','OH-'),
    'Cd(OH)2': ('Cd2+','OH-'), 'Pb(OH)2': ('Pb2+','OH-'),
    'Co+Mg(OH)2': ('Co2+','OH-'), 'Co+Ni(OH)2': ('Co2+','OH-'),
    'Co+Cu(OH)2': ('Co2+','OH-'), 'Mg+Ca/PO4': ('Mg2+','PO4_3-'),
    'F-apatite': ('Ca2+','F-'), 'PbI2': ('Pb2+','I-'),
    'AgCl': ('Ag+','Cl-'), 'AgBr': ('Ag+','Br-'), 'AgI': ('Ag+','I-'),
    'HgI2': ('Hg2+','I-'), 'CuI': ('Cu+','I-'), 'CdI2': ('Cd2+','I-'),
    'PbBr2': ('Pb2+','Br-'), 'HgBr2': ('Hg2+','Br-'), 'AgSCN': ('Ag+','SCN-'),
    'Pb(SCN)2': ('Pb2+','SCN-'), 'PbI2+HgI2': ('Pb2+','I-'),
    'CuS': ('Cu2+','S2-'), 'CdS': ('Cd2+','S2-'), 'CaC2O4': ('Ca2+','C2O4_2-'),
    'BaC2O4': ('Ba2+','C2O4_2-'), 'CdC2O4': ('Cd2+','C2O4_2-'),
    'CaCO3': ('Ca2+','CO3_2-'), 'BaCO3': ('Ba2+','CO3_2-'),
    'SrCO3': ('Sr2+','CO3_2-'), 'SrSO4': ('Sr2+','SO4_2-'),
    'CaSO4': ('Ca2+','SO4_2-'), 'РЗЭ тартраты': ('La3+','тартрат-'),
    'Fe4[Fe(CN)6]3': ('Fe3+','Fe(CN)6_4-'), 'Cu2[Fe(CN)6]': ('Cu2+','Fe(CN)6_4-'),
    'NH4Cl': ('NH4+','Cl-'), 'Ag2CrO4+Ag4[Fe(CN)6]': ('Ag+','CrO4_2-'),
    'PbCrO4+Pb2[Fe(CN)6]': ('Pb2+','CrO4_2-'), 'асфальтены': ('C','C'),
    'CaSiO3': ('Ca2+','SiO3_2-'), 'CuSiO3': ('Cu2+','SiO3_2-'),
    'MnO2': ('Mn2+','O2-'), 'Ag2MoO4': ('Ag+','MoO4_2-'), 'Ag2WO4': ('Ag+','WO4_2-'),
    'CaWO4': ('Ca2+','WO4_2-'), 'La2(MoO4)3': ('La3+','MoO4_2-')
}

def get_viscosity(gel_type, conc):
    if gel_type == 'желатин':
        if conc <= 5: return 1.8
        elif conc <= 10: return 1.8 + (2.5 - 1.8) * (conc - 5) / 5
        else: return 2.5
    elif gel_type == 'агароза':
        if conc <= 1: return 1.3
        elif conc <= 2: return 1.3 + (1.6 - 1.3) * (conc - 1)
        else: return 1.6
    elif gel_type == 'смесь':
        return 2.0
    return 2.0

def get_pore_size(gel_type, conc):
    if gel_type == 'желатин':
        if conc <= 3: return 120
        elif conc <= 5: return 120 - (120-100)*(conc-3)/2
        elif conc <= 7: return 100 - (100-90)*(conc-5)/2
        elif conc <= 10: return 90 - (90-80)*(conc-7)/3
        else: return 80 - (conc-10)*5
    elif gel_type == 'агароза':
        if conc <= 1: return 200
        elif conc <= 2: return 200 - (200-150)*(conc-1)
        else: return 150 - (conc-2)*30
    elif gel_type == 'смесь':
        return 100
    return 10

def compute_D_gel(ion, T_C, gel_type, gel_conc, D_water=1e-9):
    eta_rel = get_viscosity(gel_type, gel_conc)
    T_K = T_C + 273.15
    Ea_R = 18000.0 / 8.314
    factor = np.exp(Ea_R * (1/298.15 - 1/T_K))
    return D_water * factor / eta_rel

def compute_manual_features(row, params):
    z_cat, z_an = params['z_cat'], params['z_an']
    r_cat, r_an = params['r_cat'], params['r_an']
    D_cat_w, D_an_w = params['D_cat_w'], params['D_an_w']
    nu_in, nu_out = params['nu_in'], params['nu_out']
    
    T = row['T_C'] if pd.notna(row['T_C']) else 22.0
    pH = row['pH'] if pd.notna(row['pH']) else 7.0
    E = row['E_В_м'] if pd.notna(row['E_В_м']) else 0.0
    gel_raw = str(row['Гель']).lower().strip()
    C_in, C_out = row['C_in_M'], row['C_out_M']

    if 'смесь' in gel_raw:
        gel_type, gel_conc = 'смесь', 0.0
    elif 'силикагель' in gel_raw:
        gel_type, gel_conc = 'силикагель', 0.0
    elif 'агароза' in gel_raw:
        gel_type = 'агароза'
        m = re.search(r'(\d+(?:\.\d+)?)%', gel_raw)
        gel_conc = float(m.group(1)) if m else 1.0
    elif 'желатин' in gel_raw:
        gel_type = 'желатин'
        m = re.search(r'(\d+(?:\.\d+)?)%', gel_raw)
        gel_conc = float(m.group(1)) if m else 5.0
    else:
        gel_type, gel_conc = 'неизвестно', 5.0

    D_in = compute_D_gel(None, T, gel_type, gel_conc, D_cat_w)
    D_out = compute_D_gel(None, T, gel_type, gel_conc, D_an_w)
    D_eff = (D_in + D_out) / 2
    
    Xcorr = (D_out * C_out / nu_out) / (D_in * C_in / nu_in) if (D_in * C_in) > 0 else np.nan
    lnX = np.log(Xcorr) if Xcorr > 0 else 0.0
    prod_z = abs(z_cat * z_an)
    r_avg = (r_cat + r_an) / 2.0
    inv_r_avg = 1.0 / r_avg if r_avg > 0 else 0
    
    return {
        'lnX': lnX, 'prod_z': prod_z, 'inv_r_avg': inv_r_avg, 'pH': pH, 'E_В_м': E, 'T_C': T,
        'gel_желатин': 1 if gel_type == 'желатин' else 0,
        'gel_агароза': 1 if gel_type == 'агароза' else 0,
        'gel_силикагель': 1 if gel_type == 'силикагель' else 0,
        'ionic_strength': 0.5 * (C_in * z_cat**2 + C_out * z_an**2),
        'D_ratio': D_in/D_out if D_out>0 else 1.0,
        'pore_size': get_pore_size(gel_type, gel_conc),
        'Da': (prod_z * C_in * C_out) / (D_eff * (1 + gel_conc/10) * 0.01**2),
        'Pe': (E * abs(z_cat - z_an) * 1e-2) / (D_eff * (T + 273)) if E != 0 else 0.0,
        'logKsp': -10.0, 'S_saturation': np.log10(C_in * C_out + 1e-10),
        'lnX_pH': lnX * pH, 'lnX_ionic': lnX * 0.5 * (C_in * z_cat**2 + C_out * z_an**2),
        'pH_E': pH * E, 'pH_logKsp': pH * -10.0, 'lnX_S': lnX * np.log10(C_in * C_out + 1e-10)
    }

def compute_features_from_dict(row):
    system = row['Система']
    if system not in mapping: return None
    cation, anion = mapping[system]
    if pd.isna(cation) or pd.isna(anion): return None
    
    z_cat, z_an = ion_charges.get(cation, 1), ion_charges.get(anion, 1)
    r_cat, r_an = ion_radii.get(cation, 1.0), ion_radii.get(anion, 1.5)
    d_cat, d_an = D_water_25.get(cation, 1e-9), D_water_25.get(anion, 1e-9)
    nu_in, nu_out = stoichiometry.get(system, (1, 1))
    
    params = {'z_cat': z_cat, 'z_an': z_an, 'r_cat': r_cat, 'r_an': r_an,
              'D_cat_w': d_cat, 'D_an_w': d_an, 'nu_in': nu_in, 'nu_out': nu_out}
    feats = compute_manual_features(row, params)
    feats['Система'] = system
    return feats

def select_features_by_vif(X_df, threshold=10):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    
    features = list(X_df.columns)
    while True:
        X_sub = X_df[features]
        X_vif = sm.add_constant(X_sub)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        vif_data = vif_data[vif_data["feature"] != "const"]
        max_vif = vif_data["VIF"].max()
        
        if max_vif <= threshold or len(features) <= 10:
            break
        worst_feature = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
        features.remove(worst_feature)
    return features

def objective(trial, X_train, y_train, X_test, y_test, task='classification', n_pos=1, n_neg=1):
    if task == 'classification':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42
        }
        clf = xgb.XGBClassifier(**params, scale_pos_weight=n_neg/n_pos if n_pos>0 else 1.0)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        return roc_auc_score(y_test, pred)
    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42
        }
        reg = xgb.XGBRegressor(**params)
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        if len(pred) == 0 or len(y_test) == 0: return np.nan
        mae = mean_absolute_error(y_test, pred)
        return -mae

def cv_optuna(df_data, feature_list, systems, n_trials=20, log_widget=None, root=None):
    y_true_class, y_pred_class, y_proba_class = [], [], []
    y_true_reg, y_pred_reg = [], []
    best_clf_params = None
    best_reg_params = None
    default_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
                      'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3}
    
    n_pos = (df_data['Кольца'] == 1).sum()
    n_neg = (df_data['Кольца'] == 0).sum()

    for system in tqdm(systems, desc="CV (Optuna)"):
        if log_widget:
            log_widget.insert(tk.END, f"    -> Optimizing for system: {system}\n")
            log_widget.see(tk.END)
            if root: root.update()
            
        test_idx = df_data[df_data['Система'] == system].index
        train_idx = df_data[~df_data['Система'].isin([system])].index
        if len(train_idx) == 0: continue

        X_train = df_data.loc[train_idx, feature_list]
        y_train = df_data.loc[train_idx, 'Кольца'].astype(int)
        X_test = df_data.loc[test_idx, feature_list]
        y_test = df_data.loc[test_idx, 'Кольца'].astype(int)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        try:
            study_clf = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study_clf.optimize(lambda trial: objective(trial, X_train_scaled, y_train, X_test_scaled, y_test, 'classification', n_pos, n_neg), n_trials=n_trials, show_progress_bar=False)
            best_clf = study_clf.best_params if len(study_clf.trials) > 0 else default_params
        except Exception:
            best_clf = default_params
        if best_clf_params is None: best_clf_params = best_clf

        clf = xgb.XGBClassifier(**best_clf, scale_pos_weight=n_neg/n_pos if n_pos>0 else 1.0, eval_metric='logloss', use_label_encoder=False, random_state=42)
        clf.fit(X_train_scaled, y_train)
        y_pred_class.extend(clf.predict(X_test_scaled))
        y_proba_class.extend(clf.predict_proba(X_test_scaled)[:,1])
        y_true_class.extend(y_test)

        reg_mask = (df_data['Кольца'] == 1) & (df_data['p'].notna())
        df_reg_local = df_data[reg_mask].copy()
        if system in df_reg_local['Система'].unique():
            test_idx_reg = df_reg_local[df_reg_local['Система'] == system].index
            train_idx_reg = df_reg_local[~df_reg_local['Система'].isin([system])].index
            if len(test_idx_reg) > 0 and len(train_idx_reg) >= 5:
                Xr_train = df_reg_local.loc[train_idx_reg, feature_list]
                yr_train = df_reg_local.loc[train_idx_reg, 'p']
                Xr_test = df_reg_local.loc[test_idx_reg, feature_list]
                yr_test = df_reg_local.loc[test_idx_reg, 'p']

                scaler_reg = StandardScaler()
                Xr_train_scaled = scaler_reg.fit_transform(Xr_train)
                Xr_test_scaled = scaler_reg.transform(Xr_test)
                try:
                    study_reg = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
                    study_reg.optimize(lambda trial: objective(trial, Xr_train_scaled, yr_train, Xr_test_scaled, yr_test, 'regression'), n_trials=n_trials, show_progress_bar=False)
                    best_reg = study_reg.best_params if len(study_reg.trials) > 0 else default_params
                except Exception:
                    best_reg = default_params
                if best_reg_params is None: best_reg_params = best_reg

                reg = xgb.XGBRegressor(**best_reg, random_state=42)
                reg.fit(Xr_train_scaled, yr_train)
                y_pred_reg.extend(reg.predict(Xr_test_scaled))
                y_true_reg.extend(yr_test.values)

    acc = accuracy_score(y_true_class, y_pred_class)
    prec = precision_score(y_true_class, y_pred_class, zero_division=0)
    rec = recall_score(y_true_class, y_pred_class)
    auc_val = roc_auc_score(y_true_class, y_proba_class)
    mae = mean_absolute_error(y_true_reg, y_pred_reg) if len(y_true_reg) > 0 else np.nan
    r2 = r2_score(y_true_reg, y_pred_reg) if len(y_true_reg) > 0 else np.nan
    return acc, prec, rec, auc_val, mae, r2, best_clf_params, best_reg_params

def evaluate_model_cv(df_data, feature_list, systems, best_clf_params, best_reg_params):
    y_true_class, y_pred_class, y_proba_class = [], [], []
    y_true_reg, y_pred_reg = [], []
    n_pos = (df_data['Кольца'] == 1).sum()
    n_neg = (df_data['Кольца'] == 0).sum()

    for system in systems:
        test_idx = df_data[df_data['Система'] == system].index
        train_idx = df_data[~df_data['Система'].isin([system])].index
        if len(train_idx) == 0: continue
        
        X_train = df_data.loc[train_idx, feature_list]
        y_train = df_data.loc[train_idx, 'Кольца'].astype(int)
        X_test = df_data.loc[test_idx, feature_list]
        y_test = df_data.loc[test_idx, 'Кольца'].astype(int)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = xgb.XGBClassifier(**best_clf_params, scale_pos_weight=n_neg/n_pos if n_pos > 0 else 1.0, eval_metric='logloss', use_label_encoder=False, random_state=42)
        clf.fit(X_train_scaled, y_train)
        y_pred_class.extend(clf.predict(X_test_scaled))
        y_proba_class.extend(clf.predict_proba(X_test_scaled)[:, 1])
        y_true_class.extend(y_test)

        reg_mask = (df_data['Кольца'] == 1) & (df_data['p'].notna())
        df_reg_local = df_data[reg_mask].copy()
        if system in df_reg_local['Система'].unique():
            test_idx_reg = df_reg_local[df_reg_local['Система'] == system].index
            train_idx_reg = df_reg_local[~df_reg_local['Система'].isin([system])].index
            if len(test_idx_reg) > 0 and len(train_idx_reg) >= 5:
                Xr_train = df_reg_local.loc[train_idx_reg, feature_list]
                yr_train = df_reg_local.loc[train_idx_reg, 'p']
                Xr_test = df_reg_local.loc[test_idx_reg, feature_list]
                yr_test = df_reg_local.loc[test_idx_reg, 'p']

                scaler_reg = StandardScaler()
                Xr_train_scaled = scaler_reg.fit_transform(Xr_train)
                Xr_test_scaled = scaler_reg.transform(Xr_test)

                reg = xgb.XGBRegressor(**best_reg_params, random_state=42)
                reg.fit(Xr_train_scaled, yr_train)
                y_pred_reg.extend(reg.predict(Xr_test_scaled))
                y_true_reg.extend(yr_test.values)

    acc = accuracy_score(y_true_class, y_pred_class)
    prec = precision_score(y_true_class, y_pred_class, zero_division=0)
    rec = recall_score(y_true_class, y_pred_class)
    auc_val = roc_auc_score(y_true_class, y_proba_class)
    mae = mean_absolute_error(y_true_reg, y_pred_reg) if len(y_true_reg) > 0 else np.nan
    r2 = r2_score(y_true_reg, y_pred_reg) if len(y_true_reg) > 0 else np.nan
    return acc, prec, rec, auc_val, mae, r2

def train_final_models(df_clean, df_reg, feature_list, best_clf, best_reg):
    X_class = df_clean[feature_list]
    y_class = df_clean['Кольца'].astype(int)
    X_reg = df_reg[feature_list]
    y_reg = df_reg['p']

    scaler = StandardScaler()
    X_class_scaled = scaler.fit_transform(X_class)
    X_reg_scaled = scaler.transform(X_reg)

    n_pos = (df_clean['Кольца'] == 1).sum()
    n_neg = (df_clean['Кольца'] == 0).sum()
    clf = xgb.XGBClassifier(**best_clf, scale_pos_weight=n_neg/n_pos if n_pos > 0 else 1.0, eval_metric='logloss', use_label_encoder=False, random_state=42)
    clf.fit(X_class_scaled, y_class)
    calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
    calibrated_clf.fit(X_class_scaled, y_class)

    reg = xgb.XGBRegressor(**best_reg, random_state=42)
    reg.fit(X_reg_scaled, y_reg)

    return calibrated_clf, reg, scaler

class App:
    def __init__(self, root):
        self.root = root
        root.title("Liesegang Rings Predictor")
        root.geometry("1100x880")
        self.exp_counter = 0
        self.models_ready = False
        self.clf, self.reg, self.scaler = None, None, None
        self.optimal_threshold = 0.88

        self.setup_ui()
        self.log_text.insert(tk.END, "Ready.\n")
        self.log_text.insert(tk.END, "Load dataset to start Optuna + LOO-CV.\n")
        self.log_text.insert(tk.END, "This will take 30-60 seconds. Please do not close the window.\n")

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(top_frame, text="Load Dataset and Train (Optuna + LOO-CV)", command=self.start_training_thread).pack(side=tk.LEFT)

        name_frame = ttk.LabelFrame(main_frame, text="1. System Name", padding="5")
        name_frame.pack(fill=tk.X, pady=(0, 10))
        self.entry_name = ttk.Entry(name_frame, width=50)
        self.entry_name.insert(0, "Ag2Cr2O7 (exp. 1)")
        self.entry_name.pack(fill=tk.X)

        cond_frame = ttk.LabelFrame(main_frame, text="2. Experimental Conditions (concentrations, T, pH, E, GEL)", padding="5")
        cond_frame.pack(fill=tk.X, pady=(0, 10))

        self.entries_cond = {}
        ttk.Label(cond_frame, text="Gel type").grid(row=0, column=0, sticky="e", padx=5, pady=3)
        self.gel_type_var = tk.StringVar(value="желатин")
        self.gel_combo = ttk.Combobox(cond_frame, textvariable=self.gel_type_var, values=["желатин", "агароза", "силикагель"], width=18)
        self.gel_combo.grid(row=0, column=1, sticky="w", padx=5, pady=3)

        ttk.Label(cond_frame, text="Gel conc. (%)").grid(row=0, column=2, sticky="e", padx=5, pady=3)
        self.gel_conc_entry = ttk.Entry(cond_frame, width=10)
        self.gel_conc_entry.insert(0, "5")
        self.gel_conc_entry.grid(row=0, column=3, sticky="w", padx=5, pady=3)

        conds = [("System (Name)", "Ag2Cr2O7"), ("C_in (M)", "0.1"), ("C_out (M)", "0.1"), ("T (C)", "22.0"), ("pH", "5.5"), ("E (V/m)", "0")]
        for i, (label, default) in enumerate(conds):
            r = (i // 2) + 1
            c = (i % 2) * 2
            ttk.Label(cond_frame, text=label).grid(row=r, column=c, sticky="e", padx=5, pady=3)
            e = ttk.Entry(cond_frame, width=20)
            e.insert(0, default)
            e.grid(row=r, column=c+1, sticky="w", padx=5, pady=3)
            self.entries_cond[label] = e

        ion_frame = ttk.LabelFrame(main_frame, text="3. Ion Parameters (required for calculation)", padding="5")
        ion_frame.pack(fill=tk.X, pady=(0, 10))

        self.entries_ion = {}
        ions = [("Cation charge (z+)", "1"), ("Anion charge (z-)", "2"),
                ("Cation radius (A)", "1.15"), ("Anion radius (A)", "2.5"),
                ("D cation (water, 25C)", "1.65e-9"), ("D anion (water, 25C)", "1.00e-9"),
                ("Stoich. cation (nu_in)", "2"), ("Stoich. anion (nu_out)", "1")]
        
        for i, (label, default) in enumerate(ions):
            r = (i // 4) + 1
            c = (i % 4) * 2
            ttk.Label(ion_frame, text=label).grid(row=r, column=c, sticky="e", padx=5, pady=3)
            e = ttk.Entry(ion_frame, width=18)
            e.insert(0, default)
            e.grid(row=r, column=c+1, sticky="w", padx=5, pady=3)
            self.entries_ion[label] = e

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Predict (next experiment)", command=self.predict_single).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Concentration Map", command=self.generate_heatmap).pack(side=tk.LEFT, padx=10)

        log_frame = ttk.LabelFrame(main_frame, text="Prediction Log and Metrics", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=20, font=("Consolas", 10))
        scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def start_training_thread(self):
        file = filedialog.askopenfilename(title="Select dataset", filetypes=[("Excel/CSV", "*.xlsx *.csv")])
        if not file: return
        self.log_text.insert(tk.END, f"Starting training in background thread... ({file})\n")
        self.log_text.insert(tk.END, "Please wait. Updates will appear as they happen.\n")
        self.log_text.see(tk.END)
        thread = threading.Thread(target=self.load_and_train, args=(file,))
        thread.daemon = True
        thread.start()

    def load_and_train(self, file):
        try:
            if file.endswith('.csv'): df = pd.read_csv(file, encoding='utf-8-sig')
            else: df = pd.read_excel(file)
        except Exception as e:
            self.log_text.insert(tk.END, f"ERROR loading file: {e}\n")
            return

        df['Кольца'] = pd.to_numeric(df['Кольца'], errors='coerce')
        df['p'] = pd.to_numeric(df['p'], errors='coerce')
        df['C_in_M'] = pd.to_numeric(df['C_in_M'], errors='coerce')
        df['C_out_M'] = pd.to_numeric(df['C_out_M'], errors='coerce')
        df['T_C'] = pd.to_numeric(df['T_C'], errors='coerce').fillna(22)
        df['pH'] = pd.to_numeric(df['pH'], errors='coerce').fillna(7)
        df['E_В_м'] = pd.to_numeric(df['E_В_м'], errors='coerce').fillna(0)

        for sys in df['Система'].unique():
            if sys not in mapping:
                mapping[sys] = ('Na+', 'Cl-')
                stoichiometry[sys] = (1, 1)
                Ksp_dict[sys] = 1.0e-10

        df_feat_list = []
        for idx, row in df.iterrows():
            feats = compute_features_from_dict(row)
            if feats is not None:
                feats['_index'] = idx
                df_feat_list.append(feats)
        if not df_feat_list:
            self.log_text.insert(tk.END, "ERROR: Could not compute physical features!\n")
            return
        
        df_feat = pd.DataFrame(df_feat_list).set_index('_index')
        
        df_feat['lnX_pH'] = df_feat['lnX'] * df_feat['pH']
        df_feat['lnX_ionic'] = df_feat['lnX'] * df_feat['ionic_strength']
        df_feat['pH_E'] = df_feat['pH'] * df_feat['E_В_м']
        df_feat['pH_logKsp'] = df_feat['pH'] * df_feat['logKsp']
        df_feat['lnX_S'] = df_feat['lnX'] * df_feat['S_saturation']

        df_clean = df_feat.dropna()
        df_clean['Кольца'] = df.loc[df_clean.index, 'Кольца'].astype(int)
        df_clean['p'] = df.loc[df_clean.index, 'p']
        
        df_reg = df_clean[df_clean['Кольца'] == 1].dropna(subset=['p'])

        vif_file = 'vif_selected_features.joblib'
        if os.path.exists(vif_file):
            final_features = joblib.load(vif_file)
            self.log_text.insert(tk.END, f"Loaded VIF-selected features from file ({len(final_features)} features).\n")
        else:
            self.log_text.insert(tk.END, "Performing one-time VIF selection (2-3 seconds)...\n")
            self.log_text.see(tk.END)
            cols_for_vif = [c for c in feature_cols if c not in ['gel_желатин', 'gel_агароза', 'gel_силикагель']]
            X_vif = df_clean[cols_for_vif].dropna()
            final_features = select_features_by_vif(X_vif, threshold=10)
            for g in ['gel_желатин', 'gel_агароза', 'gel_силикагель']:
                if g in feature_cols and g not in final_features:
                    final_features.append(g)
            joblib.dump(final_features, vif_file)
            self.log_text.insert(tk.END, f"VIF selection done. Retained {len(final_features)} features.\n")
            self.log_text.see(tk.END)

        systems = df_clean['Система'].unique()
        systems_cv = [sys for sys in systems if len(df_reg[df_reg['Система'] == sys]) >= 5]

        self.log_text.insert(tk.END, f"Starting Optuna on selected features. Available systems: {len(systems_cv)}\n")
        self.log_text.see(tk.END)

        acc_opt, prec_opt, rec_opt, auc_opt, mae_opt, r2_opt, best_clf, best_reg = cv_optuna(
            df_clean, final_features, systems_cv, n_trials=20, log_widget=self.log_text, root=self.root
        )
        
        self.log_text.insert(tk.END, "\nOptuna completed. Best parameters:\n")
        self.log_text.insert(tk.END, f"Classifier: {best_clf}\n")
        self.log_text.insert(tk.END, f"Regressor: {best_reg}\n")
        
        self.clf, self.reg, self.scaler = train_final_models(df_clean, df_reg, final_features, best_clf, best_reg)
        joblib.dump(self.clf, 'xgboost_classifier.joblib')
        if self.reg: joblib.dump(self.reg, 'xgboost_regressor.joblib')
        if self.scaler: joblib.dump(self.scaler, 'feature_scaler.joblib')
        self.models_ready = True

        self.log_text.insert(tk.END, "\nCalculating final LOO-CV metrics...\n")
        acc_all, prec_all, rec_all, auc_all, mae_all, r2_all = evaluate_model_cv(df_clean, final_features, systems_cv, best_clf, best_reg)
        
        self.log_text.insert(tk.END, "=== LOO-CV (Leave-One-System-Out) ===\n")
        self.log_text.insert(tk.END, f"Classification: Accuracy={acc_all:.3f}, Precision={prec_all:.3f}, Recall={rec_all:.3f}, AUC={auc_all:.3f}\n")
        if not np.isnan(mae_all):
            self.log_text.insert(tk.END, f"Regression: MAE={mae_all:.4f}, R2={r2_all:.4f}\n")
        else:
            self.log_text.insert(tk.END, "Regression: Insufficient data.\n")
        
        self.log_text.insert(tk.END, "Calculating optimal threshold (Youden)...\n")
        X_all_scaled = self.scaler.transform(df_clean[final_features])
        proba_all = self.clf.predict_proba(X_all_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(df_clean['Кольца'].astype(int), proba_all)
        optimal_idx = np.argmax(tpr - fpr)
        self.optimal_threshold = thresholds[optimal_idx]
        
        self.log_text.insert(tk.END, f"Optimal threshold (Youden) = {self.optimal_threshold:.4f}\n")
        self.log_text.insert(tk.END, "="*60 + "\n")
        self.log_text.insert(tk.END, "MODEL IS READY. You can now make predictions.\n")
        self.log_text.see(tk.END)

    def get_vals(self):
        try:
            gel_type = self.gel_type_var.get()
            gel_conc = float(self.gel_conc_entry.get()) if gel_type != 'силикагель' else 0.0
            gel_str = f"{gel_type} {gel_conc}%" if gel_type != 'силикагель' else "силикагель"
            
            vals = {
                'system': self.entries_cond["System (Name)"].get(),
                'gel': gel_str,
                'c_in': float(self.entries_cond["C_in (M)"].get()),
                'c_out': float(self.entries_cond["C_out (M)"].get()),
                't': float(self.entries_cond["T (C)"].get()),
                'ph': float(self.entries_cond["pH"].get()),
                'e': float(self.entries_cond["E (V/m)"].get()),
                'z_cat': float(self.entries_ion["Cation charge (z+)"].get()),
                'z_an': float(self.entries_ion["Anion charge (z-)"].get()),
                'r_cat': float(self.entries_ion["Cation radius (A)"].get()),
                'r_an': float(self.entries_ion["Anion radius (A)"].get()),
                'D_cat_w': float(self.entries_ion["D cation (water, 25C)"].get()),
                'D_an_w': float(self.entries_ion["D anion (water, 25C)"].get()),
                'nu_in': float(self.entries_ion["Stoich. cation (nu_in)"].get()),
                'nu_out': float(self.entries_ion["Stoich. anion (nu_out)"].get())
            }
            return vals
        except ValueError:
            messagebox.showerror("Error", "Check numeric input (use decimal point).")
            return None

    def predict_single(self):
        if not self.models_ready:
            messagebox.showwarning("Model not trained", "First load a dataset via 'Load Dataset and Train'!")
            return
        
        vals = self.get_vals()
        if not vals: return

        row = pd.Series({
            'Система': vals['system'],
            'Гель': vals['gel'],
            'C_in_M': vals['c_in'],
            'C_out_M': vals['c_out'],
            'T_C': vals['t'],
            'pH': vals['ph'],
            'E_В_м': vals['e'],
            'Кольца': 1,
            'p': np.nan
        })
        
        params = {
            'z_cat': vals['z_cat'], 'z_an': vals['z_an'],
            'r_cat': vals['r_cat'], 'r_an': vals['r_an'],
            'D_cat_w': vals['D_cat_w'], 'D_an_w': vals['D_an_w'],
            'nu_in': vals['nu_in'], 'nu_out': vals['nu_out']
        }
        feats = compute_manual_features(row, params)
        
        if feats is None:
            self.log_text.insert(tk.END, f"   ERROR: Could not compute physical features.\n")
            return

        feats['lnX_pH'] = feats['lnX'] * feats['pH']
        feats['lnX_ionic'] = feats['lnX'] * feats['ionic_strength']
        feats['pH_E'] = feats['pH'] * feats['E_В_м']
        feats['pH_logKsp'] = feats['pH'] * feats['logKsp']
        feats['lnX_S'] = feats['lnX'] * feats['S_saturation']

        vif_file = 'vif_selected_features.joblib'
        if os.path.exists(vif_file):
            final_features = joblib.load(vif_file)
            X_input = np.array([[feats[f] for f in final_features]])
        else:
            X_input = np.array([[feats[f] for f in feature_cols]])
            
        X_scaled = self.scaler.transform(X_input)

        proba = self.clf.predict_proba(X_scaled)[0, 1]
        pred_class = "Rings will form" if proba >= self.optimal_threshold else "No rings"
        p_val = self.reg.predict(X_scaled)[0] if self.reg and pred_class == "Rings will form" else "—"

        self.exp_counter += 1
        res = f"   [Exp. #{self.exp_counter}] {vals['system']} | P={proba:.3f} | {pred_class} | p={p_val}\n\n"
        self.log_text.insert(tk.END, res)
        self.log_text.see(tk.END)

        self.entries_cond["C_in (M)"].delete(0, tk.END); self.entries_cond["C_in (M)"].insert(0, "0.1")
        self.entries_cond["C_out (M)"].delete(0, tk.END); self.entries_cond["C_out (M)"].insert(0, "0.1")
        self.entries_cond["T (C)"].delete(0, tk.END); self.entries_cond["T (C)"].insert(0, "22.0")
        self.entries_cond["pH"].delete(0, tk.END); self.entries_cond["pH"].insert(0, "5.5")
        self.entries_cond["E (V/m)"].delete(0, tk.END); self.entries_cond["E (V/m)"].insert(0, "0")

    def generate_heatmap(self):
        if not self.models_ready:
            messagebox.showwarning("Model not trained", "Load dataset first!")
            return

        vals = self.get_vals()
        if not vals: return

        steps = 20
        c_ins = np.linspace(0.01, 0.5, steps)
        c_outs = np.linspace(0.01, 0.5, steps)
        grid_res = np.zeros((steps, steps))
        grid_p = np.zeros((steps, steps))

        params = {
            'z_cat': vals['z_cat'], 'z_an': vals['z_an'],
            'r_cat': vals['r_cat'], 'r_an': vals['r_an'],
            'D_cat_w': vals['D_cat_w'], 'D_an_w': vals['D_an_w'],
            'nu_in': vals['nu_in'], 'nu_out': vals['nu_out']
        }
        
        vif_file = 'vif_selected_features.joblib'
        if os.path.exists(vif_file):
            final_features = joblib.load(vif_file)
        else:
            final_features = feature_cols

        for i, ci in enumerate(c_ins):
            for j, co in enumerate(c_outs):
                row = pd.DataFrame({
                    'Система': [vals['system']], 'Гель': [vals['gel']],
                    'C_in_M': [ci], 'C_out_M': [co],
                    'T_C': [vals['t']], 'pH': [vals['ph']], 'E_В_м': [vals['e']],
                    'Кольца': [1], 'p': [np.nan]
                })
                feats = compute_manual_features(row, params)
                if feats is None: return
                
                feats['lnX_pH'] = feats['lnX'] * feats['pH']
                feats['lnX_ionic'] = feats['lnX'] * feats['ionic_strength']
                feats['pH_E'] = feats['pH'] * feats['E_В_м']
                feats['pH_logKsp'] = feats['pH'] * feats['logKsp']
                feats['lnX_S'] = feats['lnX'] * feats['S_saturation']
                
                X_in = np.array([[feats[f] for f in final_features]])
                X_sc = self.scaler.transform(X_in)
                prob = self.clf.predict_proba(X_sc)[0, 1]
                p_res = self.reg.predict(X_sc)[0] if self.reg and prob >= self.optimal_threshold else 0
                grid_res[j, i] = 1 if prob >= self.optimal_threshold else 0
                grid_p[j, i] = p_res

        top = tk.Toplevel(self.root)
        top.title("Concentration Map")
        fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(grid_res, origin='lower', extent=[0.01, 0.5, 0.01, 0.5], cmap='RdYlGn')
        ax.set_title(f"Ring zones for {vals['system']}")
        ax.set_xlabel("C_in (M)")
        ax.set_ylabel("C_out (M)")
        for i in range(0, steps, 2):
            for j in range(0, steps, 2):
                if grid_res[j, i] == 1:
                    ax.text(c_ins[i], c_outs[j], f"p={grid_p[j,i]:.2f}", ha='center', va='center', color='black', fontsize=7)
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()