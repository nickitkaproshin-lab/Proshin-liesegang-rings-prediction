# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, mean_absolute_error, r2_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             brier_score_loss, classification_report)
from sklearn.calibration import calibration_curve
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import optuna
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import differential_evolution
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

BASE_FEATURES = [
    'lnX', 'prod_z', 'inv_r_avg',
    'D_ratio', 'Da', 'Pe', 'S_saturation',
    'pore_size', 'pH', 'logKsp', 'ionic_strength',
    'gel_желатин', 'gel_агароза', 'gel_силикагель',
    'lnX_pH', 'lnX_S', 'lnX_ionic', 'pH_logKsp', 'pH_E'
]

NEW_FEATURES = [
    'aging_factor',
    'complex_factor',
    'C_in_eff',
    'C_out_eff'
]

EXPERIMENT_SETS = {
    'baseline': BASE_FEATURES,
    'aging': BASE_FEATURES + ['aging_factor'],
    'complex': BASE_FEATURES + ['complex_factor', 'C_in_eff', 'C_out_eff'],
    'all': BASE_FEATURES + NEW_FEATURES
}

XGB_PARAMS = {
    'n_estimators': 144,
    'max_depth': 10,
    'learning_rate': 0.1205712628744377,
    'subsample': 0.8394633936788146,
    'colsample_bytree': 0.6624074561769746,
    'min_child_weight': 2,
    'random_state': 42,
    'n_jobs': -1
}

MONOTONE_CONSTRAINTS = "(1,-1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0)"

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
    'Ag+': 1.15, 'Cu2+': 0.73, 'Co2+': 0.745, 'Ni2+': 0.69,
    'Mg2+': 0.72, 'Ca2+': 1.00, 'Zn2+': 0.74, 'Cd2+': 0.95,
    'Pb2+': 1.19, 'Hg2+': 1.02, 'Mn2+': 0.83, 'Fe2+': 0.78,
    'Fe3+': 0.645, 'Al3+': 0.535, 'Cr3+': 0.615, 'La3+': 1.032,
    'NH4+': 1.48, 'Sr2+': 1.18, 'Ba2+': 1.35,
    'OH-': 1.40, 'Cl-': 1.81, 'Br-': 1.96, 'I-': 2.20,
    'SCN-': 2.15, 'CrO4_2-': 2.40, 'Cr2O7_2-': 2.50,
    'SO4_2-': 2.30, 'CO3_2-': 1.78, 'C2O4_2-': 2.12,
    'PO4_3-': 2.38, 'HPO4_2-': 2.20, 'H2PO4-': 2.00,
    'Fe(CN)6_4-': 4.00, 'Fe(CN)6_3-': 4.00,
    'SiO3_2-': 2.60, 'MoO4_2-': 2.54, 'WO4_2-': 2.60,
    'F-': 1.33, 'тартрат-': 2.50, 'O2-': 1.40, 'S2-': 1.84
}

ion_charges = {
    'Ag+': 1, 'Cu2+': 2, 'Co2+': 2, 'Ni2+': 2,
    'Mg2+': 2, 'Ca2+': 2, 'Zn2+': 2, 'Cd2+': 2,
    'Pb2+': 2, 'Hg2+': 2, 'Mn2+': 2, 'Fe2+': 2,
    'Fe3+': 3, 'Al3+': 3, 'Cr3+': 3, 'La3+': 3,
    'NH4+': 1, 'Sr2+': 2, 'Ba2+': 2,
    'OH-': 1, 'Cl-': 1, 'Br-': 1, 'I-': 1,
    'SCN-': 1, 'CrO4_2-': 2, 'Cr2O7_2-': 2,
    'SO4_2-': 2, 'CO3_2-': 2, 'C2O4_2-': 2,
    'PO4_3-': 3, 'HPO4_2-': 2, 'H2PO4-': 1,
    'Fe(CN)6_4-': 4, 'Fe(CN)6_3-': 3,
    'SiO3_2-': 2, 'MoO4_2-': 2, 'WO4_2-': 2,
    'F-': 1, 'тартрат-': 1, 'O2-': 2, 'S2-': 2
}

Ksp_dict = {
    'Ag2Cr2O7': 2.0e-12, 'Ag2CrO4': 1.12e-12, 'AgCl': 1.77e-10,
    'AgBr': 5.35e-13, 'AgI': 8.51e-17, 'AgSCN': 1.0e-12,
    'CuCrO4': 1.0e-10, 'CuCr2O7': 1.0e-12,
    'PbCrO4': 2.8e-13, 'PbCr2O7': 1.0e-12,
    'BaCrO4': 1.17e-10, 'SrCrO4': 3.6e-5,
    'CdCrO4': 1.0e-10, 'ZnCrO4': 1.0e-10, 'HgCrO4': 1.0e-10,
    'Co(OH)2': 1.6e-15, 'Mg(OH)2': 5.61e-12, 'Ca(OH)2': 5.5e-6,
    'Ni(OH)2': 5.48e-16, 'Cu(OH)2': 2.2e-20,
    'Fe(OH)3': 2.79e-39, 'Fe(OH)2': 4.87e-17,
    'Al(OH)3': 3.0e-34, 'Cr(OH)3': 6.3e-31,
    'Mn(OH)2': 1.9e-13, 'Zn(OH)2': 3.0e-17, 'Cd(OH)2': 7.2e-15,
    'Pb(OH)2': 1.43e-20,
    'CaHPO4': 1.0e-7, 'Ca3(PO4)2': 1.0e-29, 'Ca5(PO4)3OH': 1.0e-58,
    'F-apatite': 1.0e-60,
    'PbI2': 9.8e-9, 'HgI2': 3.2e-29, 'CuI': 5.06e-12, 'CdI2': 5.0e-5,
    'PbBr2': 6.6e-6, 'HgBr2': 6.2e-19,
    'CaC2O4': 2.3e-9, 'BaC2O4': 1.6e-6, 'CdC2O4': 1.0e-8,
    'CaCO3': 3.36e-9, 'BaCO3': 5.1e-9, 'SrCO3': 5.6e-10,
    'SrSO4': 3.44e-7, 'CaSO4': 2.4e-5,
    'CuS': 6.0e-37, 'CdS': 8.0e-27,
    'Ag2MoO4': 2.8e-12, 'Ag2WO4': 5.5e-12, 'CaWO4': 1.0e-10,
    'La2(MoO4)3': 1.0e-20,
    'MnO2': 1.0e-13,
    'CaSiO3': 1.0e-9, 'CuSiO3': 1.0e-10,
    'Fe4[Fe(CN)6]3': 1.0e-40,
    'Cu2[Fe(CN)6]': 1.0e-16,
    'Pb(SCN)2': 1.0e-8
}

stoichiometry = {
    'Ag2Cr2O7': (2,1), 'Ag2CrO4': (2,1), 'CuCrO4': (1,1), 'CuCr2O7': (1,2),
    'PbCrO4': (1,1), 'PbCr2O7': (1,2), 'BaCrO4': (1,1), 'SrCrO4': (1,1),
    'CdCrO4': (1,1), 'ZnCrO4': (1,1), 'HgCrO4': (1,1),
    'Co(OH)2': (1,2), 'Mg(OH)2': (1,2), 'Ca(OH)2': (1,2),
    'Ca3(PO4)2': (3,2), 'Ca5(PO4)3OH': (5,3), 'CaHPO4': (1,1),
    'Ni(OH)2': (1,2), 'Cu(OH)2': (1,2), 'Fe(OH)3': (1,3), 'Fe(OH)2': (1,2),
    'Al(OH)3': (1,3), 'Cr(OH)3': (1,3), 'Mn(OH)2': (1,2),
    'Zn(OH)2': (1,2), 'Cd(OH)2': (1,2), 'Pb(OH)2': (1,2),
    'F-apatite': (5,3),
    'PbI2': (1,2), 'AgCl': (1,1), 'AgBr': (1,1), 'AgI': (1,1),
    'HgI2': (1,2), 'CuI': (1,1), 'CdI2': (1,2),
    'PbBr2': (1,2), 'HgBr2': (1,2),
    'AgSCN': (1,1), 'Pb(SCN)2': (1,2),
    'CuS': (1,1), 'CdS': (1,1),
    'CaC2O4': (1,1), 'BaC2O4': (1,1), 'CdC2O4': (1,1),
    'CaCO3': (1,1), 'BaCO3': (1,1), 'SrCO3': (1,1),
    'SrSO4': (1,1), 'CaSO4': (1,1),
    'РЗЭ тартраты': (1,1),
    'Fe4[Fe(CN)6]3': (4,3), 'Cu2[Fe(CN)6]': (2,1),
    'CaSiO3': (1,1), 'CuSiO3': (1,1),
    'MnO2': (1,2),
    'Ag2MoO4': (2,1), 'Ag2WO4': (2,1), 'CaWO4': (1,1),
    'La2(MoO4)3': (2,3)
}

mapping = {
    'Ag2Cr2O7': ('Ag+','Cr2O7_2-'), 'Ag2CrO4': ('Ag+','CrO4_2-'),
    'CuCrO4': ('Cu2+','CrO4_2-'), 'CuCr2O7': ('Cu2+','Cr2O7_2-'),
    'PbCrO4': ('Pb2+','CrO4_2-'), 'PbCr2O7': ('Pb2+','Cr2O7_2-'),
    'BaCrO4': ('Ba2+','CrO4_2-'), 'SrCrO4': ('Sr2+','CrO4_2-'),
    'CdCrO4': ('Cd2+','CrO4_2-'), 'ZnCrO4': ('Zn2+','CrO4_2-'),
    'HgCrO4': ('Hg2+','CrO4_2-'),
    'Co(OH)2': ('Co2+','OH-'), 'Mg(OH)2': ('Mg2+','OH-'),
    'Ca(OH)2': ('Ca2+','OH-'),
    'Ca3(PO4)2': ('Ca2+','PO4_3-'), 'Ca5(PO4)3OH': ('Ca2+','PO4_3-'),
    'CaHPO4': ('Ca2+','HPO4_2-'),
    'Ni(OH)2': ('Ni2+','OH-'), 'Cu(OH)2': ('Cu2+','OH-'),
    'Fe(OH)3': ('Fe3+','OH-'), 'Fe(OH)2': ('Fe2+','OH-'),
    'Al(OH)3': ('Al3+','OH-'), 'Cr(OH)3': ('Cr3+','OH-'),
    'Mn(OH)2': ('Mn2+','OH-'), 'Zn(OH)2': ('Zn2+','OH-'),
    'Cd(OH)2': ('Cd2+','OH-'), 'Pb(OH)2': ('Pb2+','OH-'),
    'F-apatite': ('Ca2+','F-'),
    'PbI2': ('Pb2+','I-'), 'AgCl': ('Ag+','Cl-'), 'AgBr': ('Ag+','Br-'),
    'AgI': ('Ag+','I-'), 'HgI2': ('Hg2+','I-'), 'CuI': ('Cu+','I-'),
    'CdI2': ('Cd2+','I-'), 'PbBr2': ('Pb2+','Br-'), 'HgBr2': ('Hg2+','Br-'),
    'AgSCN': ('Ag+','SCN-'), 'Pb(SCN)2': ('Pb2+','SCN-'),
    'CuS': ('Cu2+','S2-'), 'CdS': ('Cd2+','S2-'),
    'CaC2O4': ('Ca2+','C2O4_2-'), 'BaC2O4': ('Ba2+','C2O4_2-'),
    'CdC2O4': ('Cd2+','C2O4_2-'),
    'CaCO3': ('Ca2+','CO3_2-'), 'BaCO3': ('Ba2+','CO3_2-'),
    'SrCO3': ('Sr2+','CO3_2-'),
    'SrSO4': ('Sr2+','SO4_2-'), 'CaSO4': ('Ca2+','SO4_2-'),
    'РЗЭ тартраты': ('La3+','тартрат-'),
    'Fe4[Fe(CN)6]3': ('Fe3+','Fe(CN)6_4-'),
    'Cu2[Fe(CN)6]': ('Cu2+','Fe(CN)6_4-'),
    'CaSiO3': ('Ca2+','SiO3_2-'), 'CuSiO3': ('Cu2+','SiO3_2-'),
    'MnO2': ('Mn2+','O2-'),
    'Ag2MoO4': ('Ag+','MoO4_2-'), 'Ag2WO4': ('Ag+','WO4_2-'),
    'CaWO4': ('Ca2+','WO4_2-'), 'La2(MoO4)3': ('La3+','MoO4_2-')
}

def load_data(file_path=None):
    if file_path is None:
        return None
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    else:
        df = pd.read_excel(file_path)
    df['Кольца'] = pd.to_numeric(df['Кольца'], errors='coerce')
    df['p'] = pd.to_numeric(df['p'], errors='coerce')
    df['C_in_M'] = pd.to_numeric(df['C_in_M'], errors='coerce')
    df['C_out_M'] = pd.to_numeric(df['C_out_M'], errors='coerce')
    df['T_C'] = pd.to_numeric(df['T_C'], errors='coerce').fillna(22)
    df['pH'] = pd.to_numeric(df['pH'], errors='coerce').fillna(7)
    df['E_В_м'] = pd.to_numeric(df['E_В_м'], errors='coerce').fillna(0)
    return df

def get_viscosity(gel_type, conc):
    if gel_type == 'желатин':
        if conc <= 5: return 1.8
        elif conc <= 10: return 1.8 + (2.5 - 1.8) * (conc - 5) / 5
        else: return 2.5
    elif gel_type == 'агароза':
        if conc <= 1: return 1.3
        elif conc <= 2: return 1.3 + (1.6 - 1.3) * (conc - 1)
        else: return 1.6
    else: return 2.0

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
    else: return 10

def parse_gel(gel_str):
    if pd.isna(gel_str): return 'unknown', 0.0
    s = str(gel_str).lower().strip()
    if 'желатин' in s:
        m = re.search(r'(\d+(?:\.\d+)?)%', s)
        conc = float(m.group(1)) if m else 5.0
        return 'желатин', conc
    elif 'агароза' in s:
        m = re.search(r'(\d+(?:\.\d+)?)%', s)
        conc = float(m.group(1)) if m else 1.0
        return 'агароза', conc
    elif 'силикагель' in s:
        return 'силикагель', 0.0
    else: return s, 0.0

def compute_D_gel(ion, T_C, gel_type, gel_conc):
    D_water = D_water_25.get(ion, 1e-9)
    eta_rel = get_viscosity(gel_type, gel_conc)
    T_K = T_C + 273.15
    Ea_R = 18000.0 / 8.314
    factor = np.exp(Ea_R * (1/298.15 - 1/T_K))
    return D_water * factor / eta_rel

def compute_basic_features(df):
    df = df.copy()
    cols = ['lnX', 'prod_z', 'inv_r_avg', 'gel_желатин', 'gel_агароза',
            'gel_силикагель', 'nu_in', 'nu_out', 'z_cat', 'z_an',
            'C_in', 'C_out', 'gel_conc', 'D_ratio', 'pore_size', 'Da', 'Pe',
            'ionic_strength', 'Ksp', 'logKsp', 'S_saturation']
    for col in cols:
        df[col] = np.nan
    df['gel_желатин'] = 0
    df['gel_агароза'] = 0
    df['gel_силикагель'] = 0

    for idx, row in df.iterrows():
        system = row['Система']
        if system not in mapping:
            continue
        cation, anion = mapping[system]
        if pd.isna(cation) or pd.isna(anion):
            continue
        C_in = row['C_in_M']
        C_out = row['C_out_M']
        if pd.isna(C_in) or pd.isna(C_out):
            continue
        T = row['T_C'] if pd.notna(row['T_C']) else 22.0
        pH = row['pH'] if pd.notna(row['pH']) else 7.0
        E = row['E_В_м'] if pd.notna(row['E_В_м']) else 0.0
        gel_raw = row['Гель']
        gel_type, gel_conc = parse_gel(gel_raw)

        nu_in, nu_out = stoichiometry.get(system, (1,1))
        df.at[idx, 'nu_in'] = nu_in
        df.at[idx, 'nu_out'] = nu_out
        df.at[idx, 'C_in'] = C_in
        df.at[idx, 'C_out'] = C_out
        df.at[idx, 'gel_conc'] = gel_conc

        D_in = compute_D_gel(cation, T, gel_type, gel_conc)
        D_out = compute_D_gel(anion, T, gel_type, gel_conc)
        D_eff = (D_in + D_out) / 2
        df.at[idx, 'D_ratio'] = D_in / D_out if D_out > 0 else 1.0

        Xcorr = (D_out * C_out / nu_out) / (D_in * C_in / nu_in) if (D_in * C_in) > 0 else np.nan
        if Xcorr > 0:
            df.at[idx, 'lnX'] = np.log(Xcorr)

        z_cat = ion_charges.get(cation, 1)
        z_an = ion_charges.get(anion, 1)
        df.at[idx, 'z_cat'] = z_cat
        df.at[idx, 'z_an'] = z_an
        df.at[idx, 'prod_z'] = abs(z_cat * z_an)

        r_cat = ion_radii.get(cation, 1.0)
        r_an = ion_radii.get(anion, 1.5)
        r_avg = (r_cat + r_an) / 2.0
        df.at[idx, 'inv_r_avg'] = 1.0 / r_avg if r_avg > 0 else 0

        if gel_type == 'желатин':
            df.at[idx, 'gel_желатин'] = 1
        elif gel_type == 'агароза':
            df.at[idx, 'gel_агароза'] = 1
        elif gel_type == 'силикагель':
            df.at[idx, 'gel_силикагель'] = 1

        df.at[idx, 'pore_size'] = get_pore_size(gel_type, gel_conc)
        L_char = 0.01
        reaction_rate = abs(z_cat * z_an) * C_in * C_out
        Da = reaction_rate / (D_eff * (1 + gel_conc/10) * L_char**2)
        df.at[idx, 'Da'] = Da
        if E != 0:
            Pe = (E * abs(z_cat - z_an) * 1e-2) / (D_eff * (T + 273))
        else:
            Pe = 0.0
        df.at[idx, 'Pe'] = Pe

        I = 0.5 * (C_in * z_cat**2 + C_out * z_an**2)
        df.at[idx, 'ionic_strength'] = I

        Ksp = Ksp_dict.get(system, 1e-10)
        df.at[idx, 'Ksp'] = Ksp
        df.at[idx, 'logKsp'] = np.log10(Ksp) if Ksp > 0 else -10
        ion_product = C_in * C_out
        S = ion_product / Ksp if Ksp > 0 else 0
        df.at[idx, 'S_saturation'] = np.log10(S + 1e-10)

    df['lnX'] = df['lnX'].fillna(0)
    df['prod_z'] = df['prod_z'].fillna(1)
    df['inv_r_avg'] = df['inv_r_avg'].fillna(0.5)
    df['ionic_strength'] = df['ionic_strength'].fillna(0.01)
    df['logKsp'] = df['logKsp'].fillna(-10)
    df['S_saturation'] = df['S_saturation'].fillna(0)
    df['D_ratio'] = df['D_ratio'].fillna(1)
    df['pore_size'] = df['pore_size'].fillna(50)
    df['Da'] = df['Da'].fillna(0.1)
    df['Pe'] = df['Pe'].fillna(0)
    return df

def add_aging_factor(df):
    tau = 1814400
    df['aging_factor'] = 1.0
    mask_gel = df['gel_желатин'] == 1
    df.loc[mask_gel, 'aging_factor'] = np.exp(-tau / tau) * (1 + 0.05 * df.loc[mask_gel, 'gel_conc'])
    mask_agar = df['gel_агароза'] == 1
    df.loc[mask_agar, 'aging_factor'] = np.exp(-tau / tau) * (1 + 0.03 * df.loc[mask_agar, 'gel_conc'])
    return df

def add_complexation_factors(df):
    K_cat = 10.0
    K_an = 5.0
    I = df['ionic_strength'].values
    df['complex_factor'] = 1.0 / (1 + (K_cat + K_an)/2 * I)
    df['C_in_eff'] = df['C_in'] * df['complex_factor']
    df['C_out_eff'] = df['C_out'] * df['complex_factor']
    return df

def compute_all_features(df):
    df = compute_basic_features(df)
    df = add_aging_factor(df)
    df = add_complexation_factors(df)
    df['lnX_pH'] = df['lnX'] * df['pH']
    df['lnX_S'] = df['lnX'] * df['S_saturation']
    df['lnX_ionic'] = df['lnX'] * df['ionic_strength']
    df['pH_logKsp'] = df['pH'] - df['logKsp']
    df['pH_E'] = df['pH'] * df['E_В_м']
    return df

def select_features_by_vif(X_df, threshold=10):
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

def compute_manual_features_from_dict(row, params):
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
    elif 'силикагель' in gel_raw or 'silica' in gel_raw:
        gel_type, gel_conc = 'силикагель', 0.0
    elif 'агароза' in gel_raw or 'agarose' in gel_raw:
        gel_type = 'агароза'
        m = re.search(r'(\d+(?:\.\d+)?)%', gel_raw)
        gel_conc = float(m.group(1)) if m else 1.0
    elif 'желатин' in gel_raw or 'gelatin' in gel_raw:
        gel_type = 'желатин'
        m = re.search(r'(\d+(?:\.\d+)?)%', gel_raw)
        gel_conc = float(m.group(1)) if m else 5.0
    else:
        gel_type, gel_conc = 'неизвестно', 5.0

    def compute_D(ion, T_C, gel_type, gel_conc, D_water=1e-9):
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
        eta_rel = get_viscosity(gel_type, gel_conc)
        T_K = T_C + 273.15
        Ea_R = 18000.0 / 8.314
        factor = np.exp(Ea_R * (1/298.15 - 1/T_K))
        return D_water * factor / eta_rel

    D_in = compute_D(None, T, gel_type, gel_conc, D_cat_w)
    D_out = compute_D(None, T, gel_type, gel_conc, D_an_w)
    D_eff = (D_in + D_out) / 2

    Xcorr = (D_out * C_out / nu_out) / (D_in * C_in / nu_in) if (D_in * C_in) > 0 else np.nan
    lnX = np.log(Xcorr) if Xcorr > 0 else 0.0
    prod_z = abs(z_cat * z_an)
    r_avg = (r_cat + r_an) / 2.0
    inv_r_avg = 1.0 / r_avg if r_avg > 0 else 0

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

    feats = {
        'lnX': lnX,
        'prod_z': prod_z,
        'inv_r_avg': inv_r_avg,
        'pH': pH,
        'E_В_м': E,
        'T_C': T,
        'gel_желатин': 1 if gel_type == 'желатин' else 0,
        'gel_агароза': 1 if gel_type == 'агароза' else 0,
        'gel_силикагель': 1 if gel_type == 'силикагель' else 0,
        'ionic_strength': 0.5 * (C_in * z_cat**2 + C_out * z_an**2),
        'D_ratio': D_in / D_out if D_out > 0 else 1.0,
        'pore_size': get_pore_size(gel_type, gel_conc),
        'Da': (prod_z * C_in * C_out) / (D_eff * (1 + gel_conc/10) * 0.01**2),
        'Pe': (E * abs(z_cat - z_an) * 1e-2) / (D_eff * (T + 273)) if E != 0 else 0.0,
        'logKsp': -10.0,
        'S_saturation': np.log10(C_in * C_out + 1e-10),
        'lnX_pH': lnX * pH,
        'lnX_ionic': lnX * 0.5 * (C_in * z_cat**2 + C_out * z_an**2),
        'pH_E': pH * E,
        'pH_logKsp': pH * (-10.0),
        'lnX_S': lnX * np.log10(C_in * C_out + 1e-10)
    }
    return feats

class App:
    def __init__(self, root):
        self.root = root
        root.title("Liesegang Rings Predictor (v2.0)")
        root.geometry("1200x900")
        self.models_ready = False
        self.clf = None
        self.reg = None
        self.scaler = None
        self.final_features = None
        self.optimal_threshold = 0.88
        self.df_clean = None
        self.df_reg = None
        self.exp_counter = 0
        self.setup_ui()
        self.log_text.insert(tk.END, "Ready.\n")
        self.log_text.insert(tk.END, "Load dataset via 'Load Dataset and Train'.\n")
        self.log_text.insert(tk.END, "This will run Optuna + LOO-CV (30-60 sec).\n")

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(top_frame, text="Load Dataset and Train (Optuna + LOO-CV)",
                   command=self.start_training_thread).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Show Diagnostics", command=self.show_diagnostics).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Find Concentrations", command=self.find_concentrations_gui).pack(side=tk.LEFT, padx=5)

        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(input_frame, text="System Name").grid(row=0, column=0, sticky="e", padx=5, pady=3)
        self.entry_system = ttk.Entry(input_frame, width=25)
        self.entry_system.insert(0, "Ag2Cr2O7")
        self.entry_system.grid(row=0, column=1, sticky="w", padx=5, pady=3)

        ttk.Label(input_frame, text="Gel type").grid(row=0, column=2, sticky="e", padx=5, pady=3)
        self.gel_type_var = tk.StringVar(value="gelatin")
        self.gel_combo = ttk.Combobox(input_frame, textvariable=self.gel_type_var,
                                      values=["gelatin", "agarose", "silica gel"], width=12)
        self.gel_combo.grid(row=0, column=3, sticky="w", padx=5, pady=3)

        ttk.Label(input_frame, text="Gel conc. (%)").grid(row=0, column=4, sticky="e", padx=5, pady=3)
        self.entry_gel_conc = ttk.Entry(input_frame, width=8)
        self.entry_gel_conc.insert(0, "5")
        self.entry_gel_conc.grid(row=0, column=5, sticky="w", padx=5, pady=3)

        conds = [("C_in (M)", "0.1"), ("C_out (M)", "0.1"), ("T (C)", "22.0"),
                 ("pH", "5.5"), ("E (V/m)", "0")]
        self.cond_entries = {}
        for i, (label, default) in enumerate(conds):
            ttk.Label(input_frame, text=label).grid(row=1, column=i*2, sticky="e", padx=5, pady=3)
            e = ttk.Entry(input_frame, width=10)
            e.insert(0, default)
            e.grid(row=1, column=i*2+1, sticky="w", padx=5, pady=3)
            self.cond_entries[label] = e

        ion_frame = ttk.LabelFrame(input_frame, text="Ion Parameters (for calculation)", padding="5")
        ion_frame.grid(row=2, column=0, columnspan=12, sticky="ew", pady=5)

        ion_labels = [
            ("z+", "1"), ("z-", "2"), ("r+ (A)", "1.15"), ("r- (A)", "2.5"),
            ("D+ (water)", "1.65e-9"), ("D- (water)", "1.00e-9"),
            ("nu_in", "2"), ("nu_out", "1")
        ]
        self.ion_entries = {}
        for i, (label, default) in enumerate(ion_labels):
            ttk.Label(ion_frame, text=label).grid(row=0, column=i*2, sticky="e", padx=3, pady=2)
            e = ttk.Entry(ion_frame, width=12)
            e.insert(0, default)
            e.grid(row=0, column=i*2+1, sticky="w", padx=3, pady=2)
            self.ion_entries[label] = e

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Predict", command=self.predict_single).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Concentration Map", command=self.generate_heatmap).pack(side=tk.LEFT, padx=10)

        log_frame = ttk.LabelFrame(main_frame, text="Log and Results", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=25, font=("Consolas", 10))
        scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def start_training_thread(self):
        file = filedialog.askopenfilename(title="Select dataset", filetypes=[("Excel/CSV", "*.xlsx *.csv")])
        if not file:
            return
        self.log_text.insert(tk.END, f"Loading dataset: {file}\n")
        thread = threading.Thread(target=self.load_and_train, args=(file,))
        thread.daemon = True
        thread.start()

    def load_and_train(self, file):
        try:
            df_raw = load_data(file)
            df_raw['Кольца'] = pd.to_numeric(df_raw['Кольца'], errors='coerce')
            df_raw['p'] = pd.to_numeric(df_raw['p'], errors='coerce')
            df_raw['C_in_M'] = pd.to_numeric(df_raw['C_in_M'], errors='coerce')
            df_raw['C_out_M'] = pd.to_numeric(df_raw['C_out_M'], errors='coerce')
            df_raw['T_C'] = pd.to_numeric(df_raw['T_C'], errors='coerce').fillna(22)
            df_raw['pH'] = pd.to_numeric(df_raw['pH'], errors='coerce').fillna(7)
            df_raw['E_В_м'] = pd.to_numeric(df_raw['E_В_м'], errors='coerce').fillna(0)

            df = compute_all_features(df_raw)
            self.log_text.insert(tk.END, f"Features computed. Shape: {df.shape}\n")

            df_cls = df.dropna(subset=BASE_FEATURES + ['Кольца']).copy()
            self.df_clean = df_cls
            self.df_reg = df_cls[(df_cls['Кольца'] == 1) & (df_cls['p'].notna())].copy()

            vif_file = 'vif_selected_features.joblib'
            if os.path.exists(vif_file):
                final_features = joblib.load(vif_file)
                self.log_text.insert(tk.END, f"Loaded VIF-selected features ({len(final_features)} features).\n")
            else:
                self.log_text.insert(tk.END, "Performing VIF selection...\n")
                cols_for_vif = [c for c in BASE_FEATURES if c not in ['gel_желатин', 'gel_агароза', 'gel_силикагель']]
                X_vif = df_cls[cols_for_vif].dropna()
                final_features = select_features_by_vif(X_vif, threshold=10)
                for g in ['gel_желатин', 'gel_агароза', 'gel_силикагель']:
                    if g in BASE_FEATURES and g not in final_features:
                        final_features.append(g)
                joblib.dump(final_features, vif_file)
                self.log_text.insert(tk.END, f"VIF done. Retained {len(final_features)} features.\n")
            self.final_features = final_features

            systems = df_cls['Система'].unique()
            systems_cv = [sys for sys in systems if len(self.df_reg[self.df_reg['Система'] == sys]) >= 5]
            self.log_text.insert(tk.END, f"Systems with >=5 regression points: {len(systems_cv)}\n")

            self.log_text.insert(tk.END, "Starting Optuna optimization...\n")
            self.log_text.see(tk.END)
            best_clf, best_reg = self.run_optuna(df_cls, final_features, systems_cv)

            self.log_text.insert(tk.END, "Training final models on full dataset...\n")
            self.clf, self.reg, self.scaler = self.train_final_models(df_cls, self.df_reg, final_features, best_clf, best_reg)

            joblib.dump(self.clf, 'classifier.joblib')
            joblib.dump(self.reg, 'regressor.joblib')
            joblib.dump(self.scaler, 'scaler.joblib')
            joblib.dump(final_features, 'final_features.joblib')

            self.log_text.insert(tk.END, "Calculating LOO-CV metrics...\n")
            acc, prec, rec, auc, mae, r2 = self.evaluate_loocv(df_cls, final_features, systems_cv, best_clf, best_reg)
            self.log_text.insert(tk.END, f"LOO-CV results:\n")
            self.log_text.insert(tk.END, f"  Classification: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, AUC={auc:.3f}\n")
            if not np.isnan(mae):
                self.log_text.insert(tk.END, f"  Regression: MAE={mae:.4f}, R2={r2:.4f}\n")

            X_all = self.scaler.transform(df_cls[final_features])
            proba_all = self.clf.predict_proba(X_all)[:, 1]
            fpr, tpr, thresholds = roc_curve(df_cls['Кольца'].astype(int), proba_all)
            opt_idx = np.argmax(tpr - fpr)
            self.optimal_threshold = thresholds[opt_idx]
            self.log_text.insert(tk.END, f"Optimal threshold (Youden) = {self.optimal_threshold:.4f}\n")

            self.models_ready = True
            self.log_text.insert(tk.END, "="*60 + "\nMODEL IS READY.\n")
            self.log_text.see(tk.END)

        except Exception as e:
            self.log_text.insert(tk.END, f"ERROR: {e}\n")
            import traceback
            traceback.print_exc()

    def objective(self, trial, X_train, y_train, X_test, y_test, task='classification', n_pos=1, n_neg=1):
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
            if len(pred) == 0:
                return np.nan
            mae = mean_absolute_error(y_test, pred)
            return -mae

    def run_optuna(self, df_clean, feature_list, systems_cv):
        n_pos = (df_clean['Кольца'] == 1).sum()
        n_neg = (df_clean['Кольца'] == 0).sum()
        best_clf = None
        best_reg = None
        default_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
                          'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3}

        for system in tqdm(systems_cv, desc="Optuna LOO"):
            test_idx = df_clean[df_clean['Система'] == system].index
            train_idx = df_clean[~df_clean['Система'].isin([system])].index
            if len(train_idx) == 0:
                continue

            X_train = df_clean.loc[train_idx, feature_list]
            y_train = df_clean.loc[train_idx, 'Кольца'].astype(int)
            X_test = df_clean.loc[test_idx, feature_list]
            y_test = df_clean.loc[test_idx, 'Кольца'].astype(int)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            try:
                study_clf = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
                study_clf.optimize(lambda trial: self.objective(trial, X_train_scaled, y_train, X_test_scaled, y_test,
                                                                'classification', n_pos, n_neg),
                                   n_trials=10, show_progress_bar=False)
                best_clf = study_clf.best_params if len(study_clf.trials) > 0 else default_params
            except Exception:
                best_clf = default_params

            reg_mask = (df_clean['Кольца'] == 1) & (df_clean['p'].notna())
            df_reg_local = df_clean[reg_mask].copy()
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
                        study_reg.optimize(lambda trial: self.objective(trial, Xr_train_scaled, yr_train, Xr_test_scaled, yr_test,
                                                                        'regression'),
                                           n_trials=10, show_progress_bar=False)
                        best_reg = study_reg.best_params if len(study_reg.trials) > 0 else default_params
                    except Exception:
                        best_reg = default_params

        if best_clf is None:
            best_clf = default_params
        if best_reg is None:
            best_reg = default_params
        return best_clf, best_reg

    def train_final_models(self, df_clean, df_reg, feature_list, best_clf, best_reg):
        X_class = df_clean[feature_list]
        y_class = df_clean['Кольца'].astype(int)
        X_reg = df_reg[feature_list]
        y_reg = df_reg['p']

        scaler = StandardScaler()
        X_class_scaled = scaler.fit_transform(X_class)
        X_reg_scaled = scaler.transform(X_reg) if len(X_reg) > 0 else None

        n_pos = (df_clean['Кольца'] == 1).sum()
        n_neg = (df_clean['Кольца'] == 0).sum()
        clf = xgb.XGBClassifier(**best_clf, scale_pos_weight=n_neg/n_pos if n_pos>0 else 1.0,
                                eval_metric='logloss', use_label_encoder=False, random_state=42)
        clf.fit(X_class_scaled, y_class)
        calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
        calibrated_clf.fit(X_class_scaled, y_class)

        if X_reg_scaled is not None:
            reg = xgb.XGBRegressor(**best_reg, random_state=42)
            reg.fit(X_reg_scaled, y_reg)
        else:
            reg = None

        return calibrated_clf, reg, scaler

    def evaluate_loocv(self, df_clean, feature_list, systems_cv, clf_params, reg_params):
        y_true_cls = []
        y_proba_cls = []
        y_true_reg = []
        y_pred_reg = []
        n_pos = (df_clean['Кольца'] == 1).sum()
        n_neg = (df_clean['Кольца'] == 0).sum()

        for system in systems_cv:
            test_idx = df_clean[df_clean['Система'] == system].index
            train_idx = df_clean[~df_clean['Система'].isin([system])].index
            if len(train_idx) == 0:
                continue

            X_train = df_clean.loc[train_idx, feature_list]
            y_train = df_clean.loc[train_idx, 'Кольца'].astype(int)
            X_test = df_clean.loc[test_idx, feature_list]
            y_test = df_clean.loc[test_idx, 'Кольца'].astype(int)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = xgb.XGBClassifier(**clf_params, scale_pos_weight=n_neg/n_pos if n_pos>0 else 1.0,
                                    eval_metric='logloss', use_label_encoder=False, random_state=42)
            clf.fit(X_train_scaled, y_train)
            y_proba = clf.predict_proba(X_test_scaled)[:, 1]
            y_true_cls.extend(y_test)
            y_proba_cls.extend(y_proba)

            reg_mask = (df_clean['Кольца'] == 1) & (df_clean['p'].notna())
            df_reg_local = df_clean[reg_mask].copy()
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
                    reg = xgb.XGBRegressor(**reg_params, random_state=42)
                    reg.fit(Xr_train_scaled, yr_train)
                    y_pred = reg.predict(Xr_test_scaled)
                    y_true_reg.extend(yr_test)
                    y_pred_reg.extend(y_pred)

        acc = accuracy_score(y_true_cls, (np.array(y_proba_cls) >= 0.5).astype(int))
        prec = precision_score(y_true_cls, (np.array(y_proba_cls) >= 0.5).astype(int), zero_division=0)
        rec = recall_score(y_true_cls, (np.array(y_proba_cls) >= 0.5).astype(int))
        auc = roc_auc_score(y_true_cls, y_proba_cls)
        mae = mean_absolute_error(y_true_reg, y_pred_reg) if len(y_true_reg) > 0 else np.nan
        r2 = r2_score(y_true_reg, y_pred_reg) if len(y_true_reg) > 0 else np.nan
        return acc, prec, rec, auc, mae, r2

    def predict_single(self):
        if not self.models_ready:
            messagebox.showwarning("Model not trained", "Load dataset and train first!")
            return
        try:
            system = self.entry_system.get()
            gel_type = self.gel_type_var.get()
            gel_conc = float(self.entry_gel_conc.get()) if 'silica' not in gel_type else 0.0
            c_in = float(self.cond_entries["C_in (M)"].get())
            c_out = float(self.cond_entries["C_out (M)"].get())
            t = float(self.cond_entries["T (C)"].get())
            ph = float(self.cond_entries["pH"].get())
            e = float(self.cond_entries["E (V/m)"].get())
            z_cat = float(self.ion_entries["z+"].get())
            z_an = float(self.ion_entries["z-"].get())
            r_cat = float(self.ion_entries["r+ (A)"].get())
            r_an = float(self.ion_entries["r- (A)"].get())
            D_cat = float(self.ion_entries["D+ (water)"].get())
            D_an = float(self.ion_entries["D- (water)"].get())
            nu_in = float(self.ion_entries["nu_in"].get())
            nu_out = float(self.ion_entries["nu_out"].get())

            if gel_type == "gelatin":
                gel_str = f"желатин {gel_conc}%"
            elif gel_type == "agarose":
                gel_str = f"агароза {gel_conc}%"
            else:
                gel_str = "силикагель"

            row = pd.Series({
                'Система': system,
                'Гель': gel_str,
                'C_in_M': c_in,
                'C_out_M': c_out,
                'T_C': t,
                'pH': ph,
                'E_В_м': e,
                'Кольца': 1,
                'p': np.nan
            })
            params = {
                'z_cat': z_cat, 'z_an': z_an,
                'r_cat': r_cat, 'r_an': r_an,
                'D_cat_w': D_cat, 'D_an_w': D_an,
                'nu_in': nu_in, 'nu_out': nu_out
            }
            feats = compute_manual_features_from_dict(row, params)
            if feats is None:
                self.log_text.insert(tk.END, "ERROR: Could not compute features.\n")
                return

            X_input = np.array([[feats[f] for f in self.final_features]])
            X_scaled = self.scaler.transform(X_input)
            proba = self.clf.predict_proba(X_scaled)[0, 1]
            pred_class = "Rings will form" if proba >= self.optimal_threshold else "No rings"
            p_val = self.reg.predict(X_scaled)[0] if self.reg and pred_class == "Rings will form" else "—"

            self.exp_counter += 1
            res = (f"[Exp. #{self.exp_counter}] {system} | "
                   f"P={proba:.3f} | {pred_class} | p={p_val}\n")
            self.log_text.insert(tk.END, res)
            self.log_text.see(tk.END)

            for key in self.cond_entries:
                self.cond_entries[key].delete(0, tk.END)
            self.cond_entries["C_in (M)"].insert(0, "0.1")
            self.cond_entries["C_out (M)"].insert(0, "0.1")
            self.cond_entries["T (C)"].insert(0, "22.0")
            self.cond_entries["pH"].insert(0, "5.5")
            self.cond_entries["E (V/m)"].insert(0, "0")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

    def generate_heatmap(self):
        if not self.models_ready:
            messagebox.showwarning("Model not trained", "Load dataset and train first!")
            return
        try:
            system = self.entry_system.get()
            gel_type = self.gel_type_var.get()
            gel_conc = float(self.entry_gel_conc.get()) if 'silica' not in gel_type else 0.0
            t = float(self.cond_entries["T (C)"].get())
            ph = float(self.cond_entries["pH"].get())
            e = float(self.cond_entries["E (V/m)"].get())
            z_cat = float(self.ion_entries["z+"].get())
            z_an = float(self.ion_entries["z-"].get())
            r_cat = float(self.ion_entries["r+ (A)"].get())
            r_an = float(self.ion_entries["r- (A)"].get())
            D_cat = float(self.ion_entries["D+ (water)"].get())
            D_an = float(self.ion_entries["D- (water)"].get())
            nu_in = float(self.ion_entries["nu_in"].get())
            nu_out = float(self.ion_entries["nu_out"].get())

            if gel_type == "gelatin":
                gel_str = f"желатин {gel_conc}%"
            elif gel_type == "agarose":
                gel_str = f"агароза {gel_conc}%"
            else:
                gel_str = "силикагель"

            params = {
                'z_cat': z_cat, 'z_an': z_an,
                'r_cat': r_cat, 'r_an': r_an,
                'D_cat_w': D_cat, 'D_an_w': D_an,
                'nu_in': nu_in, 'nu_out': nu_out
            }

            steps = 80
            c_min, c_max = 0.001, 1.0
            c_ins = np.linspace(c_min, c_max, steps)
            c_outs = np.linspace(c_min, c_max, steps)
            grid_proba = np.zeros((steps, steps))
            grid_p = np.zeros((steps, steps))

            for i, ci in enumerate(c_ins):
                for j, co in enumerate(c_outs):
                    row = pd.Series({
                        'Система': system,
                        'Гель': gel_str,
                        'C_in_M': ci,
                        'C_out_M': co,
                        'T_C': t,
                        'pH': ph,
                        'E_В_м': e,
                        'Кольца': 1,
                        'p': np.nan
                    })
                    feats = compute_manual_features_from_dict(row, params)
                    if feats is None:
                        continue
                    X_input = np.array([[feats[f] for f in self.final_features]])
                    X_scaled = self.scaler.transform(X_input)
                    prob = self.clf.predict_proba(X_scaled)[0, 1]
                    p_pred = self.reg.predict(X_scaled)[0] if self.reg else 0
                    grid_proba[j, i] = prob
                    grid_p[j, i] = p_pred

            top = tk.Toplevel(self.root)
            top.title("Concentration Map")
            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(grid_proba, origin='lower', extent=[c_min, c_max, c_min, c_max],
                           cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            ax.set_title(f"Ring formation probability for {system}\nGel: {gel_type}  |  T={t}°C, pH={ph}, E={e} V/m")
            ax.set_xlabel("C_in (M)")
            ax.set_ylabel("C_out (M)")

            if hasattr(self, 'optimal_threshold'):
                CS = ax.contour(c_ins, c_outs, grid_proba, levels=[self.optimal_threshold],
                                colors='blue', linestyles='dashed', linewidths=2)
                ax.clabel(CS, inline=True, fontsize=10, fmt=f'thr={self.optimal_threshold:.2f}')

            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Probability of rings')

            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Heatmap generation failed: {e}")
            import traceback
            traceback.print_exc()

    def find_concentrations(self, target_p, bounds=(0.01, 0.5), popsize=15, maxiter=100):
        if not self.models_ready:
            raise ValueError("Model not trained!")

        system = self.entry_system.get()
        gel_type = self.gel_type_var.get()
        gel_conc = float(self.entry_gel_conc.get()) if 'silica' not in gel_type else 0.0
        t = float(self.cond_entries["T (C)"].get())
        ph = float(self.cond_entries["pH"].get())
        e = float(self.cond_entries["E (V/m)"].get())
        z_cat = float(self.ion_entries["z+"].get())
        z_an = float(self.ion_entries["z-"].get())
        r_cat = float(self.ion_entries["r+ (A)"].get())
        r_an = float(self.ion_entries["r- (A)"].get())
        D_cat = float(self.ion_entries["D+ (water)"].get())
        D_an = float(self.ion_entries["D- (water)"].get())
        nu_in = float(self.ion_entries["nu_in"].get())
        nu_out = float(self.ion_entries["nu_out"].get())

        if gel_type == "gelatin":
            gel_str = f"желатин {gel_conc}%"
        elif gel_type == "agarose":
            gel_str = f"агароза {gel_conc}%"
        else:
            gel_str = "силикагель"

        params = {
            'z_cat': z_cat, 'z_an': z_an,
            'r_cat': r_cat, 'r_an': r_an,
            'D_cat_w': D_cat, 'D_an_w': D_an,
            'nu_in': nu_in, 'nu_out': nu_out
        }

        def objective(x):
            ci, co = x[0], x[1]
            row = pd.Series({
                'Система': system,
                'Гель': gel_str,
                'C_in_M': ci,
                'C_out_M': co,
                'T_C': t,
                'pH': ph,
                'E_В_м': e,
                'Кольца': 1,
                'p': np.nan
            })
            feats = compute_manual_features_from_dict(row, params)
            if feats is None:
                return 1e10
            X_input = np.array([[feats[f] for f in self.final_features]])
            X_scaled = self.scaler.transform(X_input)
            proba = self.clf.predict_proba(X_scaled)[0, 1]
            p_pred = self.reg.predict(X_scaled)[0] if self.reg else 0
            penalty = 0
            if proba < self.optimal_threshold:
                penalty = 10.0 * (self.optimal_threshold - proba) ** 2
            return (p_pred - target_p) ** 2 + penalty

        bounds = [(bounds[0], bounds[1]), (bounds[0], bounds[1])]
        result = differential_evolution(objective, bounds, popsize=popsize, maxiter=maxiter, seed=42)

        if result.success:
            ci_opt, co_opt = result.x
            row_check = pd.Series({
                'Система': system,
                'Гель': gel_str,
                'C_in_M': ci_opt,
                'C_out_M': co_opt,
                'T_C': t,
                'pH': ph,
                'E_В_м': e,
                'Кольца': 1,
                'p': np.nan
            })
            feats_check = compute_manual_features_from_dict(row_check, params)
            X_check = np.array([[feats_check[f] for f in self.final_features]])
            X_scaled_check = self.scaler.transform(X_check)
            proba_check = self.clf.predict_proba(X_scaled_check)[0, 1]
            p_check = self.reg.predict(X_scaled_check)[0] if self.reg else 0
            return {
                'C_in': ci_opt,
                'C_out': co_opt,
                'p_pred': p_check,
                'probability': proba_check,
                'success': True
            }
        else:
            return {'success': False}

    def find_concentrations_gui(self):
        if not self.models_ready:
            messagebox.showwarning("Model not trained", "Load dataset and train first!")
            return

        target_p = simpledialog.askfloat("Target p", "Enter desired spacing factor p:", minvalue=0.8, maxvalue=1.3)
        if target_p is None:
            return

        try:
            result = self.find_concentrations(target_p)
            if result['success']:
                msg = (f"Found concentrations for p = {target_p:.3f}:\n"
                       f"C_in = {result['C_in']:.4f} M\n"
                       f"C_out = {result['C_out']:.4f} M\n"
                       f"Predicted p = {result['p_pred']:.4f}\n"
                       f"Probability = {result['probability']:.3f}")
                messagebox.showinfo("Concentrations found", msg)
                self.cond_entries["C_in (M)"].delete(0, tk.END)
                self.cond_entries["C_in (M)"].insert(0, f"{result['C_in']:.4f}")
                self.cond_entries["C_out (M)"].delete(0, tk.END)
                self.cond_entries["C_out (M)"].insert(0, f"{result['C_out']:.4f}")
            else:
                messagebox.showwarning("Not found", "Could not find suitable concentrations. Try different bounds or target p.")
        except Exception as e:
            messagebox.showerror("Error", f"Search failed: {e}")

    def show_diagnostics(self):
        if not self.models_ready or self.df_clean is None:
            messagebox.showwarning("No model", "Train or load model first!")
            return

        top = tk.Toplevel(self.root)
        top.title("Model Diagnostics")

        notebook = ttk.Notebook(top)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        frame1 = ttk.Frame(notebook)
        notebook.add(frame1, text="ROC/PR/Calibration")
        self.plot_roc_pr_calib(frame1)

        frame2 = ttk.Frame(notebook)
        notebook.add(frame2, text="Regression")
        self.plot_regression(frame2)

        frame3 = ttk.Frame(notebook)
        notebook.add(frame3, text="Feature Importance")
        self.plot_feature_importance(frame3)

        frame4 = ttk.Frame(notebook)
        notebook.add(frame4, text="SHAP Summary")
        self.plot_shap(frame4)

    def plot_roc_pr_calib(self, parent):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        X_all = self.scaler.transform(self.df_clean[self.final_features])
        y_true = self.df_clean['Кольца'].astype(int)
        y_proba = self.clf.predict_proba(X_all)[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        axes[0].plot(fpr, tpr, label=f'AUC={auc:.3f}')
        axes[0].plot([0,1], [0,1], 'k--')
        axes[0].set_xlabel('FPR')
        axes[0].set_ylabel('TPR')
        axes[0].set_title('ROC')
        axes[0].legend()

        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        axes[1].plot(rec, prec)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall')

        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        axes[2].plot(prob_pred, prob_true, marker='o', label='Model')
        axes[2].plot([0,1], [0,1], 'k--', label='Perfect')
        axes[2].set_xlabel('Mean predicted prob')
        axes[2].set_ylabel('Fraction of positives')
        axes[2].set_title('Calibration')
        axes[2].legend()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_regression(self, parent):
        df_reg_plot = self.df_clean[self.df_clean['p'].notna()].copy()
        X_reg = self.scaler.transform(df_reg_plot[self.final_features])
        y_pred = self.reg.predict(X_reg)
        y_true = df_reg_plot['p'].values

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axes[0].set_xlabel('True p')
        axes[0].set_ylabel('Predicted p')
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        axes[0].set_title(f'MAE={mae:.3f}, R2={r2:.3f}')

        residuals = y_pred - y_true
        axes[1].hist(residuals, bins=20, edgecolor='black')
        axes[1].axvline(0, color='r', linestyle='--')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals')

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_feature_importance(self, parent):
        importance = self.clf.estimators_[0].get_booster().get_score(importance_type='gain')
        importances = np.array([importance.get(f'f{i}', 0) for i in range(len(self.final_features))])
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(self.final_features)), importances[indices], align='center')
        ax.set_yticks(range(len(self.final_features)))
        ax.set_yticklabels([self.final_features[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Gain importance')
        ax.set_title('Feature Importance (XGBoost gain)')

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_shap(self, parent):
        try:
            import shap
            X_sample = self.scaler.transform(self.df_clean[self.final_features].iloc[:100])
            explainer = shap.TreeExplainer(self.clf.estimators_[0])
            shap_values = explainer.shap_values(X_sample)

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=self.final_features, show=False)
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            tk.Label(parent, text=f"SHAP error: {e}").pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
