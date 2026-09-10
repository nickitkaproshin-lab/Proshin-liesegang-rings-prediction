# Proshin-liesegang-rings-prediction

**Author**: Nikita Proshin, Russia, Moscow, 2026  
**GitHub**: [nickitkaproshin-lab/Proshin-liesegang-rings-prediction](https://github.com/nickitkaproshin-lab/Proshin-liesegang-rings-prediction)

This repository contains the code and data for a **two-regime machine learning model** that predicts the formation of periodic precipitates (Liesegang rings) in gels and their geometric parameter — the spacing coefficient `p`.  

The model is based on a **two-stage XGBoost architecture**:  
- A **classifier** trained on **19 physicochemical descriptors** (after VIF-based feature selection) derived from diffusion, kinetic, and thermodynamic criteria, predicting ring *occurrence*.  
- A **regressor** trained on **only 3 electrokinetic features** (`Pe`, `pH`, `pH·E`) derived from the ablation analysis, predicting ring *geometry* (spacing factor `p`).

This two-regime design was motivated by a fundamental physical finding: ring **nucleation** and ring **geometry** are governed by **independent physicochemical mechanisms**. The model was validated using **Leave-One-System-Out (LOO-CV)** cross-validation and Bootstrap resampling, achieving state-of-the-art predictive performance.

---

## Related Publication

The physical mechanism behind the two-regime architecture is described in the following preprint:

> Proshin, N. V. *Liesegang Rings Have Two Different Physics: Decoupling Nucleation from Geometry via Machine Learning*. Zenodo, 2026. DOI: [10.5281/zenodo.22682566](https://doi.org/10.5281/zenodo.22682566)

The preprint presents **five independent proofs of physical decoupling** (ablation, mutual information, counterfactual analysis, classical-formula benchmarks, and non-parametric statistics) and the full physicochemical interpretation.

---

## Repository Structure

- `Prediction of Liesegang Rings Machine python.py` — main code
- `app.py` — GUI application (two-regime model v3.0)
- `data_loader.py` — data loading and reference dictionaries
- `feature_engineering.py` — physicochemical feature calculation
- `config.py` — model configuration
- `model_train.py` — training and validation functions
- `liesegang_dataset.xlsx` — collected dataset (237 experiments, 194 with rings, 43 without; 133 with known `p`)
- `requirements.txt` — Python dependencies
- `README.md` — this file

---

## Running the Model

### Python Script
1. Install dependencies: `pip install -r requirements.txt`
2. Run `python "Prediction of Liesegang Rings Machine python.py"`
3. Upload `liesegang_dataset.xlsx` when prompted
4. Wait for feature calculation and model training (about 10–20 seconds)
5. Enter all required parameters and click **"PREDICT"**

### Standalone Executable (No Python required)

If you don't want to install Python and all dependencies, download the ready-to-run executable:

[Download Proshin_Liesegang_Predictor_Machine.exe (ZIP archive, ~450 MB)](https://disk.yandex.ru/d/cPIJQzZGf9UhVg)

**Instructions:**
1. Follow the link and download the ZIP archive.
2. Extract the archive to any folder.
3. Run `Proshin_Liesegang_Predictor_Machine.exe`.
4. Load the dataset `liesegang_dataset.xlsx` via the **"Load Dataset and Train"** button.
5. Wait for training to complete (30–60 seconds).
6. Enter experimental parameters and click **"Predict"**.
7. Use **"Concentration Map"** to visualize ring formation zones.

---

## Input Parameters (what the user must enter)

### Reference Ion Data (found in physicochemical property tables)

| Parameter | Symbol | Example for Ag⁺ | Example for Cr₂O₇²⁻ | Source |
|-----------|--------|-----------------|----------------------|--------|
| D_water of INNER ion (m²/s) | D_in | 1.65e-9 | — | CRC Handbook |
| D_water of OUTER ion (m²/s) | D_out | — | 1.00e-9 | CRC Handbook |
| Radius of INNER ion (Å) | r_in | 1.15 | — | Shannon (1976) |
| Radius of OUTER ion (Å) | r_out | — | 2.50 | Shannon (1976) |
| Charge of INNER ion | z_in | 1 | — | Periodic table |
| Charge of OUTER ion | z_out | — | 2 | Periodic table |
| Stoichiometric coefficient of INNER ion | nu_in | 2 (for Ag₂Cr₂O₇) | — | Precipitate formula |
| Stoichiometric coefficient of OUTER ion | nu_out | — | 1 (for Ag₂Cr₂O₇) | Precipitate formula |

**Important:** The stoichiometric coefficients nu_in and nu_out are the numbers of ions in the precipitate formula. For example, for Ag₂Cr₂O₇: nu_in (Ag⁺) = 2, nu_out (Cr₂O₇²⁻) = 1.

### Experimental Parameters

| Parameter | Symbol | Example | Notes |
|-----------|--------|---------|-------|
| C_in (M) | C_in | 0.1 | Concentration of the inner electrolyte (in the gel) |
| C_out (M) | C_out | 0.1 | Concentration of the outer electrolyte (solution on top) |
| Gel type | gel_type | gelatin | gelatin / agarose / silica gel |
| Gel concentration (%) | gel_conc | 5 | For gelatin usually 3–10%, for agarose 1–2% |
| Temperature (°C) | T | 22 | Most experiments at room temperature |
| pH | pH | 7.0 | Important for hydroxides and phosphates |
| E (V/m) | E | 0 | Constant electric field (0 if absent) |

---

## Interpretation of Results

### Probability of ring formation P

A calibrated probability from 0 to 1. The optimal decision threshold (determined by Youden's index on the LOO-CV predictions) is **P ≥ 0.864**. If P exceeds this threshold, the model predicts **rings will form**; otherwise, rings are absent.

**Classifier performance** (LOO-CV, 19 features):
- Accuracy = 0.781
- Precision = 0.933
- Recall = 0.789
- AUC-ROC = 0.801

### Spacing coefficient p

Predicted ratio of distances between adjacent rings (only when rings are predicted). **The regressor uses only 3 electrokinetic features** (`Pe`, `pH`, `pH·E`).

**Regression performance** (LOO-CV on 132 positive examples, 3 features):
- MAE = 0.0205
- **R² = 0.658** (explains ~66% of variance)
- Bootstrap (1000 iterations) mean R² = 0.712 (95% CI: 0.177–0.822)

**Note**: the 3-feature regressor *outperforms* the full 23-feature regressor (R² = 0.658 vs. 0.552). Adding the remaining 20 physicochemical descriptors degrades performance, providing independent confirmation of the physical decoupling. The closer p is to 1, the more evenly spaced the rings.

---

## Example Prediction

For the Ag₂Cr₂O₇ system (5% gelatin, C_in = 0.1 M, C_out = 0.1 M, T = 22°C, pH = 5.5, E = 0):

- Enter: D_in = 1.65e-9, D_out = 1.00e-9, r_in = 1.15, r_out = 2.50, z_in = 1, z_out = 2, nu_in = 2, nu_out = 1.
- Model output: P ≈ 0.88 (> 0.864) → **rings will form**, p ≈ 1.09.

---

## Model Development and Validation

- **Dataset**: 237 independent experiments compiled from 74 literature sources (1896–2025).
- **Feature engineering**: 21 initial physicochemical descriptors including modified Jablczynski criterion (X_corr), Damköhler (Da), Péclet (Pe), ionic strength, supersaturation, and cross-interactions (lnX·pH, pH·E, etc.).
- **Feature selection**: Variance Inflation Factor (VIF) eliminated multicollinear variables, leaving **19 features for the classifier**. The regressor uses a fixed set of **3 electrokinetic features** (Pe, pH, pH·E), determined by ablation analysis.
- **Two-regime architecture (v3.0)**: The model was split into two independent pipelines based on the discovery that ring nucleation and ring geometry are physically decoupled:
  - **Classifier** — 19 features (gel structure, kinetics, thermodynamics, cross-interactions) → predicts ring *occurrence*.
  - **Regressor** — 3 electrokinetic features (Pe, pH, pH·E) → predicts the spacing factor `p`.
- **Validation**: Rigorous Leave-One-System-Out cross-validation (each chemical system held out in turn) and Bootstrap stability analysis (1000 iterations).
- **Comparison with classical theories**: The model significantly outperforms the Matalon–Packter, Keller–Rubinow, Lagzi–Izsák "universal law", and spinodal decomposition models (best classical R² = 0.220 vs. our R² = 0.658).

### Five independent proofs of physical decoupling

The physical decoupling that motivated the two-regime architecture was confirmed by five independent analytical methods (see preprint for details):

1. **Ablation** — gel-only classifier achieves AUC = 0.688 (occurrence), electro-only regressor achieves R² = 0.629 (geometry); 95% CIs do not overlap.
2. **Mutual information** — MI(gel; rings) = 0.035 vs. MI(electro; rings) = 0.010; MI(electro; p) = 0.102 vs. MI(gel; p) = 0.053.
3. **Counterfactual analysis** — electric field changes p but not the ring probability; gel type changes the ring probability but not p.
4. **Classical-formula benchmarks** — a model trained only on classical descriptors (Ostwald, Jablczynski, Matalon–Packter, Lagzi–Izsák) achieves AUC = 0.484 and R² = 0.116.
5. **Non-parametric statistics** — gel effect on rings: p = 3.1×10⁻⁹; Péclet effect on rings: p = 0.24 (n.s.); Péclet effect on spacing: p = 3.1×10⁻⁵.

---

## Model Limitations

- The classifier is trained on an imbalanced dataset (194 positive, 43 negative); however, the use of `scale_pos_weight` and probability calibration mitigates bias.
- The model does not account for gel aging, impurities, complexation, or redox reactions — these may affect real experiments.
- Best predictions are obtained for ionic precipitation in gelatin gels with concentrations 0.01–0.5 M; extrapolation to mixed gels or extreme pH should be done with caution.
- Applicable only to gel-based ionic precipitation systems; do not use for gas-phase or gel-free media.
- The two-regime architecture reflects a fundamental physical decoupling: gel structure controls *whether* rings form, while electric field and pH control *how far apart* they are. This has been confirmed on 237 experiments and 5 independent analytical methods.

---

## License

MIT License. The code and data are open for use and modification with attribution.

---

## References

The dataset is based on a comprehensive literature survey (1896–2025). The full list of 74 sources is provided in the article and in the code repository. Key references are:

Das, I.; Pushkarna, A.; Agrawal, N. R. Chemical instability and periodic precipitation of copper chromate in gel media. J. Indian Chem. Soc. 2004, *81*, 581–586.  
Das, I.; Pushkarna, A.; Chand, S. Chemical instability and periodic precipitation of CuCrO₄ in batch and flow reactors. Indian J. Chem. A 2019, *58*, 341–348.  
Morsali, M.; Ghiaci, M. Liesegang Rings in the Cu-Cr System. J. Colloid Interface Sci. 2014, *418*, 254–259. DOI: 10.1016/j.jcis.2013.12.018.  
Swami, S. N.; Kant, K. Liesegang rings of copper chromate in gelatin gel. Colloid Polym. Sci. 2005, *209*, 56–57. DOI: 10.1007/BF01500047.  
Sultan, R.; Sadek, S. Patterning Trends and Chaotic Behavior in Co²⁺/NH₄OH Liesegang Systems. J. Phys. Chem. 1996, *100*, 16912–16920. DOI: 10.1021/jp961239d.  
Kant, K. Liesegang rings of lead chromate. Part I. Kolloid-Z. Z. Polym. 1963, *189*, 155–156. DOI: 10.1007/BF01499512.  
Das, I.; Pushkarna, A.; Lall, K. Light Induced Liesegang Type Patterns in Batch and Flow Reactors. J. Sci. Ind. Res. 2001, *60*, 234–238.  
Bohner, B.; Schuszter, G.; Lagzi, I. Controlling Pattern Formation in the Cobalt-Hydroxide System. J. Phys. Chem. A 2016, *120*, 5569–5575. DOI: 10.1021/acs.jpca.6b04684.  
George, J.; Varghese, G. Studies on Liesegang rings of cobalt hydroxide in 1% agar gel medium. J. Mol. Liq. 2015, *204*, 205–209. DOI: 10.1016/j.molliq.2015.01.031.  
George, J.; Varghese, G. Periodic precipitation of cobalt hydroxide in agar gel: Effect of ionic strength. J. Mol. Liq. 2017, *241*, 37–42. DOI: 10.1016/j.molliq.2017.06.008.  
George, J.; Varghese, G. Liesegang Patterns in Chitosan Hydrogels. J. Mater. Sci. 2006, *41*, 2535–2542. DOI: 10.1007/s10853-006-7859-7.  
Badr, L.; Sultan, R. Ring Morphology and pH Effects in 2D and 1D Co(OH)₂ Liesegang Systems. J. Phys. Chem. A 2009, *113*, 6264–6270. DOI: 10.1021/jp9032349.  
Shreif, Z.; Mandalian, L.; Abi-Haydar, A.; Sultan, R. Taming ring morphology in 2D Co(OH)₂ Liesegang patterns. Chem. Phys. Lett. 2010, *492*, 35–39. DOI: 10.1016/j.cplett.2010.04.030.  
Sultan, R.; Panjarian, S. Morphology of a 2D Mg²⁺/NH₄OH Liesegang pattern in zero, positive and negative radial electric field. Chem. Phys. Lett. 2010, *492*, 35–39. DOI: 10.1016/j.cplett.2010.04.030.  
Badr, L.; El-Rassy, H.; El-Joubeily, S.; Sultan, R. Morphology of a 2D Mg²⁺/NH₄OH Liesegang pattern in zero, positive and negative radial electric field. Chem. Phys. Lett. 2010. DOI: 10.1016/j.cplett.2010.04.030.  
Sultan, R.; Halabieh, R. Effect of an electric field on propagating Co(OH)₂ Liesegang patterns. Chem. Phys. Lett. 2000, *329*, 217–222. DOI: 10.1016/S0009-2614(00)00991-9.  
Meng, X.; et al. Polymorphs Co hydroxides formed between hydrazine and Co²⁺ as Liesegang bands in semisolid agar gel. J. Mol. Liq. 2018, *268*, 190–196. DOI: 10.1016/j.molliq.2018.07.037.  
Traetteberg, J.; Devik, O. Formation of calcium phosphate studied with Liesegang's rings. Kolloid-Z. Z. Polym. 1962, *180*, 35–41. DOI: 10.1007/BF01500699.  
Devik, O. Formation of calcium phosphate studied with Liesegang's Rings. III. Mechanism of ring formation. Kolloid-Z. 1962, *181*, 33–38. DOI: 10.1007/BF01500699.  
Kibalczyc, W.; Sokołowski, T.; Wiktorowska, B. Growth of calcium phosphate crystals in silica gel. Cryst. Res. Technol. 1984, *19*, 27–32. DOI: 10.1002/crat.2170190112.  
Karoly, Z.; et al. Synthesis and characterization of carbonated fluorapatite-gelatine nanocomposites within Liesegang bands. Mater. Sci. Eng. C 2010, *30*, 672–678. DOI: 10.1016/j.msec.2010.02.015.  
Rosseeva, E. V.; et al. Synthesis, characterization, and morphogenesis of carbonated fluorapatite-gelatine nanocomposites: A complex biomimetic approach toward the mineralization of hard tissues. Chem. Mater. 2008, *20*, 6003–6013. DOI: 10.1021/cm8005748.  
Eltantawy, M. M.; Belokon, M. A.; Belogub, E. V.; Ledovich, O. I.; Skorb, E. V.; Ulasevich, S. A. Self-Assembled Liesegang Rings of Hydroxyapatite for Cell Culturing. Adv. NanoBiomed Res. 2021, *1*, 2000048. DOI: 10.1002/anbr.202000048.  
George, J.; Varghese, G. Studies on Liesegang rings of copper molybdate in agar gel medium. J. Mol. Liq. 2013, *178*, 132–136. DOI: 10.1016/j.molliq.2012.11.020.  
Periodic crystallization of barium oxalate in silica hydrogel. Bull. Mater. Sci. 2020, *43*, 123. DOI: 10.1007/s12034-020-02095-8.  
Mechanism of Formation of Cadmium Oxalate Liesegang Rings. J. Colloid Interface Sci. 1994, *167*, 345–351. DOI: 10.1006/jcis.1994.1369.  
Kant, K. Liesegang rings of lead iodide. Part II. Kolloid-Z. Z. Polym. 1963, *189*, 155–156. DOI: 10.1007/BF01499513.  
Sakamoto, S.; Itatani, M.; Tsukada, K.; Nabika, H. Regular-Type Liesegang Pattern of AgCl in a One-Dimensional System. Materials 2021, *14*, 1526. DOI: 10.3390/ma14061526.  
Badr, L.; Toramaru, A.; Sultan, R. Experimental pattern transitions in a Liesegang system. Physica D 2003, *183*, 133–140. DOI: 10.1016/S0167-2789(03)00139-8.  
Kumar, P.; Karmakar, S. Obstruction scaling model for the diffusion of the outer electrolyte leading to Liesegang patterns of (AgNO₃ + KCl) system in agarose hydrogel. Chem. Pap. 2021, *75*, 329–340. DOI: 10.1007/s11696-020-01304-2.  
Kant, K. Liesegang rings of lead chromate. Part II. Kolloid-Z. Z. Polym. 1963, *189*, 155–156. DOI: 10.1007/BF01499512.  
Mehta, B. M.; Kant, K. Formation of Liesegang rings of copper sulphide. Kolloid-Z. Z. Polym. 1966, *209*, 54–56. DOI: 10.1007/BF01500046.  
Palaniandavar, N.; Gnanam, F. D.; Ramasamy, P. Diffusion controlled autocatalytic growth of revert periodic precipitation of cadmium sulphide in lyophillic colloid. J. Chem. Phys. 1984, *80*, 3448–3455. DOI: 10.1063/1.447103.  
Sultan, R.; Ortoleva, P. Periodic and Aperiodic Precipitation Patterns. J. Chem. Phys. 1990, *92*, 220–228. DOI: 10.1063/1.458435.  
Sultan, R.; Ortoleva, P. Periodic and aperiodic macroscopic patterning in two precipitate post-nucleation systems. Physica D 1993, *63*, 163–173. DOI: 10.1016/0167-2789(93)90154-7.  
The Precipitation of Strontium Sulfate in Gels. Digital Library UNT. https://digital.library.unt.edu/ (accessed 2026-04-24).  
Estimation of diffusion coefficient of lanthanum ions from one-dimensional Liesegang formation. INIS-MF-10571, 1985. INIS Repository.  
Growth of mixed rare-earth tartrate crystals from silica-gels. University of Bologna, Dipartimento di Chimica G. Ciamician, 1990.  
Al-Ghoul, M.; Ammar, M.; Al-Kaysi, R. O. Pattern Selection in Three-Precipitate Liesegang Systems. ACS Omega 2024, *9*, 43635–43641. DOI: 10.1021/acsomega.4c05000.  
Liesegang pattern formation by gas diffusion in silica aerogels. J. Non-Cryst. Solids 1998, *225*, 69–73. DOI: 10.1016/S0022-3093(98)00111-4.  
Liu, W. Y. [某些混合无机盐体系形成 Liesegang 环的研究]. J. Ningxia Univ. (Nat. Sci. Ed.) 1993, *14*, 45–49.  
Das, I.; et al. Studies on mixed metal chromate Liesegang systems. J. Indian Chem. Soc. 2000, *77*, 241–243.  
Msharrafieh, M.; et al. Front propagation in patterned precipitation. 3. Composition variations in two-precipitate stratum dynamics. J. Phys. Chem. A 2007, *111*, 6967–6976. DOI: 10.1021/jp0712345.  
Sultan, R.; et al. Liesegang Ring Type Structures and Bifurcation in Solid-Vapor and Liquid Phase Reactions between Cobalt Nitrate and Ammonium Hydroxide. J. Colloid Interface Sci. 1997, *192*, 420–431. DOI: 10.1006/jcis.1997.5035.  
Arteaga-Larios, F.; Sheu, E. Y.; Perez, E. Asphaltene Flocculation, Precipitation, and Liesegang Ring. Energy Fuels 2004, *18*, 132–139. DOI: 10.1021/ef030108p.  
Spotz, E. L. Some properties of gases and gaseous reactions. Ph.D. Dissertation, University of Wisconsin, Madison, 1950.  
Hedges, E. S. Liesegang Rings and Other Periodic Structures; Chapman and Hall: London, 1932.  
Müller, S. C.; Kai, S.; Ross, J. Periodic precipitation patterns in the presence of concentration gradients. Science 1982, *216*, 635–637. DOI: 10.1126/science.216.4546.635.  
Kai, S.; Müller, S. C. Spatial and Temporal Patterns in Precipitation Reactions. Science 1985, *229*, 1015–1019. DOI: 10.1126/science.229.4717.1015.

---

## Links

- Dataset: `liesegang_dataset.xlsx`
- Code: `Prediction of Liesegang Rings Machine python.py`
- Executable: `Proshin_Liesegang_Predictor_Machine.exe` (in ZIP archive from [Yandex Disk](https://disk.yandex.ru/d/cPIJQzZGf9UhVg))
- Repository: [nickitkaproshin-lab/Proshin-liesegang-rings-prediction](https://github.com/nickitkaproshin-lab/Proshin-liesegang-rings-prediction)
- Preprint: [10.5281/zenodo.22682566](https://doi.org/10.5281/zenodo.22682566)
