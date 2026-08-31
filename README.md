# Liesegang Rings Prediction Model

**Author**: Nikita Proshin, Russia, Moscow, 2026

This repository contains the code and data for a two‑stage machine learning model that predicts the formation of periodic precipitates (Liesegang rings) in gels and their geometric parameter — the spacing coefficient `p`.  
The model is based on an **XGBoost** algorithm and uses **15 physicochemical descriptors** (after VIF‑based feature selection) derived from diffusion, kinetic, and thermodynamic criteria.  
The final model was validated using **Leave‑One‑System‑Out (LOO‑CV)** cross‑validation and Bootstrap resampling, achieving state‑of‑the‑art predictive performance.

---

## Repository Structure

- Prediction of Liesegang Rings Machine.py — main code
- `liesegang_dataset.xlsx` — collected dataset (237 experiments, 194 with rings, 43 without; 133 with known `p`)
- `README.md` — this file
- (Optional) `app.py` — desktop GUI application for local predictions (if included)

---

## Running the Model

1. Open `Prediction of Liesegang Rings Machine.py`
2. Run the first cell — a button for uploading the dataset will appear.
3. Upload the file `liesegang_dataset.xlsx`.
4. Wait for feature calculation and model training (about 10–20 seconds).
5. After the interactive form appears, choose the mode:
   - **Use trained model** — prediction based on the trained coefficients.
   - **Manual calculation (custom coefficients)** — allows you to enter any classifier and regression coefficients.
6. Enter all required parameters (see below) and click **“PREDICT”**.

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

- **Probability of ring formation P** — a calibrated probability from 0 to 1.  
  The optimal decision threshold (determined by Youden’s index) is **P ≥ 0.8363**.  
  If P exceeds this threshold, the model predicts **rings will form**; otherwise, rings are absent.  
  **Classifier performance** (LOO‑CV):  
  - Accuracy = 0.769  
  - Precision = 0.918  
  - Recall = 0.796  
  - AUC‑ROC = 0.767  

- **Spacing coefficient p** — predicted ratio of distances between adjacent rings (only when rings are predicted).  
  **Regression performance** (LOO‑CV on 133 positive examples):  
  - MAE = 0.0203  
  - R² = 0.676 (explains ~68% of variance)  
  - Bootstrap (100 repeats) mean R² = 0.493 (95% CI: 0.11–0.75).  
  The closer p is to 1, the more evenly spaced the rings.

---

## Example Prediction

For the Ag₂Cr₂O₇ system (5% gelatin, C_in=0.1 M, C_out=0.1 M, T=22°C, pH=7, E=0):  
- Enter: D_in=1.65e-9, D_out=1.00e-9, r_in=1.15, r_out=2.50, z_in=1, z_out=2, nu_in=2, nu_out=1.  
- Model output: P ≈ 0.96 (> 0.8363) → **rings will form**, p ≈ 1.075.

---

## Model Development and Validation

- **Dataset**: 237 independent experiments compiled from 74 literature sources (1896–2025).  
- **Feature engineering**: 21 initial physicochemical descriptors including modified Jablczynski criterion (X_corr), Damköhler (Da), Péclet (Pe), ionic strength, supersaturation, and cross‑interactions (lnX·pH, pH·E, etc.).  
- **Feature selection**: Variance Inflation Factor (VIF) eliminated multicollinear variables, leaving **15 features** (including categorical gel‑type indicators, which were retained for their physical significance).  
- **Validation**: Rigorous Leave‑One‑System‑Out cross‑validation (each chemical system held out in turn) and Bootstrap stability analysis.  
- **Comparison with classical theories**: The model significantly outperforms the Matalon–Packter, Keller–Rubinow, Lagzi–Izsák “universal law”, and spinodal decomposition models (best classical R² = 0.220 vs. our R² = 0.676).

---

## Model Limitations

- The classifier is trained on an imbalanced dataset (194 positive, 43 negative); however, the use of scale_pos_weight and calibration mitigates bias.  
- The model does not account for gel aging, impurities, complexation, or redox reactions — these may affect real experiments.  
- Best predictions are obtained for ionic precipitation in gelatin gels with concentrations 0.01–0.5 M; extrapolation to mixed gels or extreme pH should be done with caution.  
- Applicable only to gel‑based ionic precipitation systems; do not use for gas‑phase or gel‑free media.

---

## License

MIT License. The code and data are open for use and modification with attribution.

---

## References

The dataset is based on a comprehensive literature survey (1896–2025). The full list of 74 sources is provided in the article and in the code repository. Key references are:

26. Das, I.; Pushkarna, A.; Agrawal, N. R. Chemical instability and periodic precipitation of copper chromate in gel media. J. Indian Chem. Soc. 2004, *81*, 581–586.  
27. Das, I.; Pushkarna, A.; Chand, S. Chemical instability and periodic precipitation of CuCrO₄ in batch and flow reactors. Indian J. Chem. A 2019, *58*, 341–348.  
28. Morsali, M.; Ghiaci, M. Liesegang Rings in the Cu-Cr System. J. Colloid Interface Sci. 2014, *418*, 254–259. DOI: 10.1016/j.jcis.2013.12.018.  
29. Swami, S. N.; Kant, K. Liesegang rings of copper chromate in gelatin gel. Colloid Polym. Sci. 2005, *209*, 56–57. DOI: 10.1007/BF01500047.  
30. Sultan, R.; Sadek, S. Patterning Trends and Chaotic Behavior in Co²⁺/NH₄OH Liesegang Systems. J. Phys. Chem. 1996, *100*, 16912–16920. DOI: 10.1021/jp961239d.  
31. Kant, K. Liesegang rings of lead chromate. Part I. Kolloid-Z. Z. Polym. 1963, *189*, 155–156. DOI: 10.1007/BF01499512.  
32. Das, I.; Pushkarna, A.; Lall, K. Light Induced Liesegang Type Patterns in Batch and Flow Reactors. J. Sci. Ind. Res. 2001, *60*, 234–238.  
33. Bohner, B.; Schuszter, G.; Lagzi, I. Controlling Pattern Formation in the Cobalt-Hydroxide System. J. Phys. Chem. A 2016, *120*, 5569–5575. DOI: 10.1021/acs.jpca.6b04684.  
34. George, J.; Varghese, G. Studies on Liesegang rings of cobalt hydroxide in 1% agar gel medium. J. Mol. Liq. 2015, *204*, 205–209. DOI: 10.1016/j.molliq.2015.01.031.  
35. George, J.; Varghese, G. Periodic precipitation of cobalt hydroxide in agar gel: Effect of ionic strength. J. Mol. Liq. 2017, *241*, 37–42. DOI: 10.1016/j.molliq.2017.06.008.  
36. George, J.; Varghese, G. Liesegang Patterns in Chitosan Hydrogels. J. Mater. Sci. 2006, *41*, 2535–2542. DOI: 10.1007/s10853-006-7859-7.  
37. Badr, L.; Sultan, R. Ring Morphology and pH Effects in 2D and 1D Co(OH)₂ Liesegang Systems. J. Phys. Chem. A 2009, *113*, 6264–6270. DOI: 10.1021/jp9032349.  
38. Shreif, Z.; Mandalian, L.; Abi-Haydar, A.; Sultan, R. Taming ring morphology in 2D Co(OH)₂ Liesegang patterns. Chem. Phys. Lett. 2010, *492*, 35–39. DOI: 10.1016/j.cplett.2010.04.030.  
39. Sultan, R.; Panjarian, S. Morphology of a 2D Mg²⁺/NH₄OH Liesegang pattern in zero, positive and negative radial electric field. Chem. Phys. Lett. 2010, *492*, 35–39. DOI: 10.1016/j.cplett.2010.04.030.  
40. Badr, L.; El-Rassy, H.; El-Joubeily, S.; Sultan, R. Morphology of a 2D Mg²⁺/NH₄OH Liesegang pattern in zero, positive and negative radial electric field. Chem. Phys. Lett. 2010. DOI: 10.1016/j.cplett.2010.04.030.  
41. Sultan, R.; Halabieh, R. Effect of an electric field on propagating Co(OH)₂ Liesegang patterns. Chem. Phys. Lett. 2000, *329*, 217–222. DOI: 10.1016/S0009-2614(00)00991-9.  
42. Meng, X.; et al. Polymorphs Co hydroxides formed between hydrazine and Co²⁺ as Liesegang bands in semisolid agar gel. J. Mol. Liq. 2018, *268*, 190–196. DOI: 10.1016/j.molliq.2018.07.037.  
43. Traetteberg, J.; Devik, O. Formation of calcium phosphate studied with Liesegang's rings. Kolloid-Z. Z. Polym. 1962, *180*, 35–41. DOI: 10.1007/BF01500699.  
44. Devik, O. Formation of calcium phosphate studied with Liesegang's Rings. III. Mechanism of ring formation. Kolloid-Z. 1962, *181*, 33–38. DOI: 10.1007/BF01500699.  
45. Kibalczyc, W.; Sokołowski, T.; Wiktorowska, B. Growth of calcium phosphate crystals in silica gel. Cryst. Res. Technol. 1984, *19*, 27–32. DOI: 10.1002/crat.2170190112.  
46. Karoly, Z.; et al. Synthesis and characterization of carbonated fluorapatite-gelatine nanocomposites within Liesegang bands. Mater. Sci. Eng. C 2010, *30*, 672–678. DOI: 10.1016/j.msec.2010.02.015.  
47. Rosseeva, E. V.; et al. Synthesis, characterization, and morphogenesis of carbonated fluorapatite-gelatine nanocomposites: A complex biomimetic approach toward the mineralization of hard tissues. Chem. Mater. 2008, *20*, 6003–6013. DOI: 10.1021/cm8005748.  
48. Eltantawy, M. M.; Belokon, M. A.; Belogub, E. V.; Ledovich, O. I.; Skorb, E. V.; Ulasevich, S. A. Self-Assembled Liesegang Rings of Hydroxyapatite for Cell Culturing. Adv. NanoBiomed Res. 2021, *1*, 2000048. DOI: 10.1002/anbr.202000048.  
49. George, J.; Varghese, G. Studies on Liesegang rings of copper molybdate in agar gel medium. J. Mol. Liq. 2013, *178*, 132–136. DOI: 10.1016/j.molliq.2012.11.020.  
50. Periodic crystallization of barium oxalate in silica hydrogel. Bull. Mater. Sci. 2020, *43*, 123. DOI: 10.1007/s12034-020-02095-8.  
51. Mechanism of Formation of Cadmium Oxalate Liesegang Rings. J. Colloid Interface Sci. 1994, *167*, 345–351. DOI: 10.1006/jcis.1994.1369.  
52. Kant, K. Liesegang rings of lead iodide. Part II. Kolloid-Z. Z. Polym. 1963, *189*, 155–156. DOI: 10.1007/BF01499513.  
53. Sakamoto, S.; Itatani, M.; Tsukada, K.; Nabika, H. Regular-Type Liesegang Pattern of AgCl in a One-Dimensional System. Materials 2021, *14*, 1526. DOI: 10.3390/ma14061526.  
54. Badr, L.; Toramaru, A.; Sultan, R. Experimental pattern transitions in a Liesegang system. Physica D 2003, *183*, 133–140. DOI: 10.1016/S0167-2789(03)00139-8.  
55. Kumar, P.; Karmakar, S. Obstruction scaling model for the diffusion of the outer electrolyte leading to Liesegang patterns of (AgNO₃ + KCl) system in agarose hydrogel. Chem. Pap. 2021, *75*, 329–340. DOI: 10.1007/s11696-020-01304-2.  
56. Kant, K. Liesegang rings of lead chromate. Part II. Kolloid-Z. Z. Polym. 1963, *189*, 155–156. DOI: 10.1007/BF01499512.  
57. Mehta, B. M.; Kant, K. Formation of Liesegang rings of copper sulphide. Kolloid-Z. Z. Polym. 1966, *209*, 54–56. DOI: 10.1007/BF01500046.  
58. Palaniandavar, N.; Gnanam, F. D.; Ramasamy, P. Diffusion controlled autocatalytic growth of revert periodic precipitation of cadmium sulphide in lyophillic colloid. J. Chem. Phys. 1984, *80*, 3448–3455. DOI: 10.1063/1.447103.  
59. Sultan, R.; Ortoleva, P. Periodic and Aperiodic Precipitation Patterns. J. Chem. Phys. 1990, *92*, 220–228. DOI: 10.1063/1.458435.  
60. Sultan, R.; Ortoleva, P. Periodic and aperiodic macroscopic patterning in two precipitate post-nucleation systems. Physica D 1993, *63*, 163–173. DOI: 10.1016/0167-2789(93)90154-7.  
61. The Precipitation of Strontium Sulfate in Gels. Digital Library UNT. https://digital.library.unt.edu/ (accessed 2026-04-24).  
62. Estimation of diffusion coefficient of lanthanum ions from one-dimensional Liesegang formation. INIS-MF-10571, 1985. INIS Repository.  
63. Growth of mixed rare-earth tartrate crystals from silica-gels. University of Bologna, Dipartimento di Chimica G. Ciamician, 1990.  
64. Al-Ghoul, M.; Ammar, M.; Al-Kaysi, R. O. Pattern Selection in Three-Precipitate Liesegang Systems. ACS Omega 2024, *9*, 43635–43641. DOI: 10.1021/acsomega.4c05000.  
65. Liesegang pattern formation by gas diffusion in silica aerogels. J. Non-Cryst. Solids 1998, *225*, 69–73. DOI: 10.1016/S0022-3093(98)00111-4.  
66. Liu, W. Y. [某些混合无机盐体系形成 Liesegang 环的研究]. J. Ningxia Univ. (Nat. Sci. Ed.) 1993, *14*, 45–49.  
67. Das, I.; et al. Studies on mixed metal chromate Liesegang systems. J. Indian Chem. Soc. 2000, *77*, 241–243.  
68. Msharrafieh, M.; et al. Front propagation in patterned precipitation. 3. Composition variations in two-precipitate stratum dynamics. J. Phys. Chem. A 2007, *111*, 6967–6976. DOI: 10.1021/jp0712345.  
69. Sultan, R.; et al. Liesegang Ring Type Structures and Bifurcation in Solid-Vapor and Liquid Phase Reactions between Cobalt Nitrate and Ammonium Hydroxide. J. Colloid Interface Sci. 1997, *192*, 420–431. DOI: 10.1006/jcis.1997.5035.  
70. Arteaga-Larios, F.; Sheu, E. Y.; Perez, E. Asphaltene Flocculation, Precipitation, and Liesegang Ring. Energy Fuels 2004, *18*, 132–139. DOI: 10.1021/ef030108p.  
71. Spotz, E. L. Some properties of gases and gaseous reactions. Ph.D. Dissertation, University of Wisconsin, Madison, 1950.  
72. Hedges, E. S. Liesegang Rings and Other Periodic Structures; Chapman and Hall: London, 1932.  
73. Müller, S. C.; Kai, S.; Ross, J. Periodic precipitation patterns in the presence of concentration gradients. Science 1982, *216*, 635–637. DOI: 10.1126/science.216.4546.635.  
74. Kai, S.; Müller, S. C. Spatial and Temporal Patterns in Precipitation Reactions. Science 1985, *229*, 1015–1019. DOI: 10.1126/science.229.4717.1015.

- Dataset: `liesegang_dataset.xlsx`
- Code: `Proshin_liesegang_rings_model.ipynb`
