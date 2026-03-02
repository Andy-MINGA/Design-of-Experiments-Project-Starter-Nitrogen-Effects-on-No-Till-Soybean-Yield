# Evaluation of Starter Nitrogen Effects on No-Till Soybean Yield

## 📌 Project Overview
This research evaluates the agronomic impact of four starter nitrogen fertilizers on soybean yield in no-till systems. Using a **Randomized Complete Block Design (RCBD)** across five distinct site-years, the project assesses whether supplemental nitrogen at planting provides a statistically significant yield advantage to compensate for nitrogen immobilization.

## 🧪 Experimental Design & Data
The analysis utilizes both a real-world dataset (Allen et al., 2023) and a custom-simulated dataset to validate model performance.

* **Treatments**: Control (no fertilizer), Poultry Litter (PL), Feather Meal (FM), and Sodium Nitrate (SN).
* **Locations**: Data collected across five site-years in New York (2020–2022) and Wisconsin (2021–2022).
* **Replication**: 4 to 5 blocks per site-year to control for local soil and environmental variability.

## 📊 Statistical Methodology
The project employs a **Linear Mixed-Effects Model** to account for both fixed treatment effects and random environmental variation.

* **Fixed Effects**: Fertilizer treatment.
* **Random Effects**: Site-year and block nested within site-year.
* **Modeling Equation**: $Y_{ijk} = \mu + \tau_{i} + s_{k} + b_{j(k)} + \epsilon_{ijk}$.
* **Post-hoc Testing**: Tukey-adjusted pairwise comparisons were used to evaluate specific differences between fertilizer types.
* **Data Simulation**: A simulated dataset was generated to mimic the hierarchical RCBD structure and test the robustness of the mixed-effects model.

## 📈 Key Findings
The analysis demonstrates that environmental variability is the primary driver of soybean yield, rather than starter nitrogen application.

| Fertilizer Treatment | Estimate (kg/ha) | p-value | Significance |
| :--- | :--- | :--- | :--- |
| **Control (Baseline)** | 2633.51 | 0.00026 | Reference |
| **Feather Meal (FM)** | +95.86 | 0.49 | Not Significant |
| **Poultry Litter (PL)** | +147.64 | 0.29 | Not Significant |
| **Sodium Nitrate (SN)** | +104.50 | 0.45 | Not Significant |

* **Environmental Dominance**: Variance decomposition showed that differences among site-years accounted for the vast majority of yield variability, while block effects within site-years were negligible.
* **Assumption Validation**: Residual analysis (Q-Q plots) and the Shapiro-Wilk test (p = 0.5336) confirmed that the model assumptions of normality and constant variance were satisfied.

## 💡 Conclusion
The study concludes that starter nitrogen fertilizers did not meaningfully increase soybean yield under the tested conditions. From a sustainability and economic perspective, these findings suggest that farmers may avoid unnecessary nitrogen inputs in these systems, reducing both operational costs and the risk of environmental leaching.

## 🛠️ Tools & Technologies
* **Language**: R
* **Modeling**: Linear Mixed-Effects Models (lme4/emmeans)
* **Techniques**: RCBD, Variance Decomposition, Data Simulation, Tukey HSD

---
**Author**: Andy Minga  
**Project Date**: Fall 2025
