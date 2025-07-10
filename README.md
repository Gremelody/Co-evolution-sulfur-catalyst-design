# Co-evolution electronic reconfiguration programming for dual-behavior sulfur electrocatalyst
Conventional catalysts manipulate electronic/geometric characteristics to optimize specific reactions thermodynamically, yet their static configurations lack intrinsic adaptability for dynamically modulating catalytic activity across complex reaction pathways and varied intermediates, resulting in inefficient multi-step catalytic processes. Meanwhile, catalyst reconstruction during (electro)chemical reactions undermines predictive catalyst design with theory-practice discrepancies. Herein, we demonstrate rational harnessing of dynamic catalyst reconstruction in lithium-sulfur (Li-S) batteries, achieving the transformative co-evolution catalysis where the catalyst evolves in tandem with the transformation requirements of diverse Li-S redox products (LSRPs), thus sustaining high-efficiency catalysis throughout the full-pathway.
# 1. Setting
## 1.1 Enviroments
* Python (Jupyter notebook) 
## 1.2 Python requirements
* python=3.12
* numpy=1.26.4
* matplotlib=3.8.4
* scipy=1.14.1
* scikit-learn=1.4
* pandas=2.2.2

# 2. Datasets
* Raw and processed datasets have been deposited in `Original dataset`, `Band alignment-feature engineeringed.xlsx`, `Band alignment-prediction.xlsx`, `Shift range-feature engineeringed.xlsx`, and `Shift range-prediction.xlsx` which can be accessed at this page.

# 3. Experiment
## 3.1 Overview
In this research, to achieve precise electronic structure programming for accurately matching lithium polysulfides and catalysts in different states, we integrated a density functional theory-machine learning framework to screen optimal dopants under dual-objective conditions.

This page contains three databases:
1. `Original dataset.xlsx`: Two target values (target 1: band alignment; target 2: shift range) and 25-dimentional features (containing electronic, structural and chemical features) for each element.
2. `Band alignment-feature engineeringed.xlsx`: Under target 1，Features and target values after feature engineering
3. `Shift range-feature engineeringed.xlsx`: Under target 1，Features and target values after feature engineering
4. `Band alignment-prediction.xlsx`: Prediction Set for the target 1
5. `Shift range-prediction.xlsx`: Prediction Set for the target 2

Two Jupyter Notebooks:
1. `Feature engineering.ipynb`: Two-stage (filter-embedded integred method) feature engineering.
2. `Tree_stacking.ipynb`: Hyperparameter tuning process of seven base learners based on tree models.

## 3.2 Dataset establishment
To achieve precise computational-experimental matching, we employed the vaspsol add-on package to describe the implicit solvent environment for 1M LiTFSI + 1 wt% LiNO3. By utilizing the HSE06 hybrid functional, we mitigated the errors of the generalized gradient approximation in the electron exchange potential. The catalyst and lithium polysulfide molecules were encapsulated in a periodic boundary condition box, with a 20 Å vacuum layer separating the upper and lower mirrors to ensure alignment of their vacuum energy levels and eliminate the influence of non-physical long-range electrostatic interactions.

## 3.3 Feature engineering
Driven by the imperative to unveil intrinsic physical mechanisms, we prioritized features with explicit physical interpretability through multi-source feature collection. This involved leveraging the Materials Agnostic Platform for Informatics and Exploration (Magpie), along with DFT-calculated and Multiwfn12 wavefunction analyses, to extract chemical, electronic, and geometric features. To alleviate multicollinearity-induced attenuation risks in downstream tree-based algorithms, we employed a two-stage integrated workflow for feature selection: filtering and an integrated embedded method. This process was applied independently for each target variable (e.g., band alignment or shift range). 
### 3.3.1 Filtering Stage (Filter Method)
We conducted pairwise comparisons of Pearson correlation coefficients between all features. If the absolute Pearson correlation coefficient between two features exceeded 0.8, the feature with lower absolute correlation to the target variable was removed. This process reduced the initial 25-dimensional feature set to 8 dimensions for Target 1 (band alignment) and 9 dimensions for Target 2 (shift range). 
### 3.3.2 Integrated Embedded Method Stage
Subsequently, we utilized an embedded method for feature selection, leveraging SHAP (SHapley Additive exPlanations) analysis. A Random Forest Regressor (Supplementary Table 5) was used as the base estimator within a Leave-One-Out Cross-Validation (LOOCV) framework. In each iteration, the feature with the lowest SHAP importance was removed. The model's performance (measured by Mean Absolute Error, MAE) was re-evaluated after each removal. This iterative process continued until the model's performance no longer improved, yielding the optimal feature subset.

## 3.4 Model fusion and robustness evaluation
### 3.4.1 Model fusion
To boost the model's generalization and robustness while retaining interpretability, seven prevalent tree-based models were chosen as base learners (sub-models): Random Forest (RF), Gradient Boosting Regression Tree (GBRT), CatBoost Regression (CBR), Extra Trees Regressor (ETR), XGBoost Regression (XGBR), HistGradientBoostingRegressor (HGBR), and LightGBM (LGBM). For each base learner, Bayesian optimization, guided by the Root Mean Squared Error (RMSE) metric, was used to find optimal hyperparameters. 
To effectively utilize the limited dataset and ensure robust meta-learner training, Out-Of-Fold (OOF) predictions from the base learners were generated using LOOCV on the full training dataset. These OOF predictions served as the input for the meta-learner's training. The meta-learner's hyperparameters were then tuned using Bayesian optimization, which internally employed Leave-One-Out Cross-Validation on these OOF predictions. Crucially, the meta-learner's coefficients were fixed based on the optimal parameters found during this one-time tuning, and this fixed meta-learner was used for all subsequent outer cross-validation evaluations. To mitigate the contingency of results from a single training under a single dataset partition, we employed Leave-P-Out cross-validation (LPOCV) with P ranging from 1 to 3 for robustness evaluation across different dataset splits. This approach provides a comprehensive assessment of the model's stability and generalization capability. All computations were implemented 
### 3.4.1 robustness validation
To efficiently utilize the limited dataset, this optimization was performed using Leave-One-Out Cross-Validation (LOOCV) on the entire dataset. After the base learners were optimized, an Elastic Net Stacking method was applied for model fusion. The meta-learner (Elastic Net linear model) was also subject to hyperparameter tuning. Due to its comprehensive regularization (combining L1 and L2 penalties), which enables both implicit feature selection and robust handling of multicollinearity among base model predictions, Elastic Net was chosen as the meta-learner. Furthermore, a linear model was preferred over a tree-based model as the meta-learner for its simplicity and to mitigate overfitting to the limited base model predictions. 


## 3.6 The detailed description of electronic and geometric features used in this work
| Feature Type        | Feature                          | Abbreviations      | Description                                                                 |
|---------------------|----------------------------------|--------------------|-----------------------------------------------------------------------------|
| ​**​Electronic Features​**​ | Average electronegativity       | $\overline{EN}$    | Mean value of Pauling electronegativity of all atoms                        |
|                     | Chemical potential              | $\mu$              | the mean of the energies of HOMO and LUMO                                         |
|                     | Hardness                         | $H$                | The ability to resist changes in electron density                          |
|                     | Electrophilicity index           | $I_E$              | An indicator of the ability to attract electrons                           |
|                     | Nucleophilicity index            | $I_N$              | An indicator of the ability to supply electrons                            |
|                     | Average bond order               | $\overline{BO}$    | Mean value of all bond orders                                               |
|                     | Minimal value                    | $\varphi_{min}$    | Minimum value of van der Waals surface electrostatic potential             |
|                     | Maximal value                    | $\varphi_{max}$    | Maximum value of van der Waals surface electrostatic potential             |
|                     | Molecular polarity index         | $P_M$              | Degree of charge separation within functional groups                        |
| ​**​Geometric Features​**​ | Volume                          | $V$                | Volume of van der Waals surface                                             |
|                     | M/V                             | M/V                | Density of Functional Groups (Total Atomic Weight/Volume)                   |
|                     | Sphericity                      | $S$                | Degree of the geometric configuration's approximation to a spherical shape |
|                     | Overall surface area            | $A_S$              | The surface area of a van der Waals surface                                 |
|                     | Distance between Pyrimidine and maximum | $d_{Pyr - \varphi_{max}}$ | Distance between maximum point of electrostatic potential and center of pyrimidine |
|                     | Distance between Pyrimidine and minimum | $d_{Pyr - \varphi_{min}}$ | Distance between minimum point of electrostatic potential and center of pyrimidine |


# 4. Access
Data and code are under [MIT licence](https://github.com/terencetaothucb/intelligent-molecular-skeleton-design/blob/main/LICENSE). Correspondence to Prof. [Guangmin Zhou](mailto:guangminzhou@sz.tsinghua.edu.cn) and Prof. [Xuan Zhang](mailto:xuanzhang@sz.tsinghua.edu.cn) when you use, or have any inquiries.

# 5. Acknowledgements
[Yifei Zhu](zhuyifeiedu@126.com) and [Terence (Shengyu) Tao](mailto:terencetaotbsi@gmail.com) at Tsinghua University conceived and formulated the algorithms, deposited model/experimental code, and authored this guideline document drawing on supplementary materials.
