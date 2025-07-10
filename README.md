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

This page contains three datasets:
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
### 3.4.2 robustness validation
To efficiently utilize the limited dataset, this optimization was performed using Leave-One-Out Cross-Validation (LOOCV) on the entire dataset. After the base learners were optimized, an Elastic Net Stacking method was applied for model fusion. The meta-learner (Elastic Net linear model) was also subject to hyperparameter tuning. Due to its comprehensive regularization (combining L1 and L2 penalties), which enables both implicit feature selection and robust handling of multicollinearity among base model predictions, Elastic Net was chosen as the meta-learner. Furthermore, a linear model was preferred over a tree-based model as the meta-learner for its simplicity and to mitigate overfitting to the limited base model predictions. 

## 3.5 Feature importance analysis
To avoid biased evaluations from tree models' intrinsic feature importances, we utilized the more authoritative SHAP (SHapley Additive exPlanations) analysis for model feature importance assessment. Furthermore, to circumvent the high computational cost of shap.KernelExplainer for large datasets, we employed shap.TreeExplainer, which is optimized for tree-based models, to analyze each base model individually. The feature importance for the overall fusion model was derived by aggregating the SHAP values from individual base models. Specifically, for each held-out fold within the LPOCV, we calculated the local SHAP values for each base model's predictions. These local SHAP values were then weighted by e^(-RMSE), where RMSE represents that base model's performance on the corresponding held-out fold. These weighted local SHAP values were subsequently averaged across all base models to obtain the fusion model's local feature importance for that fold. The absolute values of these local importances were then averaged to represent the global feature importance for that specific LPOCV partition. Finally, the global feature importances obtained from all different dataset partitions (from the LPOCV runs) were arithmetically averaged to yield the final robust feature importances.

## 3.6 The detailed features used in this work
| Feature type | Feature | Abbreviations | Description |
|---|---|---|---|
| **Electronic features** | Vertical IP | *VIP* | The minimum energy required to completely remove an electron from an atom |
| | Vertical EA | *VEA* | The energy released when an atom gains an electron and forms a negative ion |
| | Chemical potential | *CP* | Approximately represented as the average of the energies of HOMO and LUMO |
| | Hardness | *H* | The ability to resist changes in electron density |
| | Electrophilicity index | *Ei* | An indicator of the ability to attract electrons |
| | Nucleophilicity index | *Ni* | An indicator of the ability to supply electrons |
| | Maximal value | *Maxv* | Maximum value of van der Waals surface electrostatic potential |
| | Minimal value | *Minv* | Minimum value of van der Waals surface electrostatic potential |
| | Internal charge separation | *Pi* | Degree of charge separation within an atom |
| **Geometric features** | Dopant-sulfur distance | *DSD* | Distance between dopant and sulfur |
| | Dopant-metal distance | *DMD* | Distance between dopant and metal |
| | Volume | *V* | Volume of van der Waals surface |
| | Density | *D* | Atomic Weight/Volume |
| | Overall surface area | *SA* | The surface area of a van der Waals surface |
| **Chemical features** | Main group number | *MGN* | The period number of an element in the periodic table. |
| | Period number | *PN* | The main group ordinal number of an element in the periodic table of elements |
| | Covalent Radius | *CR* | The covalent radii of elements |
| | Electronegativity | *EN* | The Pauling electronegativity of elements. |
| | pValence | *Pv* | The number of p - electrons possessed by the element |

# 4. Usage
## 4.1 Feature engineering.ipynb
This notebook automates the process of feature selection and dimensionality reduction.

Input File:

Requires an Excel file named Original features.xlsx.

Format: The first column should be a sample ID, feature columns in the middle, and the target variable in the second-to-last column.

Key Adjustable Parameters (in Config class):

PEARSON_CORR_THRESHOLD: Threshold for removing highly correlated features (default: 0.8).

PERFORMANCE_METRIC: Metric ('mae' or 'r2') for the iterative refinement stage.

Primary Output:

An Excel file named final_engineered_dataset_...xlsx. This file contains the selected features and is the direct input for the next script.

Other outputs include correlation matrices and a summary of the feature selection process.

2. Tree_stacking.ipynb
This integrated notebook handles hyperparameter tuning, model validation, and prediction using a tree-based stacking ensemble. It is composed of three main parts that should be run in order.

Part 1: Base-Learner Hyperparameter Optimization
This section uses Bayesian Optimization to find the best hyperparameters for several base models.

Input File:

The final_engineered_dataset_...xlsx file generated by the feature engineering script. It should be renamed to Band alignment-feature engineeringed.xlsx or the path in the script should be updated.

Key Adjustable Parameters:

N_ITER_BAYESIAN: Number of iterations for the Bayesian search (default: 50).

ENABLED_SUBMODELS_FOR_TUNING_LIST: A list to select which base models to tune (e.g., XGBR, RF, GBRT).

Output:

This part does not save files. Instead, it stores the optimized model configurations in the grid_searches variable in memory for the next steps.

Part 2: Stacking Model Cross-Validation & Evaluation
This section evaluates the performance of the complete stacking model using Leave-P-Out Cross-Validation (LPO-CV) and provides detailed model analysis.

Input:

Uses the in-memory optimized models from Part 1.

Key Adjustable Parameters:

OUTER_CV_P_VALUE: The 'P' in LPO-CV, determining how many samples are left out for testing in each fold (default: 3). For a dataset with N=7, this results in C(7,3) = 35 validation runs.

Output:

SHAP Feature Importance Plots: A bar plot and a swarm plot are displayed directly in the notebook, showing feature importance and impact.

shap_swarm_data_...xlsx: Detailed data from the SHAP analysis.

cv_predictions_pivoted_...xlsx: Detailed prediction results from each CV fold.

cv_model_scores_per_run_pivoted_...xlsx: MAE and RMSE scores for each model in every CV run.

Part 3: Prediction on Unknown Data
This final section uses the fully trained and validated stacking model to make predictions on a new, unseen dataset.

Input File:

Requires an Excel file named Band alignment-prediction.xlsx.

Format: Must contain the same feature columns (name and order) as the training data but should not contain a target variable column. The first column should contain sample IDs.

Key Adjustable Parameters:

UNKNOWN_DATA_FILE: Path to the file with data to be predicted.

OUTPUT_PREDICTIONS_FILE: The name for the output file containing the results.

Output:

An Excel file (e.g., predictions_...xlsx) containing the original features from the unknown dataset, appended with prediction columns for each base model and the final stacking model.

# 4. Access
Data and code are under [MIT licence](https://github.com/Gremelody/Co-evolution-sulfur-catalyst-design/blob/main/LICENSE). Correspondence to Prof. [Guangmin Zhou](mailto:guangminzhou@sz.tsinghua.edu.cn) when you use, or have any inquiries.

# 5. Acknowledgements
[Yifei Zhu](zhuyifeiedu@126.com) at Tsinghua University conceived and formulated the algorithms, deposited model/experimental code, and authored this guideline document drawing on supplementary materials.
