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
### 3.4.1 Hyperparameter grid search
To enhance the model’s generalization ability and robustness while maintaining interpretability of tree model, we selected six widely used tree-based models as sub-models: Random Forest (RF), Gradient Boosting Regression Tree (GBRT), CatBoost Regression (CBR), AdaBoost Regression (ABR), XGBoost Regression (XGBR), and LightGBM (LGBM). For each sub-model, the optimal hyperparameters were identified via grid search optimization guided by the coefficient of determination $$R^2$$ evaluation metric (`tree_voting_tunning.ipynb`). 

The $$R^2$$ and Root-Mean-Square Error are employed to reflect the prediction accuracy, which are defined as: 

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y_i})^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2}$$

Here, $$y_i$$ represents the value calculated by density functional theory (DFT), $$\hat{y_i}$$ denotes the result predicted by the machine learning model, and $$\bar{y}$$ is the average value of DFT data.

We have specified the hyperparameter spaces of six sub-models. In line with usage requirements, these can be self-defined.
```python
cross_Valid = KFold(n_splits=5, shuffle=True, random_state=76)

# Define the hyperparameter grid for each model.
parameter_XGBR = {
'n_estimators': [100,200,300,400,500],
'learning_rate': [0.01, 0.05, 0.1,0.15],
'max_depth': [2,3,5],
'subsample': [0.4,0.6,0.8,1]
}

parameter_RF = {
'n_estimators':[100,200,300,400,500],
'max_depth':[2,4,None],
'max_features':['auto','log2','sqrt'],
'min_samples_leaf':[1,2,3,4]
}

parameter_CBR = {
'iterations': [100,200,300,400,500],
'learning_rate': [0.01, 0.05, 0.1,0.15],
'depth': [3,4,5,6],
'l2_leaf_reg': [1,3,5]
}

parameter_LGBM = {
'n_estimators': [100,200,300,400,500],
'learning_rate': [0.01, 0.05, 0.1,0.15],
'max_depth': [3,4,5,6],
'subsample': [0.6,0.8,1]
}

parameter_ABR = {
'n_estimators': [100,200,300,400,500],
'learning_rate': [0.01, 0.05, 0.1,0.15],
'loss': ['linear', 'square'] 
}

parameter_GBRT = {
'n_estimators': [100,200,300,400,500],
'learning_rate': [0.01, 0.05, 0.1,0.15],
'max_depth': [3,4,5,6],
'max_features': ['auto', 'sqrt']
}

# Define model dictionary
estimators = {
'XGBR': XGB.XGBRegressor(random_state=0),
'RF': RandomForestRegressor(random_state=0),
'CBR': CatBoostRegressor(verbose=False, random_state=0),
'LGBM': LGBMRegressor(random_state=0, verbosity=-1),
'ABR': AdaBoostRegressor(random_state=0),
'GBRT': GradientBoostingRegressor(random_state=0)
}

# Map the model names to their corresponding hyperparameters
params_mapping = {
'XGBR': parameter_XGBR,
'RF': parameter_RF,
'CBR': parameter_CBR,
'LGBM': parameter_LGBM,
'ABR': parameter_ABR,
'GBRT': parameter_GBRT
}

grid_searches = {}
for name, estimator in estimators.items():
    params = params_mapping[name] 
    grid_searches[name] = GridSearchCV(estimator, params, scoring='r2', cv=cross_Valid, n_jobs=-1)

# Train the model and find the best parameters.
for name, grid in grid_searches.items():
    grid.fit(x_train, y_train) 
    print(f"{name} best parameters: {grid.best_params_}")
```

### 3.4.2 Homogeneous ensemble and calculation of weighted average feature importance
After finding the optimal hyperparameters for each sub-model, we first construct a model dictionary to initialize the model and inject the found optimal hyperparameters into the model. Then, we perform model integration through Voting Regression and gradually eliminate the worst-performing model (we have reserved an interface at the definition of the model dictionary and can conveniently eliminate models by commenting out specific sub-models). This process continues until the performance of the fusion model is higher than that of all sub-models. For the analysis of model feature importance, to address variations in feature importance magnitudes across different tree models, we normalized the feature importance values of each sub-model. The R2 value of each sub-model in a single training was used as a weight to compute a weighted average of the feature importance, which was assigned as the fusion model’s feature importance for that iteration (`tree_voting_ensemble.ipynb`).
```python
# Define a function to initialize the model dictionary.
def initialize_best_estimators(grid_searches):
    return {
        'XGBR': XGB.XGBRegressor(**grid_searches['XGBR'].best_params_, random_state=42),
        'RF': RandomForestRegressor(**grid_searches['RF'].best_params_, random_state=42),
        'CBR': CatBoostRegressor(**grid_searches['CBR'].best_params_, verbose=False, random_state=42),
        'LGBM': LGBMRegressor(**grid_searches['LGBM'].best_params_, random_state=42, verbosity=-1),
        'ABR': AdaBoostRegressor(**grid_searches['ABR'].best_params_, random_state=42),
        'GBRT': GradientBoostingRegressor(**grid_searches['GBRT'].best_params_, random_state=42)
    }

# Define functions for calculating model performance and feature importance.
def evaluate_models(seed, X, Y, grid_searches, submodel_r2_sums, submodel_rmse_sums, num_seeds):
    cross_validator = KFold(n_splits=10, shuffle=True, random_state=seed)
    best_estimators = initialize_best_estimators(grid_searches)
    # Create VotingRegressor and only keep the uncommented models.
    submodels = [(name, estimator) for name, estimator in best_estimators.items() if estimator is not None]
    voting_regressor = VotingRegressor(submodels)

    # Calculate the mean values of R2 and RMSE for each sub-model.
    submodel_r2_means = {}
    submodel_rmse_means = {}
    rmse_scorer = make_scorer(mean_squared_error, squared=False)
    
    # Used to store the feature importance of each model.
    submodel_feature_importances = {}
    feature_importances_weighted_sum = np.zeros(X.shape[1])
    total_r2 = 0

    for name, estimator in submodels:
        r2_scores = cross_val_score(estimator, X, Y, cv=cross_validator, scoring='r2', n_jobs=-1)
        rmse_scores = cross_val_score(estimator, X, Y, cv=cross_validator, scoring=rmse_scorer, n_jobs=-1)
        submodel_r2_means[name] = np.mean(r2_scores)
        submodel_rmse_means[name] = np.mean(rmse_scores)
        submodel_r2_sums[name] += submodel_r2_means[name]
        submodel_rmse_sums[name] += submodel_rmse_means[name]

        # Calculate feature importance
        estimator.fit(X, Y)
        importances = estimator.feature_importances_
        
        # normalization processing
        importances_normalized = importances / np.sum(importances)
        
        submodel_feature_importances[name] = dict(zip(X.columns, importances_normalized))
        feature_importances_weighted_sum += importances_normalized * submodel_r2_means[name]
        total_r2 += submodel_r2_means[name]

    # Calculate the mean R2 and weighted RMSE of the fusion model.
    voting_regressor_r2_mean = np.mean(cross_val_score(voting_regressor, X, Y, cv=cross_validator, scoring='r2', n_jobs=-1))
    voting_regressor_rmse_mean = np.mean(cross_val_score(voting_regressor, X, Y, cv=cross_validator, scoring=rmse_scorer, n_jobs=-1))

    # Calculate weighted feature importance
    weighted_feature_importances = feature_importances_weighted_sum / total_r2 if total_r2 != 0 else feature_importances_weighted_sum

    weighted_feature_importances_dict = dict(zip(X.columns, weighted_feature_importances))

    return {
        'submodel_r2_means': submodel_r2_means,
        'submodel_rmse_means': submodel_rmse_means,
        'voting_regressor_r2_mean': voting_regressor_r2_mean,
        'voting_regressor_rmse_mean': voting_regressor_rmse_mean,
        'submodel_feature_importances': submodel_feature_importances,
        'weighted_feature_importances': weighted_feature_importances_dict
}
```

### 3.4.3 Traversal of random seeds from 0 to 99.
To mitigate single-training bias, we adjusted the dataset partition and iterated over random seeds from 0 to 99. The feature importance values from 100 training iterations were arithmetically averaged to obtain the final feature importance, which are defined as:

$$\ FI_i = \frac{1}{100} \sum_{t=0}^{99} \left( \frac{\sum_{m=1}^M R^{2^{(t,m)}} f_{i_i}^{(t,m)}}{\sum_{m=1}^M R^{2^{(t,m)}}} \right) \$$

Among them, $$\ FI_i \$$ represents the feature importance extracted by the fusion model for feature i, $$R^{2^{(t,m)}}$$ represents the $$R^2$$ value of the m-th sub-model in the t-th iteration, and $$f_{i_i}^{(t,m)}$$ represents the importance of feature i assigned by the m-th sub-model for feature i in the t-th iteration.
```python
# Define and save the results of all seeds.
all_results = []
seeds = range(100)

# Initialize the accumulated sum of R2 and RMSE of the submodel.
submodel_r2_sums = {name: 0.0 for name in initialize_best_estimators(grid_searches).keys()}
submodel_rmse_sums = {name: 0.0 for name in initialize_best_estimators(grid_searches).keys()}

# Initialize the weighted sum of feature importance accumulations of the fusion model.
weighted_feature_importances_sums = np.zeros(len(X.columns))

# Traverse the seeds and save the results.
for seed in seeds:
    result = evaluate_models(seed, X, Y, grid_searches, submodel_r2_sums, submodel_rmse_sums, len(seeds))
    all_results.append(result)
    weighted_feature_importances_sums += np.array(list(result['weighted_feature_importances'].values()))
    print(f"")
    print(f"Seed {seed} - Voting Regressor R2 Mean: {result['voting_regressor_r2_mean']}, RMSE Mean: {result['voting_regressor_rmse_mean']}")
    print("Submodel R2 and RMSE Means:")
    for name in result['submodel_r2_means'].keys():
        print(f"{name}: R2 Mean = {result['submodel_r2_means'][name]}, RMSE Mean = {result['submodel_rmse_means'][name]}")

# Calculate the average value of the weighted feature importance of the fusion model for all seeds.
avg_weighted_feature_importances = weighted_feature_importances_sums / len(seeds)

# Bind the feature name and the corresponding weighted feature importance and sort in descending order of importance.
sorted_feature_importances = sorted(zip(X.columns, avg_weighted_feature_importances), key=lambda x: x[1], reverse=True)
```

### 3.4.4 Results output
Finally, output the R2 and RMSE of each sub-model and the fusion model. Sort the feature importance of the weighted average output of the fusion model under the last 100 random seeds.
```python
# Output the R2, RMSE of the fusion model and each sub-model and the sorted feature importance.
avg_voting_regressor_r2_mean = np.mean([result['voting_regressor_r2_mean'] for result in all_results])
avg_voting_regressor_rmse_mean = np.mean([result['voting_regressor_rmse_mean'] for result in all_results])

print(f"")
print(f"Average Voting Regressor R2 Mean: {avg_voting_regressor_r2_mean}")
print(f"Average Voting Regressor Weighted RMSE Mean: {avg_voting_regressor_rmse_mean}")
print("Average Submodel R2 and RMSE Means:")
for name in submodel_r2_sums.keys():
    print(f"{name}: R2 Mean = {submodel_r2_sums[name] / len(seeds)}, RMSE Mean = {submodel_rmse_sums[name] / len(seeds)}")
    
# Draw a bar chart of feature importance.
features, importances = zip(*sorted_feature_importances)
plt.figure(figsize=(10, 8))
sns.barplot(x=list(importances), y=list(features))
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances Sorted by Importance')
plt.show()

print("Sorted Feature Importances:")
for feature, importance in sorted_feature_importances:
print(f"{feature}: {importance}")
```

## 3.5 Homogeneous integration of linear models for the construction of descriptors
Under the same target value, we identified the top-ranked features across different sites as strong correlation factors and used them to construct functional group indexes. Specifically, to study the influence of functional groups on the target values, we constructed two specialized molecular databases. The first database featured site-5 as the grafting site for target functional groups and site-4 and -6 as the grafting site for -H, followed by calculating the SNAr energy barrier of molecules in this database. The second database designated site-4 and -6 as grafting sites for target functional groups and site-5 as grafting site for -H. The HOMO-LUMO energy gap of PyrMSLi-Li2S4 derived from the molecules in this database was then calculated. The -H group was chosen as the “inert group” due to its minimal electron cloud impact on the molecular skeleton. Note that to normalize the dimensional differences among features, the features of -H were used as the benchmark. A linear fusion model (containning Linear Regression (LR), Ridge Regression (RR), Least Angle Regression (LAR), Elastic Net Regression (ENR), Partial Least Squares Regression (PLSR), and Support Vector Regression (SVR). Except for setting the kernel to linear in SVR, we retain the default values for all other hyperparameters (But we still reserve an interface to implement custom hyperparameters)) was employed to search for 3 generalization coefficients, which were then used to construct the Ibarrier and Igap.
A CPyr molecule has three grafting sites. To quantify the influence of functional groups on molecular properties, we first determined the contribution of functional groups on each site to the target value based on the feature importance ranking. Specifically, the feature importance ranking revealed that site-5 exhibited a strong correlation with the SNAr energy barrier (56% contribution, calculated by summing the feature importance of site-5), while site-4 and -6 were highly correlated with the HOMO-LUMO energy gap (86% contribution, calculated by summing the feature importance of site-4 and -6). Then, we used the importance percentage of each site under different target values as weights and computed a weighted average of the functional group indexes across the three sites. The resulting descriptors, denoted as Dbarrier and Dgap, represent the molecular properties. Detailed values of Dbarrier and Dgap for 35 training samples and 5 experimental samples are provided in Supplementary Table 5. This approach allows the approximation of molecular properties by calculating limited functional group characteristics. The homogeneous integration process of linear models is similar to the above process. For details, see `linear_voting_tunning.ipynb`，`linear_voting_ensemble.ipynb`.


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
