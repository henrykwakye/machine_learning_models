# Machine Learning Algorithms Overview

## 1. Supervised Learning
Trains on labeled data (the correct target is known during training).

### 1.1 Regression (predict continuous values)
- Linear Regression (simple, multiple) — fits linear relationship between features and target
- Polynomial Regression — extends linear model with polynomial feature terms for nonlinearity
- Ridge Regression (L2) — penalizes large coefficients to reduce variance/multicollinearity
- Lasso Regression (L1) — drives some coefficients to zero for embedded feature selection
- Elastic Net — blends L1 and L2 to balance sparsity and stability
- Support Vector Regression (SVR) — fits function within epsilon-tube maximizing margin
- Decision Tree Regression — recursive splits creating piecewise constant predictions
- Random Forest Regression — bagged trees averaging to lower variance
- Gradient Boosting Regression (XGBoost, LightGBM, CatBoost) — sequential trees correcting residuals
- K-Nearest Neighbors Regression — averages targets of closest points in feature space
- Bayesian Regression — probabilistic linear model yielding parameter/posterior uncertainty

### 1.2 Classification (predict discrete classes)
- Logistic Regression — linear decision boundary modeling class log-odds
- K-Nearest Neighbors (KNN) Classification — majority vote of nearest labeled samples
- Support Vector Machines (SVM) — maximum margin classifier with kernel trick for nonlinearity
- Naive Bayes (Gaussian, Multinomial, Bernoulli) — probabilistic model assuming conditional independence
- Decision Tree Classification — splits to maximize class purity (e.g., Gini, entropy)
- Random Forest Classification — ensemble of randomized trees reducing overfitting
- Gradient Boosting Classification (XGBoost, LightGBM, CatBoost) — additive trees minimizing differentiable loss
- Neural Networks (MLP, CNN, RNN for classification) — layered nonlinear feature learners
- Linear Discriminant Analysis (LDA) — projects to maximize separation assuming shared covariance
- Quadratic Discriminant Analysis (QDA) — class-specific covariance enabling quadratic boundaries

### 1.3 (Optional future expansion)
<!-- - Multi-output / Multi-label methods -->
<!-- - Imbalanced classification techniques (SMOTE, class-weighting) -->

## 2. Unsupervised Learning
Finds structure in unlabeled data.

### 2.1 Clustering
- K-Means — partitions data minimizing within-cluster squared distance
- K-Medoids / PAM — like K-Means but centers are actual points; robust to outliers
- Hierarchical Clustering (Agglomerative, Divisive) — builds dendrogram of nested clusters
- DBSCAN — density-based clustering discovering arbitrary shapes and noise points
- OPTICS — orders points to extract clusters at varying density thresholds
- Mean Shift — shifts points toward kernel density modes as cluster centers
- Gaussian Mixture Models (GMM) — probabilistic soft clusters via mixture of Gaussians (EM)
- Spectral Clustering — uses graph Laplacian eigenvectors for clustering
- BIRCH — incremental CF-tree for large-scale hierarchical clustering

### 2.2 Dimensionality Reduction
- Principal Component Analysis (PCA) — orthogonal components capturing maximal variance
- Kernel PCA — nonlinear feature mapping prior to PCA via kernels
- t-SNE — preserves local neighborhood structure for visualization (high → 2/3D)
- UMAP — manifold learning balancing local/global structure
- Factor Analysis — models observed variables via latent factors plus noise
- Independent Component Analysis (ICA) — separates statistically independent source signals
- Singular Value Decomposition (SVD) — matrix factorization underpinning many DR methods

### 2.3 Association Rule Learning
- Apriori — breadth-first frequent itemset mining via candidate pruning
- Eclat — depth-first mining using vertical (tidset) intersections
- FP-Growth — builds compressed FP-tree to mine frequent itemsets without candidate explosion

### 2.4 (Optional future expansion)
<!-- - Anomaly detection (Isolation Forest, One-Class SVM, LOF) -->
<!-- - Topic modeling (LDA - Latent Dirichlet Allocation, NMF) -->

## 3. Reinforcement Learning
Learns via trial and error with rewards.

### 3.1 Value-Based Methods
- Q-Learning — off-policy temporal-difference optimal action-value iteration
- Deep Q-Networks (DQN) — neural approximation of Q with replay & target network
- Double DQN — separates action selection/evaluation to reduce overestimation
- SARSA — on-policy temporal-difference updating current behavior policy value

### 3.2 Policy-Based Methods
- REINFORCE — Monte Carlo policy gradient using sampled returns
- Policy Gradient Methods — directly optimize expected return via gradient ascent

### 3.3 Actor-Critic Methods
- A3C — parallel asynchronous workers stabilize/accelerate learning
- A2C — synchronous variant of A3C for deterministic updates
- PPO — clipped surrogate objective enabling stable large-batch updates
- DDPG — deterministic actor-critic for continuous actions with target networks
- TD3 — twin critics and delayed updates mitigating critic overestimation
- SAC — entropy-regularized max-entropy objective for robust exploration

### 3.4 Model-Based Methods
- Dyna-Q — integrates learned model for planning plus real experience updates
- World Models — learns latent dynamics for imagination-based planning/control

## 4. (Optional categories to add later)
<!-- - Semi-Supervised Learning (Label Propagation, Label Spreading) -->
<!-- - Self-Supervised Learning (contrastive, masked modeling) -->
<!-- - Transfer Learning / Fine-Tuning -->
<!-- - Time Series (ARIMA, Prophet, LSTM, GRU, Temporal Fusion Transformer) -->
<!-- - Recommender Systems (Matrix Factorization, Factorization Machines) -->
<!-- - Ensemble Strategies (Bagging, Boosting, Stacking, Blending) -->
<!-- - Hyperparameter Optimization (Grid, Random, Bayesian Optimization, Hyperband) -->
<!-- - Causal Inference (DoWhy, Propensity Scores) -->

---

## 5. Ensemble Methods
General strategies combining multiple base learners.
- Bagging — parallel training on bootstrap samples to reduce variance (e.g., Random Forest)
- Boosting — sequential learners focusing on residual errors (e.g., Gradient Boosting, AdaBoost)
- Stacking — meta-model learns to blend heterogeneous base model outputs
- Blending — stacking variant using hold-out validation split
- Voting (hard/soft) — aggregates class labels or probabilities from multiple models

## 6. Semi-Supervised Learning
Uses limited labeled plus abundant unlabeled data.
- Label Propagation — diffuses labels across similarity graph until convergence
- Label Spreading — variant adding normalization/smoothing for robustness
- Pseudo-Labeling — assigns high-confidence model predictions as temporary labels
- Self-Training — iteratively retrains adding confidently predicted unlabeled samples

## 7. Self-Supervised Learning
Creates surrogate pretext tasks to learn representations.
- Contrastive Learning — maximize agreement of positives vs negatives (e.g., SimCLR)
- Masked Modeling — predict masked tokens/patches to capture context (e.g., BERT style)
- Autoencoders — encode/decode to reconstruct input minimizing reconstruction error
- SimCLR — augmentation-based contrastive framework without explicit negatives memory
- BYOL — bootstrap targets enabling representation learning without negative pairs

## 8. Time Series Modeling
Handles temporal dependence and forecasting.
- ARIMA — autoregressive integrated moving average for stationary series
- SARIMA — seasonal extension capturing periodic patterns
- Exponential Smoothing (Holt-Winters) — weighted averages with level/trend/seasonality
- Prophet — additive components (trend/season/holiday) with robust fitting
- LSTM — recurrent units capturing long-range temporal dependencies
- GRU — gated recurrent alternative with fewer parameters
- Temporal Convolutional Networks (TCN) — dilated causal convolutions for long context
- Temporal Fusion Transformer (TFT) — attention-based multi-horizon forecasting with interpretability

## 9. Anomaly Detection
Identifies rare or deviating patterns.
- Isolation Forest — isolates points via random splits; anomalies require fewer splits
- One-Class SVM — learns boundary enclosing normal data in feature space
- Local Outlier Factor (LOF) — compares local density to neighbors
- Autoencoder (Reconstruction Error) — high reconstruction error flags anomalies
- Gaussian Mixture (Low Likelihood) — anomalies have low probability under learned mixture

## 10. Recommender Systems
Predict user-item relevance.
- User-Based Collaborative Filtering — neighbors via user similarity
- Item-Based Collaborative Filtering — recommends similar items to those interacted with
- Matrix Factorization (SVD) — latent user/item factors from interaction matrix
- Factorization Machines — generalizes MF with sparse feature interactions
- Neural Collaborative Filtering — deep architectures modeling complex interactions
- Session-Based (RNN/Transformer) — sequence-aware recommendations using recent events

---

## 11. Task-to-Algorithm Quick Reference

| Task | Common Algorithms (Representative) |
|------|------------------------------------|
| Regression | Linear/Polynomial, Ridge/Lasso/Elastic Net, SVR, Random Forest, Gradient Boosting, KNN, Bayesian, Neural Nets |
| Classification | Logistic, SVM, Random Forest, Gradient Boosting, KNN, Naive Bayes, Neural Nets, LDA/QDA |
| Clustering | K-Means, Hierarchical, DBSCAN, GMM, Spectral, Mean Shift, OPTICS |
| Dimensionality Reduction / Visualization | PCA, Kernel PCA, t-SNE, UMAP, ICA, SVD |
| Association Rules | Apriori, FP-Growth, Eclat |
| Reinforcement Control | Q-Learning, DQN, PPO, SAC, TD3, A3C/A2C |
| Time Series Forecasting | ARIMA/SARIMA, Prophet, Exponential Smoothing, LSTM, TCN, TFT |
| Anomaly Detection | Isolation Forest, One-Class SVM, LOF, Autoencoder, GMM |
| Recommendation | Collaborative Filtering, Matrix Factorization, Factorization Machines, Neural CF, Session Models |
| Representation Learning | Autoencoders, Contrastive (SimCLR), Masked Modeling, BYOL |
| Ensemble Strategy | Bagging, Boosting, Stacking, Voting |

---