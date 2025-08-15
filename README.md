# ML Project Structure

This repository is organized by learning paradigm and task, mirroring `all_ml_models.md`.

## Top-Level Categories
- supervised
- unsupervised
- reinforcement_learning
- ensemble_methods
- semi_supervised
- self_supervised
- time_series
- anomaly_detection
- recommender_systems

## Conventions
- snake_case directory names.
- Each algorithm folder can contain:
  - `data/` (raw or processed samples)
  - `notebooks/` (exploration & EDA)
  - `models/` (serialized models)
  - `src/` (reusable code)
  - `README.md` (brief description, usage)
- Shared utilities can go in a future `common/` folder.

## Current Migrated Examples
- Linear regression assets moved to `supervised/regression/linear_regression/`.
- KNN notebook & dataset moved to `supervised/classification/knn/`.
- SVM notebook & dataset moved to `supervised/classification/svm/`.

## Next Suggested Steps
1. Add per-algorithm README stubs (auto-generate template).
2. Create a `requirements.txt` enumerating core library versions (scikit-learn, pandas, numpy, etc.).
3. Introduce a lightweight package structure (e.g. `src/`) for shared preprocessing.
4. Add tests (pytest) for reusable functions once abstraction starts.
5. Version datasets separately or store large data outside repo if size grows.

---
Auto-generated on 2025-08-11.