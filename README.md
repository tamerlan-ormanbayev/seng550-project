# Air Pollution Risk Classification in Dublin

This project implements a machine learning-based system to classify air pollution risks in Dublin using CO monitoring data. It leverages PySpark for scalable data processing and real-time classification.

## Overview
- **Goal:** Categorize pollution levels into risk categories (`Low`, `Moderate`, `High`, `Very High`) and predict when pollution may exceed thresholds.
- **Dataset:** Hourly CO levels from two Dublin monitoring sites (2011 data).
- **Algorithm:** Random Forest Classifier for classification.

## Key Features
- **Data Processing:** Automated preprocessing, handling missing values, and feature engineering.
- **Classification:** Pollution levels categorized based on CO 8-hour averages:
  - `0 (Low): ≤ 0.2`
  - `1 (Moderate): 0.2-0.4`
  - `2 (High): 0.4-0.6`
  - `3 (Very High): > 0.6`
- **Evaluation:** Metrics include accuracy, precision, recall, F1-score, and confusion matrix.

## Results
- **Winetavern Street Model:** Accuracy: 84.34%
- **Coleraine Street Model:** Accuracy: 67.11%
- Overfitting was identified due to high training accuracy compared to test accuracy.
- Future work includes incorporating more features and data for improved performance.

## Authors
- Tamerlan Ormanbayev
- Joshua Debele

## Acknowledgements
This project was part of the SENG-550 Group Project at the University of Calgary.
