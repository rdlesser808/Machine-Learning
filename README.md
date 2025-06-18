# Machine-Learning
Master's level coursework from Machine Learning module 

# Machine Learning Models for Mental Health Classification

This repository contains research and implementation of machine learning models for classifying mental health conditions based on symptom data. The work includes comparative analysis of multiple ML algorithms and their effectiveness in diagnosing mental health disorders.

## Overview

Mental health significantly impacts emotional, psychological, and social well-being. This project aims to develop accurate, accessible diagnostic tools using machine learning to assist in early identification and intervention of mental health conditions.

### Key Objectives
- Develop ML models for mental health condition classification
- Compare effectiveness of different algorithms (SVM, Decision Trees, XGBoost, TabNet)
- Create ensemble models for improved accuracy
- Address ethical considerations in ML-based healthcare applications

## Dataset

**Mental Disorder Classification Dataset**
- **Size**: 120 samples, 17 features, 4 classes
- **Source**: Public domain via Kaggle
- **Format**: CSV (Comma Separated Values)
- **Distribution**: Nearly balanced across classes (~25% each)
- **Source**:https://www.kaggle.com/datasets/cid007/mental-disorder-classification

### Features (Symptoms)
The dataset includes patient responses to 17 mental health symptoms:
- Sadness, Exhaustion, Euphoria
- Sleep disorder, Mood swings, Suicidal thoughts
- Anorexia, Anxiety, Try-explaining
- Nervous breakdown, Ignore & Move-on
- Admitting mistakes, Overthinking
- Aggressive response, Optimism
- Sexual activity, Concentration

### Target Classes
- **Bipolar Type-1 (Mania)**
- **Bipolar Type-2 (Depressive)**
- **Major Depressive Disorder**
- **Normal** (no mental health condition)

## Methodology

### Models Implemented

1. **Support Vector Classification (SVC)**
   - Ideal for binary classification tasks
   - Handles high-dimensional data effectively
   - Suitable for complex, non-linear relationships

2. **Decision Tree**
   - High interpretability
   - Clear decision rules crucial for clinical settings
   - Effective with categorical data

3. **XGBoost**
   - High performance and robustness
   - Advanced regularization techniques
   - Handles complex feature interactions

4. **TabNet**
   - Combines interpretability of decision trees with deep learning power
   - Uses attention mechanisms for feature selection
   - Particularly effective for tabular data

5. **Ensemble Model**
   - Stacking technique combining XGBoost and TabNet
   - Meta-model for improved performance
   - Leverages strengths of individual models

### Model Evaluation Metrics
- **Accuracy**: Overall classification effectiveness
- **Precision**: Agreement of data labels with positive predictions
- **Recall**: Model's ability to identify class labels
- **F1-Score**: Harmonic mean of precision and recall

## Results Summary

### Initial Model Comparison (75:25 split)
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVC | 0.67 | 0.45-1.00 | 0.56-1.00 | 0.62-0.75 |
| Decision Tree | 0.73 | 0.76 | 0.75 | 0.75 |
| XGBoost | 0.73 | 0.76 | 0.75 | 0.75 |

### Optimized Models Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost (Optimized) | 0.73 | 0.76 | 0.75 | 0.75 |
| TabNet (Optimized) | 0.70 | 0.70 | 0.81 | 0.79 |
| Ensemble Model | 0.83 | 0.83 | 0.84 | 0.82 |

### Key Findings
- **XGBoost** consistently performed well across different data splits
- **TabNet** showed significant improvement after hyperparameter tuning (~23% increase)
- **Ensemble model** achieved highest overall performance
- Perfect classification for Bipolar Type-1 in optimized models
- Challenges remain in distinguishing between Bipolar Type-2 and Depression

## Technical Implementation

### Software & Libraries
- **Python 3**: Primary programming language
- **pandas**: Data manipulation
- **NumPy**: Numerical operations
- **scikit-learn**: Model development and evaluation
- **XGBoost**: Gradient boosting framework
- **TabNet**: Attention-based neural networks
- **matplotlib/seaborn**: Data visualization
- **Jupyter Notebooks**: Development environment

### Hyperparameter Tuning
- **Grid Search**: Systematic parameter exploration
- **Random Search**: Efficient parameter sampling
- **Cross-Validation**: 5-fold and 10-fold validation
- **Early Stopping**: Prevent overfitting

## Repository Structure

```
├── ML AE1/
│   └── Machine Learning AE1 Report Final.pdf
    ├── ML AE1 Code Final.ipynb
├── ML AE2/
│   ├── Machine Learning AE2 Report Final.pdf
│   ├── ML AE2 Code Final.ipynb
└── README.md
```

## Limitations & Future Work

### Current Limitations
- **Small Dataset**: Only 120 samples limit model generalization
- **Computing Resources**: Limited hyperparameter exploration
- **Clinical Validation**: Requires extensive testing before clinical deployment

### Future Improvements
- **Larger Dataset**: Expand to thousands of samples for better reliability
- **Deep Learning**: Explore recurrent neural networks and advanced architectures
- **Feature Engineering**: Incorporate additional data sources (EHR, genetic data)
- **Bias Mitigation**: Ensure fairness across demographic groups
- **Real-time Deployment**: Develop production-ready systems

## Ethical Considerations

### Data Privacy
- Anonymized dataset usage
- Compliance with HIPAA and GDPR regulations
- Secure data handling protocols

### Clinical Responsibility
- Models supplement, not replace, professional diagnosis
- Continuous validation and monitoring required
- Transparent decision-making processes

### Bias and Fairness
- Regular algorithm audits
- Diverse training data representation
- Equitable treatment across patient groups

## Research Papers

This repository includes two comprehensive research papers:
1. **Initial Study**: Comparative analysis of SVM, Decision Tree, and XGBoost
2. **Extended Study**: Integration of TabNet and ensemble methods

Both papers provide detailed methodology, results, and ethical considerations for ML-based mental health classification.

## Acknowledgments

- World Health Organization for mental health statistics
- Kaggle community for dataset availability
- Research community for machine learning methodologies
- Healthcare professionals for domain expertise

---

**Disclaimer**: This research is for academic purposes only. The models are not intended for clinical diagnosis without proper validation and professional oversight. Always consult qualified mental health professionals for medical advice.
