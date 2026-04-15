```markdown
# 🏥 Disease Prediction from Medical Data (TASK 4)

## 📌 Description
Ce projet implémente un système de Machine Learning permettant de prédire si un patient est atteint d'une maladie à partir de ses données médicales.

Datasets utilisés :
- Heart Disease
- Diabetes
- Breast Cancer

---

## 🎯 Objectifs
- Prétraiter les données
- Entraîner plusieurs modèles
- Comparer les performances
- Faire des prédictions

---

## 🛠️ Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

---

## 📁 Structure
```

disease-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── src/
├── tests/
├── notebooks/
│
├── config.py
├── requirements.txt
├── predict_example.py
└── README.md

````

---

## ⚙️ Installation
```bash
git clone <repo-url>
cd disease-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
````

---

## 🚀 Utilisation

### Entraîner les modèles

```bash
python -m src.train
```

### Faire une prédiction

```bash
python predict_example.py
```

### Lancer les tests

```bash
python -m unittest tests/test_models.py -v
```

---

## 🧠 Modèles

* Logistic Regression
* Random Forest
* SVM
* XGBoost

---

## 📊 Évaluation

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

---

## ⚡ Fonctionnalités

* Gestion des valeurs manquantes
* Encodage des variables
* Normalisation
* SMOTE
* Cross-validation

---

## 🧪 Exemple

```python
from src.predict import DiseasePredictor

predictor = DiseasePredictor('heart_disease', 'random_forest')

patient = {
    'age': 50,
    'sex': 1,
    'cp': 3,
    'trestbps': 140,
    'chol': 200
}

result = predictor.predict(patient)
print(result)
```

---

## 👩‍💻 Auteur

Nadine Omri

---

## 🎉 Conclusion

Pipeline complet : Data → Modèle → Prédiction

```
```
