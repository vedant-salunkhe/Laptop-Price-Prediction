💻 Laptop Price Prediction Using Machine Learning

---1. Project Overview
This project predicts laptop prices based on different specifications such as brand, category, processor, graphics, RAM, and storage.

The main objective is to compare different Machine Learning algorithms and find the best model for laptop price prediction.

---2. Dataset
- Dataset: laptops_simple.csv
- Total Records: 943
- Total Columns: 8
- Target Variable: Price

---3. Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---4. Machine Learning Algorithms
Three regression algorithms were used:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

---5. Model Performance

| Algorithm | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 20286.89 | 29643.02 | 0.7956 |
| Decision Tree | 17990.11 | 34261.26 | 0.7270 |
| Random Forest | 15278.98 | 23883.48 | 0.8673 |

Best Model
Random Forest Regressor performed best with an R² Score of 0.8673.

---6. Project Files
- `app.py` – Streamlit application
- `trainmodel.py` – Model training
- `save_model.py` – Model saving
- `laptops_simple.csv` – Dataset
- `model.joblib` – Trained model
- `columns.joblib` – Saved feature columns

---7. Conclusion
This project helped me understand data preprocessing, One-Hot Encoding, regression algorithms, and model evaluation.
After comparing three algorithms, Random Forest Regressor provided the best prediction performance.


