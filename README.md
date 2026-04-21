# 🔬 NutriPredict — Calorie Intelligence

Streamlit app that predicts caloric content of foods using ML on USDA food data.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stack
- Data: USDA FoodData Central API
- Models: Ridge, Lasso, Linear Regression (10-Fold CV)
- Stats: ANOVA, Tukey HSD, T-Test
- UI: Streamlit + Matplotlib + Seaborn
