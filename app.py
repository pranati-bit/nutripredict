import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriPredict",
    page_icon="🔬",
    layout="wide"
)

# ── STYLING ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Space+Mono&display=swap');

html, body, [class*="css"] { background-color: #0a0f1e; color: #e2eaf4; }
.stApp { background-color: #0a0f1e; }
#MainMenu, footer, header { visibility: hidden; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b35 100%);
    border: 1px solid #1e3050;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00cfff, #ffffff, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero p { color: #6b8caa; font-family: 'Space Mono', monospace; font-size: 0.8rem; margin-top: 0.5rem; letter-spacing: 2px; }

.kpi-box {
    background: #0d1b35;
    border: 1px solid #1e3050;
    border-top: 3px solid #00cfff;
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.kpi-label { color: #6b8caa; font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase; font-family: 'Space Mono', monospace; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #00cfff; }

.pred-card {
    background: linear-gradient(135deg, #0d1b35, #121e38);
    border: 1px solid #00cfff44;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.pred-num { font-family: 'Syne', sans-serif; font-size: 4.5rem; font-weight: 800; color: #00cfff; line-height: 1; }
.pred-sub { color: #6b8caa; font-size: 0.8rem; letter-spacing: 3px; font-family: 'Space Mono', monospace; margin-top: 0.5rem; }

.section-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #00cfff;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-left: 3px solid #00cfff;
    padding-left: 0.6rem;
    margin: 1.5rem 0 0.8rem;
}

.stButton > button {
    background: linear-gradient(90deg, #00cfff, #0099cc) !important;
    color: #0a0f1e !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    letter-spacing: 1px !important;
    padding: 0.5rem 1.5rem !important;
}
[data-testid="stSidebar"] { background: #0d1b35 !important; border-right: 1px solid #1e3050 !important; }
</style>
""", unsafe_allow_html=True)

# ── FUNCTIONS ─────────────────────────────────────────────────────────────────
def group_category(x):
    x = str(x)
    if any(w in x for w in ["Beef","Chicken","Pork","Fish","Seafood","Egg"]): return "Protein"
    elif any(w in x for w in ["Vegetable","Coleslaw","Potato"]): return "Vegetables"
    elif any(w in x for w in ["Fruit","Juice"]): return "Fruits"
    elif any(w in x for w in ["Grain","Rice","Pasta","Bread","Oatmeal"]): return "Grains"
    elif any(w in x for w in ["Milk","Dairy","Cheese","Ice cream"]): return "Dairy"
    elif any(w in x for w in ["Fast","Burger","Pizza","Restaurant"]): return "FastFood"
    elif "Baby" in x: return "BabyFood"
    elif any(w in x for w in ["Snack","Pretzel","Dessert","Cake"]): return "Snacks"
    else: return "Other"

@st.cache_data(show_spinner=False)
def fetch_data():
    API_KEY = "bZmXgiYs9nc7fNxnpms9eXBWE7RWbn8eyLg4NE3j"
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    all_rows = []
    for page in range(1, 10):
        try:
            r = requests.get(url, params={"api_key": API_KEY, "query": "food", "pageSize": 150, "pageNumber": page}, timeout=15)
            for food in r.json().get("foods", []):
                n = {x.get("nutrientName"): x.get("value", x.get("amount")) for x in food.get("foodNutrients", [])}
                all_rows.append({
                    "food_name": food.get("description"), "category": food.get("foodCategory"),
                    "protein": n.get("Protein"), "fat": n.get("Total lipid (fat)"),
                    "carbs": n.get("Carbohydrate, by difference"), "fiber": n.get("Fiber, total dietary"),
                    "sodium": n.get("Sodium, Na"), "potassium": n.get("Potassium, K"),
                    "calcium": n.get("Calcium, Ca"), "iron": n.get("Iron, Fe"),
                    "cholesterol": n.get("Cholesterol"), "energy": n.get("Energy")
                })
            time.sleep(0.3)
        except: pass
    return pd.DataFrame(all_rows)

@st.cache_data(show_spinner=False)
def build_pipeline(_df):
    df = _df.dropna().copy()
    df['category_grouped'] = df['category'].apply(group_category)
    df['energy_log'] = np.log(df['energy'].replace(0, np.nan))
    df = df[np.isfinite(df['energy_log'])]

    df1 = pd.get_dummies(df, columns=['category_grouped'])
    df1 = df1.drop(columns=['food_name', 'category'])

    num_cols = ['protein','fat','carbs','fiber','sodium','potassium','calcium','iron','cholesterol']
    scaler = StandardScaler()
    df1[num_cols] = scaler.fit_transform(df1[num_cols])

    Q1, Q3 = df1['energy_log'].quantile(0.25), df1['energy_log'].quantile(0.75)
    IQR = Q3 - Q1
    df1 = df1[(df1['energy_log'] >= Q1 - 1.5*IQR) & (df1['energy_log'] <= Q3 + 1.5*IQR)]
    df1 = df1.drop(columns=['energy'])

    y = df1['energy_log']
    x_all = df1.drop(columns=['energy_log'])

    alphas = 10**np.linspace(10, -2, 100) * 0.5
    ridge_cv = RidgeCV(alphas=alphas).fit(x_all, y)
    lasso_cv = LassoCV(alphas=alphas, max_iter=10000, cv=5).fit(x_all, y)

    ridge_model = Ridge(alpha=ridge_cv.alpha_).fit(x_all, y)
    lasso_model = Lasso(alpha=lasso_cv.alpha_, max_iter=10000).fit(x_all, y)

    feat_cols = ['protein','fat','carbs','sodium'] + [c for c in df1.columns if 'FastFood' in c]
    x = df1[feat_cols].copy()
    x['fat_carbs']    = x['fat'] * x['carbs']
    x['fat_protein']  = x['fat'] * x['protein']
    x['carbs_protein']= x['carbs'] * x['protein']
    mask = x.notna().all(axis=1) & y.notna()
    x, y_f = x[mask], y[mask]

    lr = LinearRegression().fit(x, y_f)
    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    cv = cross_val_score(lr, x, y_f, cv=kf, scoring='neg_mean_squared_error')
    rmse = float(np.sqrt(np.mean(-cv)))

    return df, df1, scaler, ridge_model, lasso_model, lr, x, y_f, x_all, y, rmse, ridge_cv.alpha_, lasso_cv.alpha_

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero"><h1>🔬 NutriPredict</h1><p>CALORIE INTELLIGENCE · USDA · RIDGE · LASSO · ANOVA · OLS</p></div>', unsafe_allow_html=True)

with st.spinner("Fetching USDA food data..."):
    raw_df = fetch_data()

if raw_df.empty:
    st.error("Failed to fetch data. Check internet connection.")
    st.stop()

with st.spinner("Training models..."):
    df, df1, scaler, ridge_model, lasso_model, lr_model, x_tr, y_tr, x_all, y_all, rmse, best_r_a, best_l_a = build_pipeline(raw_df)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, label, val in zip(
    [c1, c2, c3, c4],
    ["Foods Fetched", "After Cleaning", "CV RMSE", "Features Used"],
    [f"{len(raw_df):,}", f"{len(df1):,}", f"{rmse:.4f}", f"{x_tr.shape[1]}"]
):
    col.markdown(f'<div class="kpi-box"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["⚡ Predictor", "📊 Statistics", "🤖 Models", "📋 Data"])

# ══ TAB 1: PREDICTOR ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-tag">// Live Calorie Predictor</div>', unsafe_allow_html=True)
    col_in, col_out = st.columns([1, 1])

    with col_in:
        protein  = st.slider("🥩 Protein (g)",      0.0, 100.0, 20.0, 0.5)
        fat      = st.slider("🧈 Fat (g)",           0.0, 100.0, 10.0, 0.5)
        carbs    = st.slider("🌾 Carbohydrates (g)", 0.0, 150.0, 30.0, 0.5)
        sodium   = st.slider("🧂 Sodium (mg)",       0.0, 2000.0, 300.0, 10.0)
        is_ff    = st.toggle("🍔 Is Fast Food?", value=False)

    with col_out:
        # Scale inputs same as training
        raw_arr = np.array([[protein, fat, carbs, 0, sodium, 0, 0, 0, 0]])
        scaled  = scaler.transform(raw_arr)[0]
        p_s, f_s, c_s, so_s = scaled[0], scaled[1], scaled[2], scaled[4]
        ff = 1.0 if is_ff else 0.0
        feats = np.array([[p_s, f_s, c_s, so_s, ff, f_s*c_s, f_s*p_s, c_s*p_s]])
        pred_cal = float(np.exp(lr_model.predict(feats)[0]))

        if pred_cal < 100:   level, color = "Very Low Calorie",  "#39ff14"
        elif pred_cal < 200: level, color = "Low Calorie",       "#00cfff"
        elif pred_cal < 350: level, color = "Moderate Calorie",  "#ffd700"
        elif pred_cal < 500: level, color = "High Calorie",      "#ff6b35"
        else:                level, color = "Very High Calorie", "#ff2d55"

        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-sub">PREDICTED CALORIES</div>
            <div class="pred-num" style="color:{color}">{pred_cal:.0f}</div>
            <div class="pred-sub">kcal per 100g</div>
            <br>
            <span style="background:{color}22; border:1px solid {color}; color:{color};
            padding:5px 18px; border-radius:20px; font-size:0.85rem; font-weight:700">
            {level}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Macro bar
        total = protein*4 + fat*9 + carbs*4
        if total > 0:
            fig, ax = plt.subplots(figsize=(5, 0.6))
            fig.patch.set_facecolor('#0d1b35')
            ax.set_facecolor('#0d1b35')
            pct_p = protein*4/total
            pct_f = fat*9/total
            pct_c = carbs*4/total
            ax.barh(0, pct_p, color='#00cfff', height=0.5)
            ax.barh(0, pct_f, left=pct_p, color='#ff6b35', height=0.5)
            ax.barh(0, pct_c, left=pct_p+pct_f, color='#ffd700', height=0.5)
            ax.set_xlim(0, 1); ax.axis('off')
            patches = [mpatches.Patch(color='#00cfff', label=f'Protein {pct_p*100:.0f}%'),
                       mpatches.Patch(color='#ff6b35', label=f'Fat {pct_f*100:.0f}%'),
                       mpatches.Patch(color='#ffd700', label=f'Carbs {pct_c*100:.0f}%')]
            ax.legend(handles=patches, loc='upper center', ncol=3, frameon=False,
                      labelcolor='white', fontsize=8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ══ TAB 2: STATISTICS ═════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-tag">// ANOVA — Does Category Affect Calories?</div>', unsafe_allow_html=True)
    anova_model = ols('energy_log ~ C(category_grouped)', data=df).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)
    p_val = anova_table['PR(>F)'].iloc[0]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(anova_table, use_container_width=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">p-value</div>
            <div class="kpi-value" style="font-size:1.3rem; color:{'#39ff14' if p_val<0.05 else '#ff2d55'}">{p_val:.2e}</div>
            <div class="kpi-label" style="margin-top:0.3rem">{'✅ Significant' if p_val<0.05 else '❌ Not Significant'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-tag">// T-Test — FastFood vs Grains</div>', unsafe_allow_html=True)
    c1_t = df[df['category_grouped'] == 'FastFood']['energy_log']
    c2_t = df[df['category_grouped'] == 'Grains']['energy_log']
    if len(c1_t) > 1 and len(c2_t) > 1:
        t_stat, p_t = stats.ttest_ind(c1_t, c2_t, equal_var=True, alternative='greater')
        col1, col2, col3 = st.columns(3)
        col1.metric("FastFood Mean log(cal)", f"{c1_t.mean():.3f}")
        col2.metric("Grains Mean log(cal)", f"{c2_t.mean():.3f}")
        col3.metric("p-value", f"{p_t:.4f}", "FastFood > Grains ✅" if p_t < 0.05 else "Not significant")

    st.markdown('<div class="section-tag">// Calorie Distribution by Category</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0d1b35')
    ax.set_facecolor('#0d1b35')
    cats = sorted(df['category_grouped'].unique())
    data_by_cat = [df[df['category_grouped'] == c]['energy_log'].dropna().values for c in cats]
    bp = ax.boxplot(data_by_cat, labels=cats, patch_artist=True, notch=False)
    colors = ['#00cfff','#ff6b35','#ffd700','#39ff14','#ff2d55','#c77dff','#06d6a0','#ff9f1c','#e63946']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color + '55')
        patch.set_edgecolor(color)
    for element in ['whiskers','caps','medians','fliers']:
        for item in bp[element]:
            item.set_color('#aabbcc')
    ax.set_xlabel('Category', color='#aabbcc')
    ax.set_ylabel('log(Energy)', color='#aabbcc')
    ax.tick_params(colors='#aabbcc', rotation=30)
    ax.spines[:].set_color('#1e3050')
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-tag">// Nutrient Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = ['protein','fat','carbs','fiber','sodium','potassium','calcium','iron','cholesterol','energy']
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0d1b35')
    ax.set_facecolor('#0d1b35')
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                annot_kws={'size': 7, 'color': 'white'},
                linewidths=0.5, linecolor='#1e3050',
                cbar_kws={'shrink': 0.8})
    ax.tick_params(colors='#aabbcc')
    plt.xticks(rotation=45, ha='right', color='#aabbcc')
    plt.yticks(rotation=0, color='#aabbcc')
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══ TAB 3: MODELS ═════════════════════════════════════════════════════════════
with tab3:
    col1, col2, col3 = st.columns(3)
    col1.metric("10-Fold CV RMSE", f"{rmse:.4f}", "log scale")
    col2.metric("Best Ridge α", f"{best_r_a:.5f}")
    col3.metric("Best Lasso α", f"{best_l_a:.5f}")

    st.markdown('<div class="section-tag">// Actual vs Predicted</div>', unsafe_allow_html=True)
    y_pred = lr_model.predict(x_tr)
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0d1b35')
    ax.set_facecolor('#0d1b35')
    ax.scatter(y_tr, y_pred, alpha=0.4, color='#00cfff', s=15, label='Predictions')
    mn, mx = float(y_tr.min()), float(y_tr.max())
    ax.plot([mn, mx], [mn, mx], color='#ff6b35', linestyle='--', lw=2, label='Perfect fit')
    ax.set_xlabel('Actual log(Energy)', color='#aabbcc')
    ax.set_ylabel('Predicted log(Energy)', color='#aabbcc')
    ax.tick_params(colors='#aabbcc')
    ax.spines[:].set_color('#1e3050')
    ax.legend(frameon=False, labelcolor='white')
    st.pyplot(fig, use_container_width=True)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-tag">// Ridge Coefficients</div>', unsafe_allow_html=True)
        ridge_coef = pd.Series(ridge_model.coef_, index=x_all.columns).sort_values()
        fig, ax = plt.subplots(figsize=(5, 6))
        fig.patch.set_facecolor('#0d1b35')
        ax.set_facecolor('#0d1b35')
        colors_bar = ['#00cfff' if v >= 0 else '#ff6b35' for v in ridge_coef.values]
        ax.barh(ridge_coef.index, ridge_coef.values, color=colors_bar)
        ax.tick_params(colors='#aabbcc', labelsize=7)
        ax.spines[:].set_color('#1e3050')
        ax.set_xlabel('Coefficient', color='#aabbcc')
        ax.axvline(0, color='#aabbcc', lw=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-tag">// Lasso Coefficients</div>', unsafe_allow_html=True)
        lasso_coef = pd.Series(lasso_model.coef_, index=x_all.columns)
        lasso_nonzero = lasso_coef[lasso_coef != 0].sort_values()
        fig, ax = plt.subplots(figsize=(5, 6))
        fig.patch.set_facecolor('#0d1b35')
        ax.set_facecolor('#0d1b35')
        colors_bar = ['#39ff14' if v >= 0 else '#ff2d55' for v in lasso_nonzero.values]
        ax.barh(lasso_nonzero.index, lasso_nonzero.values, color=colors_bar)
        ax.tick_params(colors='#aabbcc', labelsize=7)
        ax.spines[:].set_color('#1e3050')
        ax.set_xlabel('Coefficient', color='#aabbcc')
        ax.axvline(0, color='#aabbcc', lw=0.5)
        ax.set_title(f'{len(lasso_nonzero)} features selected by Lasso', color='#aabbcc', fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-tag">// Residual Distribution</div>', unsafe_allow_html=True)
    residuals = y_tr.values - y_pred
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#0d1b35')
    ax.set_facecolor('#0d1b35')
    ax.hist(residuals, bins=40, color='#00cfff', edgecolor='#0a0f1e', alpha=0.8)
    ax.axvline(0, color='#ff6b35', lw=2, linestyle='--', label='Zero residual')
    ax.set_xlabel('Residual', color='#aabbcc')
    ax.set_ylabel('Count', color='#aabbcc')
    ax.tick_params(colors='#aabbcc')
    ax.spines[:].set_color('#1e3050')
    ax.legend(frameon=False, labelcolor='white')
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══ TAB 4: DATA ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-tag">// Raw Dataset Explorer</div>', unsafe_allow_html=True)
    search = st.text_input("Search food name", placeholder="e.g. chicken, apple, rice...")
    disp = raw_df.dropna()
    if search:
        disp = disp[disp['food_name'].str.contains(search, case=False, na=False)]
    st.dataframe(
        disp[['food_name','category','protein','fat','carbs','sodium','energy']].head(300),
        use_container_width=True, height=420
    )
    st.caption(f"Showing {min(len(disp), 300)} of {len(disp)} records")
