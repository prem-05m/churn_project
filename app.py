import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb
import shap
import os

st.set_page_config(
    page_title="ChurnSense + AdmitIQ",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0d14; color: #ffffff; }
    .metric-card {
        background: linear-gradient(135deg, #141829, #1a1f35);
        border-radius: 12px; padding: 1.2rem 1.5rem;
        border: 1px solid #2a2f4a; margin-bottom: 0.5rem;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #7c83ff; }
    .metric-val-green { font-size: 2rem; font-weight: 700; color: #10b981; }
    .metric-lbl { font-size: 0.82rem; color: #9ca3af; margin-top: 2px; }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        border-bottom: 2px solid #7c83ff;
        padding-bottom: 6px; margin-bottom: 1rem; color: #e2e8f0;
    }
    .section-header-green {
        font-size: 1.1rem; font-weight: 600;
        border-bottom: 2px solid #10b981;
        padding-bottom: 6px; margin-bottom: 1rem; color: #e2e8f0;
    }
    .pill { display:inline-block; padding: 3px 12px; border-radius: 99px; font-size: 12px; font-weight: 600; }
    .pill-churn  { background:#ff4b4b22; color:#ff4b4b; border:1px solid #ff4b4b55; }
    .pill-stay   { background:#21c55d22; color:#21c55d; border:1px solid #21c55d55; }
    .pill-admit  { background:#10b98122; color:#10b981; border:1px solid #10b98155; }
    .pill-reject { background:#f59e0b22; color:#f59e0b; border:1px solid #f59e0b55; }
    .stButton>button {
        background: linear-gradient(135deg, #7c83ff, #a78bfa);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 2rem; font-weight: 600; width: 100%;
    }
    div[data-testid="stSidebar"] { background-color: #0d1020; }
    .module-badge-churn {
        display:inline-block; background: #7c83ff22;
        border: 1px solid #7c83ff55; border-radius: 8px;
        padding: 4px 14px; font-size: 11px; font-weight: 700;
        color: #a78bfa; letter-spacing: 1px; text-transform: uppercase;
    }
    .module-badge-admit {
        display:inline-block; background: #10b98122;
        border: 1px solid #10b98155; border-radius: 8px;
        padding: 4px 14px; font-size: 11px; font-weight: 700;
        color: #10b981; letter-spacing: 1px; text-transform: uppercase;
    }
    .info-box {
        background: #141829; border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 3px solid #10b981; margin-bottom: 1rem;
        font-size: 0.9rem; color: #d1fae5;
    }
    .warn-box {
        background: #141829; border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 3px solid #f59e0b; margin-bottom: 1rem;
        font-size: 0.9rem; color: #fef3c7;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL — CHURN
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_churn_data():
    np.random.seed(42)
    n = 3000
    tenure = np.random.randint(1, 72, n)
    monthly = np.round(np.random.uniform(18, 118, n), 2)
    total = np.round(tenure * monthly * np.random.uniform(0.85, 1.05, n), 2)
    gender = np.random.choice(['Male', 'Female'], n)
    senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n)
    dependents = np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])
    contract = np.random.choice(['Month-to-month','One year','Two year'], n, p=[0.55,0.25,0.20])
    internet = np.random.choice(['DSL','Fiber optic','No'], n, p=[0.34,0.44,0.22])
    payment = np.random.choice(['Electronic check','Mailed check','Bank transfer','Credit card'], n)
    paperless = np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
    tech_support = np.random.choice(['Yes','No','No internet service'], n)
    online_backup = np.random.choice(['Yes','No','No internet service'], n)
    churn_prob = (
        0.35*(contract=='Month-to-month').astype(float)+0.20*(internet=='Fiber optic').astype(float)+
        0.15*(tenure<12).astype(float)+0.10*(monthly>80).astype(float)+0.08*senior.astype(float)-
        0.12*(contract=='Two year').astype(float)-0.08*(tech_support=='Yes').astype(float)+
        np.random.normal(0,0.08,n)
    )
    churn = (np.random.rand(n) < np.clip(churn_prob,0.04,0.92)).astype(int)
    return pd.DataFrame({'gender':gender,'SeniorCitizen':senior,'Partner':partner,'Dependents':dependents,
        'tenure':tenure,'Contract':contract,'PaperlessBilling':paperless,'PaymentMethod':payment,
        'InternetService':internet,'TechSupport':tech_support,'OnlineBackup':online_backup,
        'MonthlyCharges':monthly,'TotalCharges':total,'Churn':churn})

@st.cache_resource
def train_churn_models(df):
    le = LabelEncoder()
    cat_cols = ['gender','Partner','Dependents','Contract','PaperlessBilling',
                'PaymentMethod','InternetService','TechSupport','OnlineBackup']
    df2 = df.copy()
    for c in cat_cols: df2[c] = le.fit_transform(df2[c])
    X = df2.drop('Churn',axis=1); y = df2['Churn']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler = StandardScaler(); X_tr_s = scaler.fit_transform(X_train); X_te_s = scaler.transform(X_test)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000,random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=150,learning_rate=0.1,use_label_encoder=False,eval_metric='logloss',random_state=42,verbosity=0)
    }
    results={}; trained={}
    for name,model in models.items():
        if name=='Logistic Regression': model.fit(X_tr_s,y_train); preds=model.predict(X_te_s); proba=model.predict_proba(X_te_s)[:,1]
        else: model.fit(X_train,y_train); preds=model.predict(X_test); proba=model.predict_proba(X_test)[:,1]
        results[name]={'accuracy':round(accuracy_score(y_test,preds)*100,2),'precision':round(precision_score(y_test,preds)*100,2),
            'recall':round(recall_score(y_test,preds)*100,2),'f1':round(f1_score(y_test,preds)*100,2),
            'auc':round(roc_auc_score(y_test,proba)*100,2),'cm':confusion_matrix(y_test,preds),'proba':proba,'preds':preds}
        trained[name]=model
    return trained,results,X_train,X_test,y_train,y_test,scaler,X.columns.tolist(),df2


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL — ADMISSION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_admission_data():
    np.random.seed(99)
    n = 2500
    gre = np.random.normal(316,11,n).clip(260,340).astype(int)
    toefl = np.random.normal(107,6,n).clip(92,120).astype(int)
    sop = np.round(np.random.uniform(1.5,5.0,n)*2)/2
    lor = np.round(np.random.uniform(1.5,5.0,n)*2)/2
    cgpa = np.round(np.random.normal(8.6,0.6,n).clip(6.8,10.0),2)
    research = np.random.choice([0,1],n,p=[0.45,0.55])
    uni_rating = np.random.choice([1,2,3,4,5],n,p=[0.05,0.15,0.30,0.30,0.20])
    extracurricular = np.random.choice([0,1],n,p=[0.40,0.60])
    internships = np.random.randint(0,4,n)
    projects = np.random.randint(0,6,n)
    work_exp = np.random.randint(0,5,n)
    backlog = np.random.choice([0,1,2,3],n,p=[0.65,0.20,0.10,0.05])
    course_type = np.random.choice(['Engineering','Management','Science','Arts','Medicine'],n,p=[0.35,0.25,0.20,0.10,0.10])
    scholarship = np.random.choice([0,1],n,p=[0.70,0.30])
    alumni_rel = np.random.choice([0,1],n,p=[0.80,0.20])
    admit_score = (0.25*(gre-260)/80+0.15*(toefl-92)/28+0.20*(cgpa-6.8)/3.2+0.10*(uni_rating-1)/4+
                   0.08*sop/5+0.07*lor/5+0.06*research+0.03*extracurricular+0.03*(internships/3)+
                   0.02*alumni_rel-0.05*(backlog>0).astype(float)+np.random.normal(0,0.05,n))
    admitted = (np.random.rand(n) < np.clip(admit_score,0.05,0.98)).astype(int)
    return pd.DataFrame({'GRE_Score':gre,'TOEFL_Score':toefl,'University_Rating':uni_rating,
        'SOP':sop,'LOR':lor,'CGPA':cgpa,'Research':research,'Extracurricular':extracurricular,
        'Internships':internships,'Projects':projects,'Work_Experience':work_exp,'Backlog':backlog,
        'Course_Type':course_type,'Scholarship':scholarship,'Alumni_Relation':alumni_rel,'Admitted':admitted})

@st.cache_resource
def train_admission_models(df):
    df2 = df.copy(); le = LabelEncoder(); df2['Course_Type'] = le.fit_transform(df2['Course_Type'])
    X = df2.drop('Admitted',axis=1); y = df2['Admitted']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler = StandardScaler(); X_tr_s = scaler.fit_transform(X_train); X_te_s = scaler.transform(X_test)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000,random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=200,learning_rate=0.08,use_label_encoder=False,eval_metric='logloss',random_state=42,verbosity=0),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150,random_state=42)
    }
    results={}; trained={}
    for name,model in models.items():
        if name=='Logistic Regression': model.fit(X_tr_s,y_train); preds=model.predict(X_te_s); proba=model.predict_proba(X_te_s)[:,1]
        else: model.fit(X_train,y_train); preds=model.predict(X_test); proba=model.predict_proba(X_test)[:,1]
        results[name]={'accuracy':round(accuracy_score(y_test,preds)*100,2),'precision':round(precision_score(y_test,preds)*100,2),
            'recall':round(recall_score(y_test,preds)*100,2),'f1':round(f1_score(y_test,preds)*100,2),
            'auc':round(roc_auc_score(y_test,proba)*100,2),'cm':confusion_matrix(y_test,preds),'proba':proba,'preds':preds}
        trained[name]=model
    return trained,results,X_train,X_test,y_train,y_test,scaler,X.columns.tolist(),le


def plot_dark(fig):
    fig.patch.set_facecolor('#141829')
    for ax in fig.get_axes():
        ax.set_facecolor('#141829'); ax.tick_params(colors='#9ca3af')
        ax.xaxis.label.set_color('#9ca3af'); ax.yaxis.label.set_color('#9ca3af')
        ax.title.set_color('#e2e8f0')
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2f4a')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎓 ChurnSense + AdmitIQ")
st.sidebar.markdown("*Dual Intelligent Prediction Platform*")
st.sidebar.markdown("---")
module = st.sidebar.selectbox("🔀 Select Module", ["📉  Customer Churn Prediction","🎓  College Admission Prediction"])
st.sidebar.markdown("---")
if "Churn" in module:
    page = st.sidebar.radio("Navigation", ["🏠  Overview","📊  Exploratory Analysis","🤖  Model Comparison","🔍  Explainability (SHAP)","🎯  Predict a Customer"])
    st.sidebar.markdown("**Dataset:** Synthetic Telco (3,000 customers)")
    st.sidebar.markdown("**Models:** LR · RF · XGBoost")
    st.sidebar.markdown("**Best Model:** XGBoost")
else:
    page = st.sidebar.radio("Navigation", ["🏠  Overview","📊  Exploratory Analysis","🤖  Model Comparison","🔍  Explainability (SHAP)","🎓  Predict an Applicant","📋  Diagrams & Architecture"])
    st.sidebar.markdown("**Dataset:** Synthetic Admissions (2,500 students)")
    st.sidebar.markdown("**Models:** LR · RF · XGBoost · GBM")
    st.sidebar.markdown("**Best Model:** XGBoost")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading data and training models…"):
    df_churn = load_churn_data()
    c_trained,c_results,c_Xtr,c_Xte,c_ytr,c_yte,c_scaler,c_feats,df_ce = train_churn_models(df_churn)
    df_admit = load_admission_data()
    a_trained,a_results,a_Xtr,a_Xte,a_ytr,a_yte,a_scaler,a_feats,adm_le = train_admission_models(df_admit)

best_c = c_trained['XGBoost']; best_cr = c_results['XGBoost']
best_a = a_trained['XGBoost']; best_ar = a_results['XGBoost']


# ══════════════════════════════════════════════════════════════════════════════
# MODULE A — CHURN
# ══════════════════════════════════════════════════════════════════════════════
if "Churn" in module:

    if page == "🏠  Overview":
        st.markdown('<span class="module-badge-churn">Customer Churn Module</span>', unsafe_allow_html=True)
        st.title("Customer Churn Prediction System")
        st.markdown("Full ML pipeline: data engineering → EDA → model comparison → explainability → live prediction.")
        st.markdown("---")
        churn_rate = round(df_churn['Churn'].mean()*100,1)
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><div class="metric-val">{len(df_churn):,}</div><div class="metric-lbl">Total customers</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="metric-val">{churn_rate}%</div><div class="metric-lbl">Churn rate</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><div class="metric-val">{best_cr["auc"]}%</div><div class="metric-lbl">Best AUC-ROC</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card"><div class="metric-val">{best_cr["f1"]}%</div><div class="metric-lbl">Best F1 Score</div></div>', unsafe_allow_html=True)
        st.markdown("---")
        col1,col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Churn by contract type</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(6,3.5))
            ct = df_churn.groupby('Contract')['Churn'].mean().reset_index(); ct['Churn']*=100
            ax.barh(ct['Contract'],ct['Churn'],color=['#7c83ff','#a78bfa','#c4b5fd'],height=0.5); ax.set_xlabel('Churn Rate (%)')
            for i,v in enumerate(ct['Churn']): ax.text(v+1,i,f'{v:.1f}%',va='center',color='#e2e8f0',fontsize=10)
            st.pyplot(plot_dark(fig)); plt.close()
        with col2:
            st.markdown('<div class="section-header">Monthly charges vs churn</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(6,3.5))
            ax.hist(df_churn[df_churn['Churn']==0]['MonthlyCharges'],bins=30,alpha=0.7,color='#21c55d',label='Stayed',density=True)
            ax.hist(df_churn[df_churn['Churn']==1]['MonthlyCharges'],bins=30,alpha=0.7,color='#ff4b4b',label='Churned',density=True)
            ax.legend(facecolor='#141829',labelcolor='white'); ax.set_xlabel('Monthly Charges ($)')
            st.pyplot(plot_dark(fig)); plt.close()

    elif page == "📊  Exploratory Analysis":
        st.markdown('<span class="module-badge-churn">Customer Churn Module</span>', unsafe_allow_html=True)
        st.title("Exploratory Data Analysis — Churn")
        tab1,tab2,tab3 = st.tabs(["Distribution","Correlations","Churn Drivers"])
        with tab1:
            col1,col2 = st.columns(2)
            with col1:
                fig,ax = plt.subplots(figsize=(5,3))
                ax.hist(df_churn[df_churn['Churn']==0]['tenure'],bins=25,alpha=0.7,color='#21c55d',label='Stayed',density=True)
                ax.hist(df_churn[df_churn['Churn']==1]['tenure'],bins=25,alpha=0.7,color='#ff4b4b',label='Churned',density=True)
                ax.legend(facecolor='#141829',labelcolor='white',fontsize=9); ax.set_xlabel('Tenure (months)')
                st.markdown('<div class="section-header">Tenure distribution</div>', unsafe_allow_html=True)
                st.pyplot(plot_dark(fig)); plt.close()
            with col2:
                fig,ax = plt.subplots(figsize=(5,3))
                sc = df_churn.groupby('SeniorCitizen')['Churn'].mean()*100
                ax.bar(['Non-Senior','Senior'],sc.values,color=['#7c83ff','#a78bfa'],width=0.5); ax.set_ylabel('Churn Rate (%)')
                for i,v in enumerate(sc.values): ax.text(i,v+0.5,f'{v:.1f}%',ha='center',color='#e2e8f0',fontsize=11)
                st.markdown('<div class="section-header">Senior citizens</div>', unsafe_allow_html=True)
                st.pyplot(plot_dark(fig)); plt.close()
        with tab2:
            num_cols = ['tenure','MonthlyCharges','TotalCharges','SeniorCitizen','Churn']
            fig,ax = plt.subplots(figsize=(7,5))
            sns.heatmap(df_churn[num_cols].corr(),annot=True,fmt='.2f',cmap='coolwarm',ax=ax,linewidths=0.5,linecolor='#0a0d14',annot_kws={'size':11,'color':'white'})
            ax.set_xticklabels(ax.get_xticklabels(),rotation=30,ha='right',fontsize=10)
            st.pyplot(plot_dark(fig)); plt.close()
        with tab3:
            factors={'Month-to-month contract':42.1,'Fiber optic internet':35.8,'Tenure < 12 months':31.4,'Monthly charges > $80':28.7,'Senior citizen':24.3,'No tech support':19.6,'Electronic check payment':17.2}
            fig,ax = plt.subplots(figsize=(8,4))
            keys=list(factors.keys()); vals=list(factors.values())
            colors=['#ff4b4b' if v>30 else '#f59e0b' if v>20 else '#7c83ff' for v in vals]
            ax.barh(keys,vals,color=colors,height=0.55); ax.set_xlabel('Churn Rate (%)')
            for i,v in enumerate(vals): ax.text(v+0.5,i,f'{v}%',va='center',color='#e2e8f0',fontsize=10)
            st.pyplot(plot_dark(fig)); plt.close()
            st.info("Red = high risk (>30%), Orange = medium (20-30%), Blue = moderate")

    elif page == "🤖  Model Comparison":
        st.markdown('<span class="module-badge-churn">Customer Churn Module</span>', unsafe_allow_html=True)
        st.title("Model Comparison — Churn")
        mdf = pd.DataFrame({'Model':list(c_results.keys()),'Accuracy':[r['accuracy'] for r in c_results.values()],'Precision':[r['precision'] for r in c_results.values()],'Recall':[r['recall'] for r in c_results.values()],'F1 Score':[r['f1'] for r in c_results.values()],'AUC-ROC':[r['auc'] for r in c_results.values()]}).set_index('Model')
        st.dataframe(mdf.style.highlight_max(axis=0,color='#2d3f2d').format("{:.2f}"),use_container_width=True)
        col1,col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">ROC curves</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(5,4))
            for (name,res),color in zip(c_results.items(),['#7c83ff','#21c55d','#f59e0b']):
                fpr,tpr,_ = roc_curve(c_yte,res['proba']); ax.plot(fpr,tpr,label=f"{name} ({res['auc']}%)",color=color,lw=1.8)
            ax.plot([0,1],[0,1],'w--',lw=0.8,alpha=0.4); ax.legend(facecolor='#141829',labelcolor='white',fontsize=9)
            st.pyplot(plot_dark(fig)); plt.close()
        with col2:
            st.markdown('<div class="section-header">Confusion matrix (XGBoost)</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(5,4))
            sns.heatmap(best_cr['cm'],annot=True,fmt='d',cmap='Blues',ax=ax,xticklabels=['Stay','Churn'],yticklabels=['Stay','Churn'],annot_kws={'size':16,'weight':'bold'})
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            st.pyplot(plot_dark(fig)); plt.close()

    elif page == "🔍  Explainability (SHAP)":
        st.markdown('<span class="module-badge-churn">Customer Churn Module</span>', unsafe_allow_html=True)
        st.title("Model Explainability with SHAP — Churn")
        with st.spinner("Computing SHAP values…"):
            exp = shap.TreeExplainer(best_c); X_s = c_Xte.iloc[:200]; sv = exp.shap_values(X_s)
        col1,col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Global feature impact</div>', unsafe_allow_html=True)
            fig,_ = plt.subplots(figsize=(6,5)); shap.summary_plot(sv,X_s,plot_type="dot",show=False,max_display=10)
            st.pyplot(plot_dark(plt.gcf())); plt.close()
        with col2:
            st.markdown('<div class="section-header">Mean absolute SHAP values</div>', unsafe_allow_html=True)
            fig,_ = plt.subplots(figsize=(6,5)); shap.summary_plot(sv,X_s,plot_type="bar",show=False,max_display=10)
            st.pyplot(plot_dark(plt.gcf())); plt.close()
        st.markdown('<div class="section-header">Individual customer explanation</div>', unsafe_allow_html=True)
        idx = st.slider("Select a customer from the test set",0,len(c_Xte)-1,0)
        row = c_Xte.iloc[[idx]]; sv_r = exp.shap_values(row)[0]
        prob = best_c.predict_proba(row)[0][1]
        st.markdown(f'Prediction: <span class="pill {"pill-churn" if prob>0.5 else "pill-stay"}">{"Will Churn" if prob>0.5 else "Will Stay"}</span>  — confidence: **{prob*100:.1f}%**', unsafe_allow_html=True)
        fig,_ = plt.subplots(figsize=(8,4))
        shap.waterfall_plot(shap.Explanation(values=sv_r,base_values=exp.expected_value,data=row.values[0],feature_names=c_feats),show=False,max_display=10)
        st.pyplot(plot_dark(plt.gcf())); plt.close()

    elif page == "🎯  Predict a Customer":
        st.markdown('<span class="module-badge-churn">Customer Churn Module</span>', unsafe_allow_html=True)
        st.title("Live Customer Churn Prediction")
        with st.form("predict_churn"):
            c1,c2,c3 = st.columns(3)
            with c1:
                gender=st.selectbox("Gender",["Male","Female"]); senior=st.selectbox("Senior Citizen",["No","Yes"])
                partner=st.selectbox("Partner",["Yes","No"]); dependents=st.selectbox("Dependents",["No","Yes"])
                tenure=st.slider("Tenure (months)",1,72,12)
            with c2:
                contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
                internet=st.selectbox("Internet Service",["Fiber optic","DSL","No"])
                tech_support=st.selectbox("Tech Support",["No","Yes","No internet service"])
                online_backup=st.selectbox("Online Backup",["No","Yes","No internet service"])
                paperless=st.selectbox("Paperless Billing",["Yes","No"])
            with c3:
                payment=st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer","Credit card"])
                monthly=st.slider("Monthly Charges ($)",18,118,65)
                total=st.number_input("Total Charges ($)",value=float(tenure*monthly),min_value=0.0)
            submitted=st.form_submit_button("🔍 Predict Churn Risk")
        if submitted:
            cm = {'gender':{'Male':1,'Female':0},'Partner':{'Yes':1,'No':0},'Dependents':{'Yes':1,'No':0},
                  'Contract':{'Month-to-month':0,'One year':1,'Two year':2},'PaperlessBilling':{'Yes':1,'No':0},
                  'PaymentMethod':{'Bank transfer':0,'Credit card':1,'Electronic check':2,'Mailed check':3},
                  'InternetService':{'DSL':0,'Fiber optic':1,'No':2},
                  'TechSupport':{'No':0,'No internet service':1,'Yes':2},'OnlineBackup':{'No':0,'No internet service':1,'Yes':2}}
            row=pd.DataFrame([{'gender':cm['gender'][gender],'SeniorCitizen':1 if senior=='Yes' else 0,'Partner':cm['Partner'][partner],
                'Dependents':cm['Dependents'][dependents],'tenure':tenure,'Contract':cm['Contract'][contract],
                'PaperlessBilling':cm['PaperlessBilling'][paperless],'PaymentMethod':cm['PaymentMethod'][payment],
                'InternetService':cm['InternetService'][internet],'TechSupport':cm['TechSupport'][tech_support],
                'OnlineBackup':cm['OnlineBackup'][online_backup],'MonthlyCharges':monthly,'TotalCharges':total}])[c_feats]
            prob=best_c.predict_proba(row)[0][1]
            risk="High Risk" if prob>0.7 else "Medium Risk" if prob>0.4 else "Low Risk"
            rc="#ff4b4b" if prob>0.7 else "#f59e0b" if prob>0.4 else "#21c55d"
            st.markdown("---")
            r1,r2,r3=st.columns(3)
            with r1: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{rc}">{prob*100:.1f}%</div><div class="metric-lbl">Churn probability</div></div>',unsafe_allow_html=True)
            with r2: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{rc}">{"Will Churn" if prob>0.5 else "Will Stay"}</div><div class="metric-lbl">Prediction</div></div>',unsafe_allow_html=True)
            with r3: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{rc}">{risk}</div><div class="metric-lbl">Risk category</div></div>',unsafe_allow_html=True)
            exp2=shap.TreeExplainer(best_c); sv2=exp2.shap_values(row)[0]
            fig,_=plt.subplots(figsize=(8,4))
            shap.waterfall_plot(shap.Explanation(values=sv2,base_values=exp2.expected_value,data=row.values[0],feature_names=c_feats),show=False,max_display=10)
            st.pyplot(plot_dark(plt.gcf())); plt.close()
            st.markdown("**Recommendations:**")
            recs=[]
            if contract=="Month-to-month": recs.append("Offer a discounted annual contract")
            if monthly>80: recs.append("Review pricing — high monthly charges increase risk")
            if tenure<12: recs.append("Early-tenure customer — assign a retention specialist")
            if tech_support=="No": recs.append("Offer complimentary tech support onboarding")
            if not recs: recs.append("Customer is stable — maintain current engagement")
            for r in recs: st.markdown(f"- {r}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE B — COLLEGE ADMISSION
# ══════════════════════════════════════════════════════════════════════════════
else:

    if page == "🏠  Overview":
        st.markdown('<span class="module-badge-admit">College Admission Module</span>', unsafe_allow_html=True)
        st.title("College Student Admission Prediction")
        st.markdown("Predict admission outcomes using academic scores, research experience, and student profile features.")
        st.markdown("---")
        admit_rate=round(df_admit['Admitted'].mean()*100,1)
        c1,c2,c3,c4=st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><div class="metric-val-green">{len(df_admit):,}</div><div class="metric-lbl">Total applicants</div></div>',unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="metric-val-green">{admit_rate}%</div><div class="metric-lbl">Admission rate</div></div>',unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><div class="metric-val-green">{best_ar["auc"]}%</div><div class="metric-lbl">Best AUC-ROC</div></div>',unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card"><div class="metric-val-green">{best_ar["f1"]}%</div><div class="metric-lbl">Best F1 Score</div></div>',unsafe_allow_html=True)
        st.markdown("---")
        col1,col2=st.columns(2)
        with col1:
            st.markdown('<div class="section-header-green">Admission rate by university rating</div>', unsafe_allow_html=True)
            fig,ax=plt.subplots(figsize=(6,3.5))
            ur=df_admit.groupby('University_Rating')['Admitted'].mean()*100
            ax.bar([f"Rating {i}" for i in ur.index],ur.values,color=['#10b981','#34d399','#6ee7b7','#a7f3d0','#d1fae5'],width=0.6)
            ax.set_ylabel('Admission Rate (%)')
            for i,v in enumerate(ur.values): ax.text(i,v+0.5,f'{v:.1f}%',ha='center',color='#e2e8f0',fontsize=10)
            st.pyplot(plot_dark(fig)); plt.close()
        with col2:
            st.markdown('<div class="section-header-green">CGPA distribution by outcome</div>', unsafe_allow_html=True)
            fig,ax=plt.subplots(figsize=(6,3.5))
            ax.hist(df_admit[df_admit['Admitted']==0]['CGPA'],bins=25,alpha=0.7,color='#f59e0b',label='Rejected',density=True)
            ax.hist(df_admit[df_admit['Admitted']==1]['CGPA'],bins=25,alpha=0.7,color='#10b981',label='Admitted',density=True)
            ax.legend(facecolor='#141829',labelcolor='white'); ax.set_xlabel('CGPA')
            st.pyplot(plot_dark(fig)); plt.close()
        st.markdown("---")
        st.markdown('<div class="section-header-green">Key features for admission prediction</div>', unsafe_allow_html=True)
        features_info={"📐 GRE Score":"Graduate Record Examinations (260–340)","📝 TOEFL Score":"English proficiency (92–120)","🎓 CGPA":"Cumulative GPA out of 10","🔬 Research":"Research publications","🏛️ University Rating":"Institution rating (1–5)","📄 SOP / LOR":"Statement of Purpose & LOR strength","💼 Internships":"Internships completed","🏆 Projects":"Academic & personal projects"}
        cols=st.columns(4)
        for i,(k,v) in enumerate(features_info.items()):
            cols[i%4].markdown(f'<div class="metric-card"><div style="font-size:1rem;font-weight:600;color:#10b981">{k}</div><div class="metric-lbl">{v}</div></div>',unsafe_allow_html=True)

    elif page == "📊  Exploratory Analysis":
        st.markdown('<span class="module-badge-admit">College Admission Module</span>', unsafe_allow_html=True)
        st.title("Exploratory Data Analysis — Admissions")
        tab1,tab2,tab3=st.tabs(["Score Distributions","Correlations","Admission Drivers"])
        with tab1:
            col1,col2=st.columns(2)
            with col1:
                st.markdown('<div class="section-header-green">GRE score by outcome</div>', unsafe_allow_html=True)
                fig,ax=plt.subplots(figsize=(5,3))
                ax.hist(df_admit[df_admit['Admitted']==0]['GRE_Score'],bins=20,alpha=0.7,color='#f59e0b',label='Rejected',density=True)
                ax.hist(df_admit[df_admit['Admitted']==1]['GRE_Score'],bins=20,alpha=0.7,color='#10b981',label='Admitted',density=True)
                ax.legend(facecolor='#141829',labelcolor='white',fontsize=9); ax.set_xlabel('GRE Score')
                st.pyplot(plot_dark(fig)); plt.close()
            with col2:
                st.markdown('<div class="section-header-green">CGPA distribution</div>', unsafe_allow_html=True)
                fig,ax=plt.subplots(figsize=(5,3))
                ax.hist(df_admit[df_admit['Admitted']==0]['CGPA'],bins=20,alpha=0.7,color='#f59e0b',label='Rejected',density=True)
                ax.hist(df_admit[df_admit['Admitted']==1]['CGPA'],bins=20,alpha=0.7,color='#10b981',label='Admitted',density=True)
                ax.legend(facecolor='#141829',labelcolor='white',fontsize=9); ax.set_xlabel('CGPA')
                st.pyplot(plot_dark(fig)); plt.close()
            col3,col4=st.columns(2)
            with col3:
                st.markdown('<div class="section-header-green">Research vs admission</div>', unsafe_allow_html=True)
                fig,ax=plt.subplots(figsize=(5,3))
                res=df_admit.groupby('Research')['Admitted'].mean()*100
                ax.bar(['No Research','Has Research'],res.values,color=['#f59e0b','#10b981'],width=0.5)
                ax.set_ylabel('Admission Rate (%)')
                for i,v in enumerate(res.values): ax.text(i,v+0.5,f'{v:.1f}%',ha='center',color='#e2e8f0',fontsize=11)
                st.pyplot(plot_dark(fig)); plt.close()
            with col4:
                st.markdown('<div class="section-header-green">Admission by course type</div>', unsafe_allow_html=True)
                fig,ax=plt.subplots(figsize=(5,3))
                ct=df_admit.groupby('Course_Type')['Admitted'].mean()*100
                ax.barh(ct.index,ct.values,color='#10b981',height=0.5); ax.set_xlabel('Admission Rate (%)')
                for i,v in enumerate(ct.values): ax.text(v+0.3,i,f'{v:.1f}%',va='center',color='#e2e8f0',fontsize=9)
                st.pyplot(plot_dark(fig)); plt.close()
        with tab2:
            num_c=['GRE_Score','TOEFL_Score','CGPA','SOP','LOR','Research','Internships','Projects','Admitted']
            fig,ax=plt.subplots(figsize=(8,6))
            sns.heatmap(df_admit[num_c].corr(),annot=True,fmt='.2f',cmap='YlGn',ax=ax,linewidths=0.5,linecolor='#0a0d14',annot_kws={'size':9,'color':'white'})
            ax.set_xticklabels(ax.get_xticklabels(),rotation=35,ha='right',fontsize=9)
            st.pyplot(plot_dark(fig)); plt.close()
        with tab3:
            st.markdown('<div class="section-header-green">Top admission factors</div>', unsafe_allow_html=True)
            af={'CGPA > 9.0':78.4,'GRE > 325':71.2,'Has Research Paper':65.3,'TOEFL > 112':60.8,'Strong LOR (4.5+)':55.1,'Has Internship':48.7,'Alumni Relation':45.2,'No Backlog':42.6}
            fig,ax=plt.subplots(figsize=(8,4))
            keys=list(af.keys()); vals=list(af.values())
            colors=['#10b981' if v>65 else '#34d399' if v>50 else '#6ee7b7' for v in vals]
            ax.barh(keys,vals,color=colors,height=0.55); ax.set_xlabel('Admission Rate when factor present (%)')
            for i,v in enumerate(vals): ax.text(v+0.5,i,f'{v}%',va='center',color='#e2e8f0',fontsize=10)
            st.pyplot(plot_dark(fig)); plt.close()

    elif page == "🤖  Model Comparison":
        st.markdown('<span class="module-badge-admit">College Admission Module</span>', unsafe_allow_html=True)
        st.title("Model Comparison — Admissions")
        st.markdown("Four models trained and evaluated: LR, RF, XGBoost, Gradient Boosting.")
        mdf=pd.DataFrame({'Model':list(a_results.keys()),'Accuracy':[r['accuracy'] for r in a_results.values()],'Precision':[r['precision'] for r in a_results.values()],'Recall':[r['recall'] for r in a_results.values()],'F1 Score':[r['f1'] for r in a_results.values()],'AUC-ROC':[r['auc'] for r in a_results.values()]}).set_index('Model')
        st.dataframe(mdf.style.highlight_max(axis=0,color='#1a3a2a').format("{:.2f}"),use_container_width=True)
        col1,col2=st.columns(2)
        with col1:
            st.markdown('<div class="section-header-green">ROC curves — all models</div>', unsafe_allow_html=True)
            fig,ax=plt.subplots(figsize=(5,4))
            for (name,res),color in zip(a_results.items(),['#10b981','#34d399','#f59e0b','#7c83ff']):
                fpr,tpr,_=roc_curve(a_yte,res['proba']); ax.plot(fpr,tpr,label=f"{name} ({res['auc']}%)",color=color,lw=1.8)
            ax.plot([0,1],[0,1],'w--',lw=0.8,alpha=0.4); ax.legend(facecolor='#141829',labelcolor='white',fontsize=8)
            st.pyplot(plot_dark(fig)); plt.close()
        with col2:
            st.markdown('<div class="section-header-green">Confusion matrix (XGBoost)</div>', unsafe_allow_html=True)
            fig,ax=plt.subplots(figsize=(5,4))
            sns.heatmap(best_ar['cm'],annot=True,fmt='d',cmap='Greens',ax=ax,xticklabels=['Rejected','Admitted'],yticklabels=['Rejected','Admitted'],annot_kws={'size':16,'weight':'bold'})
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            st.pyplot(plot_dark(fig)); plt.close()
        st.markdown('<div class="section-header-green">Feature importance (XGBoost)</div>', unsafe_allow_html=True)
        fi=pd.DataFrame({'Feature':a_feats,'Importance':best_a.feature_importances_}).sort_values('Importance',ascending=True).tail(12)
        fig,ax=plt.subplots(figsize=(9,4))
        ax.barh(fi['Feature'],fi['Importance'],color='#10b981',height=0.6); ax.set_xlabel('Importance Score')
        st.pyplot(plot_dark(fig)); plt.close()

    elif page == "🔍  Explainability (SHAP)":
        st.markdown('<span class="module-badge-admit">College Admission Module</span>', unsafe_allow_html=True)
        st.title("Model Explainability with SHAP — Admissions")
        with st.spinner("Computing SHAP values for admission model…"):
            adm_exp=shap.TreeExplainer(best_a); X_as=a_Xte.iloc[:200]; asv=adm_exp.shap_values(X_as)
        col1,col2=st.columns(2)
        with col1:
            st.markdown('<div class="section-header-green">Global feature impact (beeswarm)</div>', unsafe_allow_html=True)
            fig,_=plt.subplots(figsize=(6,5)); shap.summary_plot(asv,X_as,plot_type="dot",show=False,max_display=12)
            st.pyplot(plot_dark(plt.gcf())); plt.close()
        with col2:
            st.markdown('<div class="section-header-green">Mean absolute SHAP values</div>', unsafe_allow_html=True)
            fig,_=plt.subplots(figsize=(6,5)); shap.summary_plot(asv,X_as,plot_type="bar",show=False,max_display=12)
            st.pyplot(plot_dark(plt.gcf())); plt.close()
        st.markdown('<div class="section-header-green">Individual applicant explanation</div>', unsafe_allow_html=True)
        idx=st.slider("Select an applicant from the test set",0,len(a_Xte)-1,0)
        row_s=a_Xte.iloc[[idx]]; sv_s=adm_exp.shap_values(row_s)[0]; prob_s=best_a.predict_proba(row_s)[0][1]
        st.markdown(f'Prediction: <span class="pill {"pill-admit" if prob_s>0.5 else "pill-reject"}">{"Likely Admitted" if prob_s>0.5 else "Likely Rejected"}</span> — confidence: **{prob_s*100:.1f}%**', unsafe_allow_html=True)
        fig,_=plt.subplots(figsize=(8,4))
        shap.waterfall_plot(shap.Explanation(values=sv_s,base_values=adm_exp.expected_value,data=row_s.values[0],feature_names=a_feats),show=False,max_display=12)
        st.pyplot(plot_dark(plt.gcf())); plt.close()

    elif page == "🎓  Predict an Applicant":
        st.markdown('<span class="module-badge-admit">College Admission Module</span>', unsafe_allow_html=True)
        st.title("Live Applicant Admission Prediction")
        st.markdown("Enter the student's profile to get an instant admission prediction with confidence score and recommendations.")
        with st.form("predict_admit"):
            st.markdown("#### 📐 Academic Scores")
            c1,c2,c3=st.columns(3)
            with c1:
                gre=st.slider("GRE Score",260,340,315); toefl=st.slider("TOEFL Score",92,120,107)
                cgpa=st.slider("CGPA (out of 10)",6.0,10.0,8.5,step=0.1)
            with c2:
                uni_rating=st.selectbox("University Rating",[1,2,3,4,5],index=2)
                sop=st.select_slider("SOP Strength",options=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],value=3.5)
                lor=st.select_slider("LOR Strength",options=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],value=3.5)
            with c3:
                course_type=st.selectbox("Course Type",['Engineering','Management','Science','Arts','Medicine'])
                research=st.selectbox("Research Paper Published?",["No","Yes"])
                scholarship=st.selectbox("Has Scholarship?",["No","Yes"])
            st.markdown("#### 💼 Experience & Profile")
            c4,c5,c6=st.columns(3)
            with c4:
                internships=st.slider("Number of Internships",0,3,1)
                projects=st.slider("Number of Projects",0,5,2)
            with c5:
                work_exp=st.slider("Work Experience (years)",0,4,0)
                backlog=st.selectbox("Backlogs (failed subjects)",[0,1,2,3])
            with c6:
                extracurricular=st.selectbox("Extracurricular Activities?",["No","Yes"])
                alumni_rel=st.selectbox("Alumni Relation?",["No","Yes"])
            submitted=st.form_submit_button("🎓 Predict Admission Chance")
        if submitted:
            ce={'Arts':0,'Engineering':1,'Management':2,'Medicine':3,'Science':4}
            row=pd.DataFrame([{'GRE_Score':gre,'TOEFL_Score':toefl,'University_Rating':uni_rating,
                'SOP':sop,'LOR':lor,'CGPA':cgpa,'Research':1 if research=="Yes" else 0,
                'Extracurricular':1 if extracurricular=="Yes" else 0,'Internships':internships,
                'Projects':projects,'Work_Experience':work_exp,'Backlog':backlog,
                'Course_Type':ce[course_type],'Scholarship':1 if scholarship=="Yes" else 0,
                'Alumni_Relation':1 if alumni_rel=="Yes" else 0}])[a_feats]
            prob=best_a.predict_proba(row)[0][1]
            pred=int(prob>0.5)
            chance="Excellent" if prob>0.80 else "Good" if prob>0.65 else "Moderate" if prob>0.50 else "Low" if prob>0.35 else "Very Low"
            cc="#10b981" if prob>0.65 else "#f59e0b" if prob>0.40 else "#ff4b4b"
            st.markdown("---")
            r1,r2,r3=st.columns(3)
            with r1: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{cc}">{prob*100:.1f}%</div><div class="metric-lbl">Admission probability</div></div>',unsafe_allow_html=True)
            with r2: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{cc}">{"Admitted ✓" if pred else "Rejected ✗"}</div><div class="metric-lbl">Predicted outcome</div></div>',unsafe_allow_html=True)
            with r3: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{cc}">{chance}</div><div class="metric-lbl">Admission chance</div></div>',unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<div class="section-header-green">What drove this prediction (SHAP)</div>', unsafe_allow_html=True)
            ae=shap.TreeExplainer(best_a); sv2=ae.shap_values(row)[0]
            fig,_=plt.subplots(figsize=(8,4))
            shap.waterfall_plot(shap.Explanation(values=sv2,base_values=ae.expected_value,data=row.values[0],feature_names=a_feats),show=False,max_display=12)
            st.pyplot(plot_dark(plt.gcf())); plt.close()
            st.markdown("**📋 Personalized Recommendations:**")
            recs=[]
            if gre<310: recs.append("🎯 Improve GRE score — aim for 315+ to significantly boost chances")
            if toefl<105: recs.append("📝 Retake TOEFL — target 110+ for competitive programs")
            if cgpa<8.0: recs.append("📚 Focus on CGPA — 8.5+ is preferred by top institutions")
            if research=="No": recs.append("🔬 Publish or assist in research — major differentiator for admissions")
            if internships==0: recs.append("💼 Complete at least 1 internship before applying")
            if backlog>0: recs.append("⚠️ Clear all backlogs — even 1 backlog significantly lowers chances")
            if sop<3.5: recs.append("📄 Strengthen your SOP — get it reviewed by mentors")
            if lor<3.5: recs.append("📬 Request LOR from professors who know your work closely")
            if not recs: recs.append("✅ Strong profile! Apply to top-tier programs with confidence.")
            for r in recs: st.markdown(f"- {r}")

    elif page == "📋  Diagrams & Architecture":
        st.markdown('<span class="module-badge-admit">College Admission Module</span>', unsafe_allow_html=True)
        st.title("System Architecture & Diagrams")
        st.markdown("ER Diagram, DFD Level 1, and Use Case Diagram — extended for the College Admission Prediction Module.")
        tab1,tab2,tab3=st.tabs(["🗂️ ER Diagram","🔄 DFD Level 1","👥 Use Case Diagram"])

        with tab1:
            st.markdown('<div class="section-header-green">Entity-Relationship Diagram — Admission System</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">The ER Diagram shows database entities and relationships for the College Admission Prediction System, extended from the original ChurnSense data model.</div>', unsafe_allow_html=True)
            st.markdown("""
**Entities & Attributes:**

| Entity | Primary Key | Key Attributes |
|--------|------------|----------------|
| **Student** | StudentID (PK) | Name, Email, DOB, Gender, Nationality |
| **AcademicProfile** | ProfileID (PK) | StudentID (FK), GRE_Score, TOEFL_Score, CGPA, Backlogs |
| **Application** | ApplicationID (PK) | StudentID (FK), CourseID (FK), CollegeID (FK), Status, AppliedDate |
| **Course** | CourseID (PK) | CourseName, Department, Duration, Tier |
| **College** | CollegeID (PK) | CollegeName, Location, Ranking, AcceptanceRate |
| **Admission_Prediction** | PredictionID (PK) | ApplicationID (FK), Model_Version, AdmissionScore, PredictionLabel, Timestamp |
| **Experience** | ExpID (PK) | StudentID (FK), Internships, Projects, WorkExperience, Research, Extracurricular |
| **Documents** | DocID (PK) | StudentID (FK), SOP_Score, LOR_Score, Scholarship, Alumni_Relation |

**Relationships:**
- Student **1:N** AcademicProfile — one student has multiple academic records over time
- Student **1:N** Application — one student can apply to multiple colleges/courses
- Student **1:1** Experience — each student has one experience profile
- Student **1:1** Documents — each student has one document strength record
- Application **1:1** Admission_Prediction — each application generates one prediction
- Course **1:N** Application — one course receives many applications
- College **1:N** Application — one college receives many applications
            """)

        with tab2:
            st.markdown('<div class="section-header-green">Data Flow Diagram (DFD Level 1) — Admission System</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">DFD Level 1 shows how data flows through the admission prediction pipeline, from raw student input to final decision output.</div>', unsafe_allow_html=True)
            col1,col2=st.columns(2)
            with col1:
                st.markdown("""
**🌐 External Entities:**
- 🎓 **Student** — Submits application, academic scores, documents
- 🏛️ **College Admin** — Views predictions, manages admission decisions
- 📊 **Data Scientist** — Trains/updates the prediction model
- 🔔 **Notification System** — Sends admission result emails

**⚙️ Processes:**
- **P1 — Data Collection:** Captures GRE, TOEFL, CGPA, SOP, LOR, experience details
- **P2 — Data Validation & Preprocessing:** Normalizes scores, encodes categoricals, handles missing values
- **P3 — Feature Engineering:** Computes derived features (score ratios, profile strength index)
- **P4 — Model Inference:** XGBoost generates admission probability score
- **P5 — Result Generation:** Produces prediction label + confidence + personalized recommendations
- **P6 — Notification & Storage:** Stores result in DB, triggers email notification
                """)
            with col2:
                st.markdown("""
**🗄️ Data Stores:**
- **D1** — Student Database (profiles, scores, documents)
- **D2** — Application Logs (submission history, status tracking)
- **D3** — Prediction Results Store (all prediction records)
- **D4** — Trained Model Store (versioned ML models)

**📊 Data Flows:**
- Student Data → D1 Customer DB → P1 Data Collection
- P1 → P2 Preprocessed Data → P3 Feature Engineering
- P3 → P4 Model Inference → P5 Result Generation
- P5 → D3 Prediction Results → College Admin
- P5 → P6 Notification → Student (email)
- Data Scientist → D4 Model Store → P4
                """)

        with tab3:
            st.markdown('<div class="section-header-green">Use Case Diagram — College Admission Prediction System</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">The Use Case Diagram identifies actors and their interactions with the College Admission Prediction System.</div>', unsafe_allow_html=True)
            col1,col2=st.columns(2)
            with col1:
                st.markdown("""
**👤 Student**
- Create Account / Register
- Submit Application Form
- Upload Academic Documents (SOP, LOR)
- Enter Scores (GRE, TOEFL, CGPA)
- View Admission Prediction & Probability
- Receive Personalized Recommendations
- Track Application Status

**🔧 System Administrator**
- Manage Users & Roles
- Monitor Model Performance
- Maintain Database & Backups
- Generate System Reports

**🧪 Data Scientist / ML Engineer**
- Upload New Training Data
- Retrain Prediction Model
- Evaluate Model Metrics (AUC, F1)
- Deploy Updated Model Version
                """)
            with col2:
                st.markdown("""
**🏛️ College Admission Officer**
- View Applicant Predictions Dashboard
- Analyze Full Applicant Pool
- Filter by Score Range / Course / Status
- Export Admission Reports (CSV/PDF)
- Send Offer / Rejection Letters

**📣 Marketing Manager**
- View Admission Analytics Dashboard
- Create Targeted Outreach Campaigns
- Track Enrollment Trends by Course/Region

**🤖 System (Automated)**
- Run Batch Predictions on New Applications
- Send Email Notifications to Students
- Schedule Periodic Model Retraining
- Log All Prediction History for Audit
                """)
            st.markdown('<div class="warn-box">💡 <b>Note:</b> This Use Case Diagram extends the original ChurnSense architecture to accommodate student-centric workflows including document management, score submission, and personalized recommendation generation.</div>', unsafe_allow_html=True)
