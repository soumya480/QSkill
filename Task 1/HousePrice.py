import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.datasets import fetch_california_housing # Import the dataset loader
import warnings
import os # Import the os module
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────
print("=" * 60)
print(" HOUSE PRICE PREDICTION — LINEAR REGRESSION PIPELINE")
print("=" * 60)

# Load the California Housing dataset from sklearn
housing = fetch_california_housing(as_frame=True)
df = housing.frame
df["Price"] = housing.target # Add the target variable to the DataFrame

print(f"\n[1] Dataset loaded: {df.shape[0]:,} rows w {df.shape[1]} columns")
print(f"    Missing values : {df.isnull().sum().sum()}")
print(f"\n    Feature statistics:")
print(df.describe().T[["mean","std","min","max"]].round(3).to_string())

# ─────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────
print("\n[2] Feature Engineering & Preprocessing...")

# Renaming columns to match original notebook's expected names
df = df.rename(columns={
    'MedInc': 'MedianIncome',
    'HouseAge': 'HouseAge',
    'AveRooms': 'AvgRooms',
    'AveBedrms': 'AvgBedrooms',
    'Population': 'Population',
    'AveOccup': 'AvgOccupancy',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude'
})

# Outlier capping at 1st/99th pct
for col in ["AvgRooms","AvgBedrooms","AvgOccupancy","Population"]:
    lo,hi = df[col].quantile([0.01,0.99])
    df[col] = df[col].clip(lo,hi)

# Derived features
df["RoomsPerOccupant"]  = df["AvgRooms"]  / df["AvgOccupancy"]
df["BedroomRatio"]      = df["AvgBedrooms"] / df["AvgRooms"]
df["PopDensity"]        = df["Population"] / df["AvgOccupancy"]
df["IncomePerRoom"]     = df["MedianIncome"] / df["AvgRooms"]
df["AgeIncome"]         = df["HouseAge"] * df["MedianIncome"]  # interaction

print("    Engineered features: RoomsPerOccupant, BedroomRatio,")
print("                         PopDensity, IncomePerRoom, AgeIncome")

# ─────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────
FEATURES = [
    "MedianIncome","HouseAge","AvgRooms","AvgBedrooms",
    "Population","AvgOccupancy","Latitude","Longitude",
    "RoomsPerOccupant","BedroomRatio","PopDensity",
    "IncomePerRoom","AgeIncome"
]
X = df[FEATURES]; y = df["Price"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(f"\n[3] Train/Test Split → Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────────
print("\n[4] Training Models...")

models = {
    "Linear Regression":      Pipeline([("sc",StandardScaler()),("m",LinearRegression())]),
    "Ridge  (α=1.0)":         Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=1.0))]),
    "Lasso  (α=0.001)":       Pipeline([("sc",StandardScaler()),("m",Lasso(alpha=0.001,max_iter=10000))]),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name,pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    mae  = mean_absolute_error(y_test,y_pred)
    r2   = r2_score(y_test,y_pred)
    cv   = cross_val_score(pipe,X_train,y_train,cv=kf,scoring="r2").mean()
    results[name] = dict(pipe=pipe,y_pred=y_pred,RMSE=rmse,MAE=mae,R2=r2,CV=cv)
    print(f"   {name:<30}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  CV-R²={cv:.4f}")

best_name = max(results,key=lambda k:results[k]["R2"])
best = results[best_name]
print(f"\n   ✓ Best model: {best_name}")

# Feature Importance (permutation)
perm = permutation_importance(best["pipe"],X_test,y_test,n_repeats=10,random_state=42,scoring="r2")
imp_df = pd.DataFrame({"Feature":FEATURES,"Importance":perm.importances_mean}).sort_values("Importance",ascending=False)

# ─────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────
print("\n[5] Generating visualisations...")

plt.style.use("seaborn-v0_8-whitegrid")
BLUE="#2563EB"; GREEN="#16A34A"; RED="#DC2626"; AMBER="#D97706"; GREY="#94A3B8"

fig = plt.figure(figsize=(20,26))
fig.patch.set_facecolor("#F1F5F9")
gs  = gridspec.GridSpec(4,2,figure=fig,hspace=0.48,wspace=0.35)

# A — Correlation Heatmap
ax0 = fig.add_subplot(gs[0,:])
arr_columns_in_df = [col for col in FEATURES+["Price"] if col in df.columns] # Filter out non-existent columns
corr = df[arr_columns_in_df].corr()
mask = np.triu(np.ones_like(corr,dtype=bool))
sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",cmap="RdYlGn",
            center=0,linewidths=0.4,ax=ax0,annot_kws={"size":7.5})
ax0.set_title("Feature Correlation Matrix",fontsize=14,fontweight="bold",pad=12)
ax0.tick_params(labelsize=8); ax0.set_xticklabels(ax0.get_xticklabels(),rotation=30,ha="right")

# B — Actual vs Predicted
ax1 = fig.add_subplot(gs[1,0])
ax1.scatter(y_test,best["y_pred"],alpha=0.25,s=8,color=BLUE)
lims=[min(y_test.min(),best["y_pred"].min()),max(y_test.max(),best["y_pred"].max())]
ax1.plot(lims,lims,"r--",lw=1.5,label="Perfect fit")
ax1.set_xlabel("Actual Price ($100K)",fontsize=10); ax1.set_ylabel("Predicted ($100K)",fontsize=10)
ax1.set_title(f"Actual vs Predicted\n{best_name}",fontsize=12,fontweight="bold")
ax1.legend(fontsize=9)
ax1.text(0.05,0.92,f"R² = {best['R2']:.4f}",transform=ax1.transAxes,fontsize=10,color=GREEN,fontweight="bold")

# C — Residual Histogram
ax2 = fig.add_subplot(gs[1,1])
residuals = y_test.values - best["y_pred"]
ax2.hist(residuals,bins=60,color=BLUE,edgecolor="white",alpha=0.8)
ax2.axvline(0,color=RED,linestyle="--",lw=1.5)
ax2.set_xlabel("Residuals ($100K)",fontsize=10); ax2.set_ylabel("Count",fontsize=10)
ax2.set_title("Residual Distribution",fontsize=12,fontweight="bold")
ax2.text(0.05,0.92,f"Mean={residuals.mean():.3f}  Std={residuals.std():.3f}",
         transform=ax2.transAxes,fontsize=9,color=GREY)

# D — Feature Importance
ax3 = fig.add_subplot(gs[2,0])
cols=[GREEN if v>=0 else RED for v in imp_df["Importance"]]
ax3.barh(imp_df["Feature"],imp_df["Importance"],color=cols)
ax3.set_xlabel("Permutation Importance (R² drop)",fontsize=10)
ax3.set_title("Feature Importance\n(Permutation Method)",fontsize=12,fontweight="bold")
ax3.invert_yaxis(); ax3.axvline(0,color=GREY,lw=0.8)

# E — Model Comparison
ax4 = fig.add_subplot(gs[2,1])
short_names=["Linear\nRegression","Ridge\n(α=1)","Lasso\n(α=0.001)"]
r2s  =[results[n]["R2"]   for n in results]
rmses=[results[n]["RMSE"] for n in results]
x=np.arange(3); w=0.35
b1=ax4.bar(x-w/2,r2s,  w,label="R²",  color=BLUE, alpha=0.85)
b2=ax4.bar(x+w/2,rmses,w,label="RMSE",color=AMBER,alpha=0.85)
ax4.set_xticks(x); ax4.set_xticklabels(short_names,fontsize=9)
ax4.set_ylabel("Score",fontsize=10); ax4.set_title("Model Comparison (R² & RMSE)",fontsize=12,fontweight="bold")
ax4.legend(fontsize=9)
for b in list(b1)+list(b2):
    ax4.text(b.get_x()+b.get_width()/2,b.get_height()+0.003,
             f"{b.get_height():.3f}",ha="center",va="bottom",fontsize=8)

# F — Residuals vs Fitted
ax5 = fig.add_subplot(gs[3,0])
ax5.scatter(best["y_pred"],residuals,alpha=0.2,s=7,color=BLUE)
ax5.axhline(0,color=RED,linestyle="--",lw=1.5)
ax5.set_xlabel("Fitted Values ($100K)",fontsize=10); ax5.set_ylabel("Residuals",fontsize=10)
ax5.set_title("Residuals vs Fitted",fontsize=12,fontweight="bold")

# G — Price Distribution
ax6 = fig.add_subplot(gs[3,1])
ax6.hist(y_train,bins=50,color=BLUE, alpha=0.6,label="Train",edgecolor="white")
ax6.hist(y_test, bins=50,color=GREEN,alpha=0.6,label="Test", edgecolor="white")
ax6.set_xlabel("Price ($100K)",fontsize=10); ax6.set_ylabel("Count",fontsize=10)
ax6.set_title("Target Distribution\nTrain vs Test",fontsize=12,fontweight="bold")
ax6.legend(fontsize=9)

fig.suptitle("🏠  House Price Prediction — Linear Regression Analysis",
             fontsize=16,fontweight="bold",y=0.99,color="#1E293B")

out_path="/mnt/user-data/outputs/house_price_analysis.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True) # Create the output directory if it doesn't exist
plt.savefig(out_path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
plt.close()
print(f"   Saved → {out_path}")

# ─────────────────────────────────────────────────
# 6. FINAL REPORT
# ─────────────────────────────────────────────────
print("\n" + "="*60)
print(" FINAL PERFORMANCE REPORT")
print("="*60)
print(f"\n  Best Model    : {best_name}")
print(f"  R² Score      : {best['R2']:.4f}  ({best['R2']*100:.1f}% variance explained)")
print(f"  RMSE          : ${best['RMSE']*100_000:,.0f}")
print(f"  MAE           : ${best['MAE']*100_000:,.0f}")
print(f"  CV R² (5-fold): {best['CV']:.4f}")
print(f"\n  Top 5 Features by Importance:")
for _,row in imp_df.head(5).iterrows():
    bar = "█"*int(max(row["Importance"]/imp_df["Importance"].max()*20,1))
    print(f"    {bar:<20}  {row['Feature']:<22}  {row['Importance']:.4f}")

print("\n  Sample Predictions (8 test houses):")
print(f"  {'#':<3} {'Actual':>12} {'Predicted':>12} {'Error':>10}")
print("  " + "─"*42)
for i,(a,p) in enumerate(zip(y_test.values[:8],best["y_pred"][:8])):
    err=(p-a)*100_000
    print(f"  {i+1:<3} ${a*100_000:>10,.0f}  ${p*100_000:>10,.0f}  {err:>+9,.0f}")

print("\n✓ Pipeline complete.\n")
