import pandas as pd
import numpy as np
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from meteostat import Point, Daily, Monthly
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. 데이터 수집 
# ============================================

FRED_API_KEY = "46e4ba9f721e9738794073c8fa787d2d"

def get_fred_data(series_id, start_date='2015-01-01', end_date='2025-12-31'):
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date
    }
    response = requests.get(url, params=params)
    data = response.json()['observations']
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df[['date', 'value']]

# 커피 가격
coffee_df = get_fred_data('PCOFFOTMUSDM')
coffee_df.columns = ['date', 'coffee_price']

# USD/BRL 환율
usd_brl_df = get_fred_data('DEXBZUS')
usd_brl_df.columns = ['date', 'usd_brl']

# WTI 유가
wti_df = get_fred_data('DCOILWTICO')
wti_df.columns = ['date', 'wti_price']

# 엘니뇨 지수
oni_url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
oni_df = pd.read_csv(oni_url, sep=r'\s+', engine='python')
start_oni = datetime(1950, 1, 1)
oni_df['date'] = [start_oni + pd.DateOffset(months=i) for i in range(len(oni_df))]
oni_df = oni_df[(oni_df['date'] >= '2015-01-01') & (oni_df['date'] <= '2025-12-31')]
oni_df = oni_df[['date', 'ANOM']].rename(columns={'ANOM': 'el_nino_index'})

# 기후 데이터 (브라질)
location = Point(-21.55, -45.43)
start = datetime(2015, 1, 1)
end = datetime(2025, 12, 31)

weather_monthly = Monthly(location, start, end).fetch().reset_index()
weather_monthly = weather_monthly[['time', 'prcp']].rename(
    columns={'time': 'date', 'prcp': 'br_precip'}
)

# 폭염 데이터
daily_weather = Daily(location, start, end).fetch().reset_index()
HEAT_THRESHOLD = 30.0
daily_weather['hot_day'] = daily_weather['tmax'] >= HEAT_THRESHOLD
daily_weather['hot_group'] = (
    daily_weather['hot_day'] != daily_weather['hot_day'].shift()
).cumsum()

heatwave_runs = daily_weather[daily_weather['hot_day']].groupby('hot_group').size()
heatwave_groups = heatwave_runs[heatwave_runs >= 2].index
daily_weather['heatwave_2d'] = daily_weather['hot_group'].isin(heatwave_groups).astype(int)

daily_weather['year_month'] = daily_weather['time'].dt.to_period('M')
monthly_events = daily_weather.groupby('year_month').agg({'heatwave_2d': 'max'}).reset_index()
monthly_events['date'] = monthly_events['year_month'].dt.to_timestamp()
monthly_events = monthly_events.rename(columns={'heatwave_2d': 'heatwave_2d_month'})

# ============================================
# 2. 데이터 병합
# ============================================

df = coffee_df.merge(usd_brl_df, on='date', how='left')
df = df.merge(wti_df, on='date', how='left')
df = df.merge(oni_df, on='date', how='left')
df = df.merge(weather_monthly, on='date', how='left')
df = df.merge(monthly_events[['date', 'heatwave_2d_month']], on='date', how='left')

# 결측치 처리
df = df.sort_values('date').reset_index(drop=True)
df[['usd_brl', 'wti_price', 'el_nino_index', 'br_precip', 'heatwave_2d_month']] = (
    df[['usd_brl', 'wti_price', 'el_nino_index', 'br_precip', 'heatwave_2d_month']]
    .fillna(method='ffill').fillna(method='bfill')
)

print(f"총 데이터 개수: {len(df)}")
print(f"기간: {df['date'].min()} ~ {df['date'].max()}")

# ============================================
# 3. Feature Engineering (개선!)
# ============================================

# 3-1. 시차 변수 (1, 3, 6개월)
df['coffee_lag1'] = df['coffee_price'].shift(1)
df['coffee_lag3'] = df['coffee_price'].shift(3)
df['coffee_lag6'] = df['coffee_price'].shift(6)

# 3-2. 변화율 (1개월 전 대비)
df['coffee_pct_1m'] = df['coffee_price'].pct_change(1) * 100
df['usd_brl_pct_1m'] = df['usd_brl'].pct_change(1) * 100
df['wti_pct_1m'] = df['wti_price'].pct_change(1) * 100

# 3-3. 이동평균 (3개월, 6개월)
df['coffee_ma3'] = df['coffee_price'].rolling(window=3, min_periods=1).mean()
df['coffee_ma6'] = df['coffee_price'].rolling(window=6, min_periods=1).mean()

# 3-4. 기후 시차 (폭염은 3개월 전, 강수량은 1개월 전)
df['heatwave_lag3'] = df['heatwave_2d_month'].shift(3)
df['precip_lag1'] = df['br_precip'].shift(1)
df['precip_lag3'] = df['br_precip'].shift(3)

# 3-5. 엘니뇨 시차 (1개월 전)
df['elnino_lag1'] = df['el_nino_index'].shift(1)

# 3-6. 계절성 (월)
df['month'] = df['date'].dt.month

# 3-7. 이동평균 대비 현재가 비율
df['price_vs_ma3'] = (df['coffee_price'] / df['coffee_ma3'] - 1) * 100

# ============================================
# 4. 타겟 변수 생성 (명확하게!)
# ============================================

# 다음 달 가격이 오르면 1, 아니면 0
df['price_next_month'] = df['coffee_price'].shift(-1)
df['price_up'] = (df['price_next_month'] > df['coffee_price']).astype(int)

# 마지막 행 제거 (미래값 없음)
df = df[:-1].copy()

# 결측치 제거 (시차 변수로 인한)
df = df.dropna().reset_index(drop=True)

print(f"\n최종 데이터: {len(df)}개")
print(f"상승 비율: {df['price_up'].mean():.2%}")
print(f"하락 비율: {(1-df['price_up'].mean()):.2%}")

# ============================================
# 5. Feature 선택 (개선된 버전)
# ============================================

features = [
    # 시차 변수
    'coffee_lag1',
    'coffee_lag3',
    
    # 변화율
    'coffee_pct_1m',
    'usd_brl_pct_1m',
    'wti_pct_1m',
    
    # 이동평균 대비
    'price_vs_ma3',
    
    # 기후 (시차 적용)
    'heatwave_lag3',
    'precip_lag1',
    
    # 엘니뇨 (시차)
    'elnino_lag1',
    
    # 계절성
    'month'
]

X = df[features]
y = df['price_up']

print("\n사용할 Feature:")
for i, f in enumerate(features, 1):
    print(f"{i}. {f}")

# ============================================
# 6. 상관관계 분석
# ============================================

corr_df = X.copy()
corr_df['price_up'] = y

corr_matrix = corr_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap", fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('01_correlation_heatmap.png', dpi=300)
plt.show()

print("\nprice_up과의 상관계수:")
print(corr_matrix['price_up'].sort_values(ascending=False))

# ============================================
# 7. 데이터 분할 (시계열 유지!)
# ============================================

from sklearn.model_selection import train_test_split

# 시계열 데이터는 shuffle=False!
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"\n학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")

# ============================================
# 8. 스케일링
# ============================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 9. 모델 학습 및 비교
# ============================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7)
}

results = []

for name, model in models.items():
    # 스케일링 필요 여부
    if name in ['Logistic Regression', 'SVM', 'KNN']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
print("\n모델 성능 비교:")
print(results_df.to_string(index=False))

# ============================================
# 10. 최고 모델 선택 및 상세 분석
# ============================================

best_model_name = results_df.iloc[0]['Model']
print(f"\n✅ 최고 성능 모델: {best_model_name}")

# 최고 모델 재학습
best_model = models[best_model_name]

if best_model_name in ['Logistic Regression', 'SVM', 'KNN']:
    best_model.fit(X_train_scaled, y_train)
    y_pred_best = best_model.predict(X_test_scaled)
    y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    y_prob_best = best_model.predict_proba(X_test)[:, 1]

# ============================================
# 11. 시각화
# ============================================

# 11-1. 모델 성능 비교 바 차트
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']
for idx, metric in enumerate(metrics):
    ax = axes[idx //2, idx % 2]
    results_df_sorted = results_df.sort_values(metric)
    ax.barh(results_df_sorted['Model'], results_df_sorted[metric])
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'{metric} by Model', fontsize=14, weight='bold')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('02_model_comparison.png', dpi=300)
plt.show()

# 11-2. Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"])

plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}", fontsize=14, weight='bold')
plt.savefig('03_confusion_matrix.png', dpi=300)
plt.show()

# 11-3. ROC Curve 비교
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))

for name, model in models.items():
    if name in ['Logistic Regression', 'SVM', 'KNN']:
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison', fontsize=16, weight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('04_roc_curve_comparison.png', dpi=300)
plt.show()

# 11-4. Feature Importance (Random Forest 사용)
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=features).sort_values()

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='steelblue')
plt.title('Feature Importance (Random Forest)', fontsize=16, weight='bold')
plt.xlabel('Importance', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('05_feature_importance.png', dpi=300)
plt.show()

# 11-5. Logistic Regression 계수 (있는 경우)
if best_model_name == 'Logistic Regression':
    coef_df = pd.DataFrame({
        'feature': features,
        'coef': best_model.coef_[0]
    }).sort_values('coef')
    
    plt.figure(figsize=(10, 6))
    plt.barh(coef_df['feature'], coef_df['coef'])
    plt.axvline(0, color='black', linewidth=1)
    plt.title('Logistic Regression Coefficients', fontsize=16, weight='bold')
    plt.xlabel('Coefficient', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('06_logistic_coefficients.png', dpi=300)
    plt.show()

# 11-6. 예측 확률 분포
plt.figure(figsize=(10, 5))
plt.hist(y_prob_best[y_test==0], bins=20, alpha=0.5, label='Down (Actual)', color='red')
plt.hist(y_prob_best[y_test==1], bins=20, alpha=0.5, label='Up (Actual)', color='green')
plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Predicted Probability (Up)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'Prediction Probability Distribution - {best_model_name}', fontsize=14, weight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('07_probability_distribution.png', dpi=300)
plt.show()

# ============================================
# 12. 모델 저장 (전체 모델 저장)
# ============================================

import joblib

# 모든 모델 저장
for name, model in models.items():
    filename = f"{name.replace(' ', '_').lower()}_model.pkl"
    
    # 스케일링 필요 여부에 따라 재학습 후 저장
    if name in ['Logistic Regression', 'SVM', 'KNN']:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    
    joblib.dump(model, filename)
    print(f"✅ {filename} 저장 완료")

# Scaler와 Features 저장
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "features.pkl")

print(f"\n✅ 전체 저장 완료:")
print(f"   - 모델 5개")
print(f"   - scaler.pkl")
print(f"   - features.pkl")

# ============================================
# 13. 최종 결과 CSV 저장
# ============================================

test_results = df.iloc[split_idx:].copy()
test_results['predicted'] = y_pred_best
test_results['probability_up'] = y_prob_best

test_results[['date', 'coffee_price', 'price_up', 'predicted', 'probability_up']].to_csv(
    'test_predictions.csv', index=False
)

print("\n✅ 예측 결과 저장: test_predictions.csv")