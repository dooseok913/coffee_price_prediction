import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(
    page_title="커피 가격 예측 시스템",
    page_icon="☕",
    layout="wide"
)

# 제목
st.title("☕ 커피 가격 상승/하락 예측 시스템")
st.markdown("---")

# ============================================
# 1. 모델 로드
# ============================================

@st.cache_resource
def load_models():
    try:
        # 여러 모델 중 하나 선택 (파일명에 맞게 수정)
        model_files = {
            'Random Forest': 'ml_2/random_forest_model.pkl',
            'Logistic Regression': 'ml_2/logistic_regression_model.pkl',
            'Gradient Boosting': 'ml_2/gradient_boosting_model.pkl',
            
            'KNN': 'ml_2/knn_model.pkl',
            'SVM': 'ml_2/svm_model.pkl'
        }
        
        models = {}
        for name, file in model_files.items():
            try:
                models[name] = joblib.load(file)
            except:
                pass
        
        scaler = joblib.load('ml_2/scaler.pkl')
        features = joblib.load('ml_2/features.pkl')
        
        return models, scaler, features
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, None, None

models, scaler, features = load_models()

if models is None:
    st.error("⚠️ 모델 파일을 찾을 수 없습니다. pkl 파일을 업로드해주세요.")
    st.stop()

# ============================================
# 2. 사이드바 - 모델 선택 및 입력
# ============================================

st.sidebar.header("📊 예측 설정")

# 모델 선택
selected_model_name = st.sidebar.selectbox(
    "사용할 모델 선택",
    list(models.keys())
)

selected_model = models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.header("📝 Feature 입력")

# Feature 입력 UI
input_data = {}

# 시차 변수
st.sidebar.subheader("1. 과거 가격 정보")
input_data['coffee_lag1'] = st.sidebar.number_input(
    "1개월 전 커피 가격 ($/lb)", 
    min_value=0.0, 
    max_value=10.0, 
    value=2.5, 
    step=0.01,
    help="1개월 전의 커피 선물 가격"
)

input_data['coffee_lag3'] = st.sidebar.number_input(
    "3개월 전 커피 가격 ($/lb)", 
    min_value=0.0, 
    max_value=10.0, 
    value=2.4, 
    step=0.01,
    help="3개월 전의 커피 선물 가격"
)

# 변화율
st.sidebar.subheader("2. 변화율 (%)")
input_data['coffee_pct_1m'] = st.sidebar.slider(
    "커피 가격 변화율 (1개월)", 
    min_value=-30.0, 
    max_value=30.0, 
    value=2.0, 
    step=0.1,
    help="전월 대비 가격 변화율"
)

input_data['usd_brl_pct_1m'] = st.sidebar.slider(
    "USD/BRL 환율 변화율 (1개월)", 
    min_value=-15.0, 
    max_value=15.0, 
    value=1.0, 
    step=0.1,
    help="브라질 헤알 환율 변화율"
)

input_data['wti_pct_1m'] = st.sidebar.slider(
    "WTI 유가 변화율 (1개월)", 
    min_value=-30.0, 
    max_value=30.0, 
    value=0.5, 
    step=0.1,
    help="유가 변화율 (물류비 영향)"
)

# 이동평균 대비
st.sidebar.subheader("3. 기술적 지표")
input_data['price_vs_ma3'] = st.sidebar.slider(
    "3개월 이동평균 대비 (%)", 
    min_value=-20.0, 
    max_value=20.0, 
    value=3.0, 
    step=0.1,
    help="현재가가 3개월 이동평균보다 높으면 양수"
)

# 기후 변수
st.sidebar.subheader("4. 기후 요인")
input_data['heatwave_lag3'] = st.sidebar.selectbox(
    "3개월 전 폭염 발생 여부",
    [0, 1],
    format_func=lambda x: "발생" if x == 1 else "미발생",
    help="브라질 주요 산지의 폭염 (연속 2일 이상 30°C 초과)"
)

input_data['precip_lag1'] = st.sidebar.number_input(
    "1개월 전 강수량 (mm)", 
    min_value=0.0, 
    max_value=500.0, 
    value=120.0, 
    step=1.0,
    help="브라질 주요 산지의 월 강수량"
)

# 엘니뇨
input_data['elnino_lag1'] = st.sidebar.slider(
    "1개월 전 엘니뇨 지수 (ONI)", 
    min_value=-3.0, 
    max_value=3.0, 
    value=0.0, 
    step=0.1,
    help="양수: 엘니뇨 / 음수: 라니냐"
)

# 계절성
st.sidebar.subheader("5. 계절 정보")
input_data['month'] = st.sidebar.selectbox(
    "예측 대상 월",
    list(range(1, 13)),
    index=datetime.now().month - 1,
    format_func=lambda x: f"{x}월"
)

# ============================================
# 3. 예측 실행
# ============================================

if st.sidebar.button("🔮 예측하기", type="primary"):
    
    # 입력 데이터 DataFrame 변환
    input_df = pd.DataFrame([input_data])
    input_df = input_df[features]  # Feature 순서 맞추기
    
    # 스케일링 (필요 시)
    if selected_model_name in ['Logistic Regression', 'SVM', 'KNN']:
        input_scaled = scaler.transform(input_df)
        prediction = selected_model.predict(input_scaled)[0]
        probability = selected_model.predict_proba(input_scaled)[0]
    else:
        prediction = selected_model.predict(input_df)[0]
        probability = selected_model.predict_proba(input_df)[0]
    
    prob_down = probability[0]
    prob_up = probability[1]
    
    # ============================================
    # 4. 결과 표시
    # ============================================
    
    st.markdown("## 📈 예측 결과")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="예측 방향",
            value="📈 상승" if prediction == 1 else "📉 하락",
            delta=f"{abs(prob_up - prob_down)*100:.1f}% 확신"
        )
    
    with col2:
        st.metric(
            label="상승 확률",
            value=f"{prob_up*100:.1f}%",
            delta=f"{(prob_up - 0.5)*100:+.1f}%p"
        )
    
    with col3:
        st.metric(
            label="하락 확률",
            value=f"{prob_down*100:.1f}%",
            delta=f"{(prob_down - 0.5)*100:+.1f}%p"
        )
    
    # 확률 시각화
    st.markdown("### 📊 확률 분포")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    bars = ax.barh(['하락', '상승'], [prob_down, prob_up], 
                   color=['#ff6b6b', '#51cf66'])
    
    # 바 위에 값 표시
    for bar, prob in zip(bars, [prob_down, prob_up]):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%',
                ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('확률', fontsize=12)
    ax.set_title(f'{selected_model_name} 예측 결과', fontsize=14, fontweight='bold')
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    st.pyplot(fig)
    
    # ============================================
    # 5. 해석 및 권장사항
    # ============================================
    
    st.markdown("### 💡 해석 및 권장사항")
    
    if prob_up > 0.6:
        st.success(f"""
        **강한 상승 신호** (확률: {prob_up*100:.1f}%)
        - 다음 달 커피 가격이 상승할 가능성이 높습니다.
        - 현물 구매 시기를 앞당기는 것을 고려하세요.
        - 선물 매수 포지션 검토를 권장합니다.
        """)
    elif prob_up > 0.5:
        st.info(f"""
        **약한 상승 신호** (확률: {prob_up*100:.1f}%)
        - 소폭 상승 가능성이 있으나 불확실성이 있습니다.
        - 추가 지표를 모니터링하며 신중히 판단하세요.
        """)
    elif prob_down > 0.6:
        st.warning(f"""
        **강한 하락 신호** (확률: {prob_down*100:.1f}%)
        - 다음 달 커피 가격이 하락할 가능성이 높습니다.
        - 현물 구매를 늦추는 것을 고려하세요.
        - 재고 조정 전략 검토를 권장합니다.
        """)
    else:
        st.info(f"""
        **약한 하락 신호** (확률: {prob_down*100:.1f}%)
        - 소폭 하락 가능성이 있으나 불확실성이 있습니다.
        - 시장 상황을 주시하며 대응하세요.
        """)
    
    # ============================================
    # 6. 주요 영향 요인 분석
    # ============================================
    
    st.markdown("### 🔍 주요 영향 요인")
    
    # Feature Importance (Random Forest가 있는 경우)
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        importances = pd.DataFrame({
            'Feature': features,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importances['Feature'], importances['Importance'], color='steelblue')
        ax.set_xlabel('중요도', fontsize=12)
        ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        st.pyplot(fig)
    
    # ============================================
    # 7. 입력 데이터 요약
    # ============================================
    
    with st.expander("📋 입력 데이터 요약 보기"):
        st.dataframe(input_df.T, use_container_width=True)

# ============================================
# 8. 정보 탭
# ============================================

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📖 Feature 설명", "📊 모델 정보", "ℹ️ 사용 가이드"])

with tab1:
    st.markdown("""
    ## Feature 설명
    
    ### 1️⃣ 과거 가격 정보
    - **coffee_lag1**: 1개월 전 커피 가격 - 단기 추세 파악
    - **coffee_lag3**: 3개월 전 커피 가격 - 중기 추세 파악
    
    ### 2️⃣ 변화율
    - **coffee_pct_1m**: 커피 가격 변화율 - 모멘텀 지표
    - **usd_brl_pct_1m**: 환율 변화율 - 브라질 수출 경쟁력 영향
    - **wti_pct_1m**: 유가 변화율 - 물류비 및 생산비 영향
    
    ### 3️⃣ 기술적 지표
    - **price_vs_ma3**: 3개월 이동평균 대비 - 과매수/과매도 판단
    
    ### 4️⃣ 기후 요인
    - **heatwave_lag3**: 3개월 전 폭염 - 생육에 영향 (시차 반영)
    - **precip_lag1**: 1개월 전 강수량 - 개화/결실기 영향
    
    ### 5️⃣ 글로벌 기후
    - **elnino_lag1**: 엘니뇨 지수 - 장기 기후 패턴
    
    ### 6️⃣ 계절성
    - **month**: 월 - 수확기/비수확기 영향
    """)

with tab2:
    st.markdown(f"""
    ## 모델 정보
    
    ### 현재 사용 중인 모델
    **{selected_model_name}**
    
    ### 학습 데이터
    - 기간: 2015년 1월 ~ 2025년 12월
    - 데이터 출처: FRED, NOAA, Meteostat
    
    ### 평가 지표
    - Accuracy: 모델의 전체 정확도
    - ROC-AUC: 예측 능력 종합 평가 (0.5=무작위, 1.0=완벽)
    
    ### 주의사항
    ⚠️ 이 모델은 과거 데이터 기반 예측이며, 실제 투자 결정 시 참고용으로만 사용하세요.
    """)

with tab3:
    st.markdown("""
    ## 사용 가이드
    
    ### 1. Feature 입력
    좌측 사이드바에서 각 Feature 값을 입력합니다.
    
    ### 2. 예측 실행
    "🔮 예측하기" 버튼을 클릭합니다.
    
    ### 3. 결과 해석
    - 상승/하락 확률을 확인합니다.
    - 확률이 60% 이상일 때 신뢰도가 높습니다.
    
    ### 4. 의사결정
    - 예측 결과와 함께 시장 상황을 종합적으로 판단하세요.
    - 리스크 관리를 위해 다른 지표도 함께 고려하세요.
    
    ### 📞 문의
    모델 개선 제안이나 버그 리포트는 이슈 트래커에 등록해주세요.
    """)

# ============================================
# 9. 푸터
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>☕ Coffee Price Prediction System v1.0 | Powered by Machine Learning</p>
    <p style='font-size: 12px; color: gray;'>
        데이터 출처: FRED, NOAA, Meteostat | 
        모델: Scikit-learn
    </p>
</div>
""", unsafe_allow_html=True)