import streamlit as st
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="FoodLens - AI 음식 영양 분석",
    page_icon="🍽️",
    layout="centered",
)

# 과제용 시제품 영양 DB
# 실제 서비스에서는 식품영양 DB/API와 연결하는 방식으로 확장할 수 있습니다.
NUTRITION_DB = {
    "pizza": {
        "name": "피자",
        "serving": "1조각 (약 100g)",
        "calories": 266, "carbs": 33.0, "protein": 11.0,
        "fat": 10.0, "sodium": 598, "sugar": 3.6
    },
    "hamburger": {
        "name": "햄버거",
        "serving": "1개 (약 200g)",
        "calories": 500, "carbs": 45.0, "protein": 25.0,
        "fat": 25.0, "sodium": 900, "sugar": 8.0
    },
    "ramen": {
        "name": "라면",
        "serving": "1봉지",
        "calories": 500, "carbs": 65.0, "protein": 10.0,
        "fat": 16.0, "sodium": 1800, "sugar": 3.0
    },
    "fried_rice": {
        "name": "볶음밥",
        "serving": "1인분 (약 300g)",
        "calories": 450, "carbs": 70.0, "protein": 12.0,
        "fat": 14.0, "sodium": 850, "sugar": 3.0
    },
    "sushi": {
        "name": "초밥",
        "serving": "약 8개",
        "calories": 400, "carbs": 60.0, "protein": 18.0,
        "fat": 8.0, "sodium": 700, "sugar": 8.0
    },
    "salad": {
        "name": "샐러드",
        "serving": "1그릇 (약 250g)",
        "calories": 180, "carbs": 15.0, "protein": 8.0,
        "fat": 9.0, "sodium": 350, "sugar": 5.0
    },
    "steak": {
        "name": "스테이크",
        "serving": "약 200g",
        "calories": 500, "carbs": 0.0, "protein": 52.0,
        "fat": 32.0, "sodium": 450, "sugar": 0.0
    },
}

# Food-101의 실제 영문 라벨 일부를 시제품 DB 키에 연결
LABEL_MAP = {
    "pizza": "pizza",
    "hamburger": "hamburger",
    "ramen": "ramen",
    "fried_rice": "fried_rice",
    "sushi": "sushi",
    "caesar_salad": "salad",
    "greek_salad": "salad",
    "steak": "steak",
    "beef_tartare": "steak",
}

def normalize_label(label: str):
    label = label.lower().strip().replace(" ", "_")
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    # 부분 일치
    for key in LABEL_MAP:
        if key in label or label in key:
            return LABEL_MAP[key]
    return None

@st.cache_resource
def load_model():
    """Food-101 이미지 분류 모델을 처음 실행할 때 다운로드합니다."""
    from transformers import pipeline
    return pipeline(
        "image-classification",
        model="nateraw/food",
    )

st.title("🍽️ FoodLens")
st.subheader("AI 음식 이미지 분석 기반 영양성분표 제공 서비스")
st.write("음식 사진을 업로드하면 AI가 음식 종류를 예측하고, 해당 음식의 주요 영양정보를 보여줍니다.")

st.info(
    "📌 과제용 시제품입니다. 현재는 음식 종류 분류 + 예시 영양 DB를 사용합니다. "
    "실제 서비스에서는 식품영양 DB/API와 연동하여 더 정확한 정보를 제공할 수 있습니다."
)

uploaded_file = st.file_uploader(
    "음식 사진을 업로드하세요.",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="업로드한 음식 사진", use_container_width=True)

    if st.button("🔍 AI로 음식 분석하기", type="primary", use_container_width=True):
        with st.spinner("AI가 음식 종류를 분석하고 있습니다..."):
            try:
                model = load_model()
                results = model(image, top_k=5)
            except Exception as e:
                st.error("AI 모델을 불러오는 중 문제가 발생했습니다.")
                st.exception(e)
                st.stop()

        st.success("분석이 완료되었습니다!")

        # 상위 예측 결과
        st.markdown("### 🤖 AI 분석 결과")
        result_df = pd.DataFrame([
            {
                "예측 음식": r["label"].replace("_", " ").title(),
                "신뢰도": f'{r["score"] * 100:.1f}%'
            }
            for r in results
        ])
        st.dataframe(result_df, hide_index=True, use_container_width=True)

        top_label = results[0]["label"]
        food_key = normalize_label(top_label)

        if food_key and food_key in NUTRITION_DB:
            nutrition = NUTRITION_DB[food_key]

            st.markdown(f"### 🥗 {nutrition['name']} 영양성분표")
            st.caption(f"기준량: {nutrition['serving']}")

            c1, c2, c3 = st.columns(3)
            c1.metric("🔥 열량", f"{nutrition['calories']} kcal")
            c2.metric("🍚 탄수화물", f"{nutrition['carbs']} g")
            c3.metric("🥩 단백질", f"{nutrition['protein']} g")

            c4, c5, c6 = st.columns(3)
            c4.metric("🥑 지방", f"{nutrition['fat']} g")
            c5.metric("🧂 나트륨", f"{nutrition['sodium']} mg")
            c6.metric("🍬 당류", f"{nutrition['sugar']} g")

            st.markdown("#### 📊 상세 영양성분")
            detail = pd.DataFrame({
                "영양성분": ["열량", "탄수화물", "단백질", "지방", "나트륨", "당류"],
                "함량": [
                    f"{nutrition['calories']} kcal",
                    f"{nutrition['carbs']} g",
                    f"{nutrition['protein']} g",
                    f"{nutrition['fat']} g",
                    f"{nutrition['sodium']} mg",
                    f"{nutrition['sugar']} g",
                ]
            })
            st.table(detail)

            st.warning(
                "⚠️ 영양정보는 시제품용 예시값입니다. 실제 음식은 조리법, 재료, 양에 따라 "
                "영양성분이 크게 달라질 수 있습니다."
            )
        else:
            st.warning(
                f"현재 시제품 영양 DB에 '{top_label.replace('_', ' ')}'의 정보가 없습니다. "
                "AI의 음식 분류 결과는 확인할 수 있지만 영양성분표는 제공하지 않습니다."
            )

st.divider()
st.caption("FoodLens | AI 음식 이미지 분석 기반 영양성분표 제공 웹 서비스 | 과제용 프로토타입")
