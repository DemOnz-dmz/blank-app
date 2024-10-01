import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# โหลดโมเดล MobileNetV2 ที่ฝึกมาแล้ว
model = MobileNetV2(weights='imagenet')

# ฐานข้อมูลสูตรอาหาร
recipe_db = {
    'apple': ['Apple Pie', 'Apple Crumble', 'Fruit Salad'],
    'banana': ['Banana Smoothie', 'Banana Bread'],
    'chicken': ['Grilled Chicken', 'Chicken Curry'],
}

# ฟังก์ชันแนะนำสูตรอาหาร
def recommend_recipe(ingredient):
    recipes = recipe_db.get(ingredient.lower(), ["No recipe available"])
    return recipes

# ฟังก์ชันแปลงภาพและทำนายวัตถุดิบ
def preprocess_and_predict(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)

    # ทำนายวัตถุดิบ
    predictions = model.predict(processed_img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    return decoded_predictions

# ส่วนติดต่อผู้ใช้ (UI)
st.title("แนะนำเมนู")

# ให้ผู้ใช้อัปโหลดไฟล์รูปภาพ
uploaded_file = st.file_uploader("Choose an image of an ingredient...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # แสดงภาพที่อัปโหลด
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    st.write("Identifying the ingredient...")

    # ทำนายวัตถุดิบ
    predictions = preprocess_and_predict(img)

    # แสดงผลลัพธ์การทำนาย
    st.write("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        st.write(f"{i+1}. {label} (confidence: {score:.4f})")
    
    # ใช้วัตถุดิบที่มีความมั่นใจสูงสุดในการแนะนำสูตรอาหาร
    top_ingredient = predictions[0][1]
    recipes = recommend_recipe(top_ingredient)
    
    st.write(f"Recommended Recipes for {top_ingredient}:")
    for recipe in recipes:
        st.write(f"- {recipe}")
