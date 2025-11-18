import os
import streamlit as st
import joblib
from PIL import Image
import numpy as np
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel
import torch

# Load model and processors
model = joblib.load('performance_predictor.pkl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)

# Feature extraction functions (copy from above)
def extract_text_features(text):
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def extract_image_features(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = clip_processor(images=image, return_tensors='pt').to(device)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        return features.squeeze().cpu().numpy()
    except:
        return np.zeros(512)

def predict_performance(caption, hashtags, image_path=None):
    text = caption + ' ' + hashtags
    text_feat = extract_text_features(text)
    img_feat = extract_image_features(image_path) if image_path else np.zeros(512)
    features = np.concatenate([text_feat, img_feat]).reshape(1, -1)
    pred = model.predict(features)[0]
    labels = ['Low', 'Medium', 'High']
    return labels[pred]

st.title('Instagram Post Performance Predictor')
st.write('Upload an image, enter caption and hashtags to predict performance.')

caption = st.text_input('Caption')
hashtags = st.text_input('Hashtags (comma-separated)')

uploaded_file = st.file_uploader('Upload Image', type=['jpg', 'png'])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    if st.button('Predict'):
    temp_path = 'temp_image.jpg'
    image.save(temp_path)
    pred = predict_performance(caption, hashtags, temp_path)
    st.write(f'Predicted Performance: {pred}')
        if os.path.exists(temp_path):  # Check if file exists before removing
            os.remove(temp_path)  # Clean up

