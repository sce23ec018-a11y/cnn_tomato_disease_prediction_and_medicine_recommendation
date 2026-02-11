import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------------
#  CONFIGURATION
# ------------------------------
IMG_SIZE = 224
CLASS_NAMES = [
    'Bacterial_Spot',
    'Early_Blight',
    'Healthy',
    'Late_Blight',
    'Septoria_Leaf_Spot'
]

# ------------------------------
#  MEDICINE RECOMMENDER
# ------------------------------
class FastMedicineRecommender:
    def __init__(self):
        self.recommendations = {
            'Bacterial_Spot': {
                'chemical': ['Copper hydroxide every 7-10 days', 'Streptomycin for severe cases'],
                'organic': ['Copper soap weekly', 'Neem oil spray'],
                'prevention': ['Use disease-free seeds', 'Avoid overhead watering', 'Crop rotation']
            },
            'Early_Blight': {
                'chemical': ['Chlorothalonil every 7-10 days', 'Azoxystrobin systemic'],
                'organic': ['Copper fungicide', 'Baking soda spray'],
                'prevention': ['Remove lower leaves', 'Improve air circulation', 'Mulch']
            },
            'Healthy': {
                'chemical': ['No treatment needed'],
                'organic': ['Continue organic practices'],
                'prevention': ['Regular monitoring', 'Proper watering', 'Balanced fertilizer']
            },
            'Late_Blight': {
                'chemical': ['Chlorothalonil immediately', 'Metalaxyl systemic'],
                'organic': ['Copper fungicide before rain', 'Potassium bicarbonate'],
                'prevention': ['Destroy infected plants', 'Use resistant varieties', 'Drip irrigation']
            },
            'Septoria_Leaf_Spot': {
                'chemical': ['Chlorothalonil weekly', 'Mancozeb protective'],
                'organic': ['Copper soap', 'Sulfur spray'],
                'prevention': ['Remove infected leaves', 'Water at base', 'Stake plants']
            }
        }
    
    def get_recommendation(self, disease, confidence):
        if disease not in self.recommendations:
            return "Disease not recognized"
        rec = self.recommendations[disease]
        return f"""
🔍 **Diagnosis:** {disease} ({confidence:.1%})
        
💊 **Chemical Treatments:**  
{chr(10).join(['• ' + t for t in rec['chemical']])}

🌿 **Organic/Biological:**  
{chr(10).join(['• ' + t for t in rec['organic']])}

✅ **Prevention Measures:**  
{chr(10).join(['• ' + t for t in rec['prevention']])}
"""

# ------------------------------
#  LOAD MODEL – DIRECT .KERAS (NO DOWNLOAD, NO LEGACY)
# ------------------------------
@st.cache_resource
def load_model():
    """Load the converted .keras model – works on TF 2.13+ and 2.20+."""
    model = tf.keras.models.load_model('tomato_model_fast.keras', compile=False)
    return model

# ------------------------------
#  IMAGE PREPROCESSING
# ------------------------------
def preprocess_image(uploaded_img):
    img = Image.open(uploaded_img).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# ------------------------------
#  PREDICTION
# ------------------------------
def predict(image_array, model):
    predictions = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:3]
    top_classes = [CLASS_NAMES[i] for i in top_indices]
    top_confidences = predictions[top_indices]
    return top_classes, top_confidences, predictions

# ------------------------------
#  STREAMLIT UI
# ------------------------------
def main():
    st.set_page_config(page_title="Tomato Disease Detection", layout="wide")
    st.title("🌱 Tomato Disease Prediction & Medicine Recommendation")
    st.markdown("Upload a photo of a tomato leaf to identify the disease and get treatment recommendations.")
    
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses a deep learning model (MobileNetV2) trained on 5 classes of tomato leaf diseases. "
        "It achieves **90-95% accuracy** on test images. Upload a clear, well-lit leaf image for best results."
    )
    st.sidebar.markdown("**Supported diseases:**")
    for cls in CLASS_NAMES:
        st.sidebar.write(f"- {cls}")
    
    with st.spinner("Loading AI model... ⏳"):
        model = load_model()
    st.success("Model loaded successfully!")
    
    uploaded_file = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Analyzing image..."):
            img_array, _ = preprocess_image(uploaded_file)
            top_classes, top_confidences, all_preds = predict(img_array, model)
        
        with col2:
            st.subheader("🔬 Prediction Results")
            primary_class = top_classes[0]
            primary_conf = top_confidences[0]
            st.markdown(f"### **{primary_class}**")
            st.markdown(f"**Confidence:** {primary_conf:.2%}")
            
            st.markdown("**Top-3 possibilities:**")
            for i, (cls, conf) in enumerate(zip(top_classes, top_confidences)):
                st.write(f"{i+1}. {cls} – {conf:.2%}")
            
            st.subheader("💊 Treatment Recommendation")
            recommender = FastMedicineRecommender()
            rec_text = recommender.get_recommendation(primary_class, primary_conf)
            st.markdown(rec_text)
        
        with st.expander("📊 Confidence Scores for All Classes"):
            fig, ax = plt.subplots(figsize=(8, 4))
            y_pos = np.arange(len(CLASS_NAMES))
            ax.barh(y_pos, all_preds, color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(CLASS_NAMES)
            ax.set_xlabel('Confidence')
            ax.set_title('Model Confidence per Class')
            st.pyplot(fig)
    else:
        st.info("👆 Please upload an image to begin.")
    
    st.markdown("---")
    st.markdown("📁 **Model trained on [Tomato Leaf Disease Dataset](https://www.kaggle.com/datasets)** • ⚖️ MIT License")

if __name__ == "__main__":
    main()
