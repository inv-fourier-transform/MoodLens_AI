import streamlit as st
import pandas as pd
from model_helper import detect_emotion
import os
from dotenv import load_dotenv

# load_dotenv(dotenv_path='../.env')  # Go up one level to find .env
# model_path = os.getenv("MODEL_PATH")

import os
import streamlit as st

# Streamlit Cloud: use secrets | Local: use .env file
try:
    model_path = st.secrets["MODEL_PATH"]["value"]
except:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path='../.env')
    model_path = os.getenv("MODEL_PATH")

# Page config
st.set_page_config(
    page_title="MoodLens AI 🔮",
    page_icon="🔮",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .dominant-emotion {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .disclaimer {
        font-size: 0.85rem;
        color: #888;
        font-style: italic;
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        border-left: 3px solid #ccc;
    }
    .emotion-table {
        font-size: 1.1rem;
    }
    .stDataFrame {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title with symbol
st.markdown('<h1 class="main-title">MoodLens AI 🔮</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">✨ Decode the emotional vibe of any image with AI-powered perception ✨</p>',
            unsafe_allow_html=True)

# Divider
st.markdown("---")

# File uploader with better styling
uploaded_image = st.file_uploader(
    "📤 Drop your image here or click to upload",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to analyze its emotional content"
)

if uploaded_image:
    # Save temp file
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(uploaded_image, use_container_width=True, caption="")

    with col2:
        # Show loading spinner
        with st.spinner("🔍 Analyzing emotional vibes..."):
            result = detect_emotion(image_path)

        st.subheader("🎭 Emotional Analysis")

        # Emotion to emoji mapping
        emotion_emojis = {
            'anger': '😠',
            'disgust': '😖',
            'fear': '😨',
            'joy': '😄',
            'sadness': '😢',
            'surprise': '😲'
        }

        # Prepare data for table
        emotions_data = []
        for emotion, prob in result['soft_probabilities'].items():
            emotions_data.append({
                'Emotion': f"{emotion_emojis.get(emotion, '❓')} {emotion.capitalize()}",
                'Predicted Probability': f"{prob * 100:.2f}%",
                '_prob': prob  # For sorting
            })

        # Sort by probability descending
        emotions_data = sorted(emotions_data, key=lambda x: x['_prob'], reverse=True)

        # Create DataFrame (drop the helper column)
        df = pd.DataFrame(emotions_data)
        df_display = df.drop(columns=['_prob'])

        # Display table with styling
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Emotion": st.column_config.Column(
                    "🎭 Emotion",
                    width="medium",
                    help="Detected emotion with corresponding emoji"
                ),
                "Predicted Probability": st.column_config.Column(
                    "📊 Probability",
                    width="medium",
                    help="Confidence score for each emotion"
                )
            }
        )

        # Progress bars for visual appeal
        st.subheader("📈 Probability Distribution")
        for item in emotions_data[:3]:  # Show top 3
            emotion_name = item['Emotion'].split(' ', 1)[1]  # Remove emoji
            prob = item['_prob']
            st.progress(prob, text=f"{item['Emotion']}: {prob * 100:.1f}%")

    # Dominant emotion display
    dominant = result['hard_label']
    dominant_emoji = emotion_emojis.get(dominant, '❓')
    confidence = result['confidence'] * 100

    st.markdown(f"""
        <div class="dominant-emotion">
            🎯 The dominant emotion predicted is <br>
            <span style="font-size: 2rem;">{dominant_emoji} {dominant.capitalize()}</span> <br>
            with a confidence of {confidence:.2f}%
        </div>
    """, unsafe_allow_html=True)

    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)

    # Disclaimer
    st.markdown("""
        <div class="disclaimer">
            💡 <strong>Note:</strong> Emotions are inherently subjective & shaped by individual experiences, 
            cultural backgrounds, & personal perceptions. The AI predictions represent probabilistic interpretations 
            & may not reflect how every viewer would perceive the image. Results should be considered as 
            informed estimates rather than definitive emotional truths.
        </div>
    """, unsafe_allow_html=True)
