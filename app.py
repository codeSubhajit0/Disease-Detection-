import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
import os


# Custom CSS for responsive sidebar
st.markdown("""
<style>
    /* Hide default sidebar */
    .css-1d391kg {
        display: none;
    }
    
    /* Custom responsive navigation */
    .nav-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .nav-title {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .nav-buttons {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
    }
    
    .nav-button {
        background: rgba(255, 255, 255, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.3);
        color: white;
        padding: 0.7rem 1.2rem;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        font-weight: 500;
        backdrop-filter: blur(10px);
        min-width: 120px;
        text-align: center;
    }
    
    .nav-button:hover {
        background: rgba(255, 255, 255, 0.3);
        border-color: rgba(255, 255, 255, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .nav-button.active {
        background: rgba(255, 255, 255, 0.9);
        color: #667eea;
        border-color: white;
        font-weight: bold;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .nav-buttons {
            flex-direction: column;
            align-items: center;
        }
        
        .nav-button {
            width: 100%;
            max-width: 250px;
        }
        
        .nav-title {
            font-size: 1.2rem;
        }
    }
    
    /* Tablet responsiveness */
    @media (min-width: 769px) and (max-width: 1024px) {
        .nav-buttons {
            justify-content: space-around;
        }
        
        .nav-button {
            flex: 1;
            min-width: 140px;
        }
    }
    
    /* Desktop responsiveness */
    @media (min-width: 1025px) {
        .nav-buttons {
            justify-content: center;
        }
        
        .nav-button {
            min-width: 160px;
        }
    }
    
    /* Content styling */
    .main-content {
        padding: 1rem;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Custom header styling */
    .custom-header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tomato header styling */
    .tomato-header {
        background: linear-gradient(90deg, #FF6B6B, #EE5A24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Potato header styling */
    .potato-header {
        background: linear-gradient(90deg, #8B4513, #A0522D);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tomato success message styling */
    .tomato-success-message {
        background: linear-gradient(135deg, #FF6B6B, #EE5A24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Potato success message styling */
    .potato-success-message {
        background: linear-gradient(135deg, #8B4513, #A0522D);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Rice header styling */
    .rice-header {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Rice success message styling */
    .rice-success-message {
        background: linear-gradient(135deg, #2E8B57, #3CB371);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Remedy section styling */
    .remedy-section {
        background: linear-gradient(135deg, #FF9800, #FF5722);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .remedy-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .remedy-content {
        line-height: 1.6;
        font-size: 1rem;
    }
    
    /* Q&A section styling */
    .qa-section {
        background: linear-gradient(135deg, #9C27B0, #673AB7);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .qa-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .qa-answer {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 4px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Loading animation */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .spinner {
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 4px solid white;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)


#Tensorflow Model Prediction
def model_prediction(test_image, modelname="newtestmodel (1).h5"):
    model = tf.keras.models.load_model(modelname)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element


# Initialize Gemini API
def initialize_gemini():
    """Initialize Gemini API with API key"""
    try:
        # You can set your API key in Streamlit secrets or environment variable
        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("⚠️ Gemini API key not found. Please add GEMINI_API_KEY to your secrets.")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini API: {str(e)}")
        return None



def get_disease_remedy(disease_name, crop_type="plant"):
    """Get remedy suggestions for detected disease using Gemini API"""
    try:
        model = initialize_gemini()
        if not model:
            return "Unable to fetch remedy information. Please check API configuration."
        
        prompt = f"""
        As an agricultural expert, provide detailed treatment and prevention remedies for {disease_name} in {crop_type} plants.
        
        Please include:
        1. Immediate treatment steps
        2. Preventive measures
        3. Organic/chemical treatment options
        4. Best practices for future prevention
        
        Keep the response practical and actionable for farmers.
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Error fetching remedy information: {str(e)}"


def get_qa_response(question, disease_context="", crop_type="plant"):
    """Get answers to user questions using Gemini API"""
    try:
        model = initialize_gemini()
        if not model:
            return "Unable to process your question. Please check API configuration."
        
        context_prompt = f"Context: We are discussing {disease_context} in {crop_type} plants." if disease_context else ""
        
        prompt = f"""
        {context_prompt}
        
        As an agricultural expert, please answer the following question about plant diseases and farming:
        
        Question: {question}
        
        Provide a comprehensive, practical answer that would be helpful for farmers and agricultural practitioners.
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Error processing your question: {str(e)}"


# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Responsive Navigation Bar
st.markdown("""
<div class="nav-container">
    <div class="nav-title">🌱 Plant Disease Recognition System</div>
    <div class="nav-buttons">
""", unsafe_allow_html=True)

# Navigation buttons
pages = ["Home", "About", "Disease Recognition", "POTATO Disease Recognition", "TOMATO Disease Recognition", "RICE Disease Recognition"]
cols = st.columns(len(pages))

for i, page in enumerate(pages):
    with cols[i]:
        if st.button(page, key=f"nav_{page}", use_container_width=True):
            st.session_state.current_page = page

st.markdown("</div></div>", unsafe_allow_html=True)

# Get current page
app_mode = st.session_state.current_page

# Main Content with fade-in animation
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Home Page
if app_mode == "Home":
    st.markdown('<div class="custom-header"><h1>🌿 PLANT DISEASE RECOGNITION SYSTEM</h1></div>', unsafe_allow_html=True)
    
    # Display image if available
    try:
        image_path = "images.jpg"
        st.image(image_path, use_column_width=True)
    except:
        st.info("📸 Upload an image to get started!")
    
    st.markdown("""
    ## Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### 🚀 How It Works
    1. **📤 Upload Image:** Go to the **Disease Recognition**, **Potato Disease Recognition**, **Tomato Disease Recognition**, or **Rice Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **🔍 Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **📊 Results:** View the results and recommendations for further action.

    ### ⭐ Why Choose Us?
    - **🎯 Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **👥 User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **⚡ Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### 🎯 Get Started
    Choose from our specialized detection pages:
    - **🥔 Potato Disease Recognition** - Specialized for potato plant diseases
    - **🍅 Tomato Disease Recognition** - Specialized for tomato plant diseases  
    - **🌾 Rice Disease Recognition** - Specialized for rice plant diseases
    - **🌱 General Disease Recognition** - For other common plant diseases

    ### 📚 About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.markdown('<div class="custom-header"><h1>📖 About Our Project</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 📊 About Dataset
    
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on the respective GitHub repository.
    
    This dataset consists of about **87,000 RGB images** of healthy and diseased crop leaves which is categorized into **38 different classes**. The total dataset is divided into an **80/20 ratio** of training and validation set preserving the directory structure.
    
    A new directory containing **33 test images** is created later for prediction purposes.
    
    ## 📁 Dataset Content
    
    | Category | Number of Images |
    |----------|-----------------|
    | 🏋️ **Training** | 70,295 images |
    | 🧪 **Testing** | 33 images |
    | ✅ **Validation** | 17,572 images |
    
    ## 🎯 Model Performance
    Our system includes specialized models for different crops:
    
    ### 🥔 Potato Disease Model
    - **Potato Early Blight**
    - **Potato Late Blight**
    - **Potato Healthy**
    - **Pepper Bell Bacterial Spot**
    - **Pepper Bell Healthy**
    
    ### 🍅 Tomato Disease Model
    - **Tomato Bacterial Spot**
    - **Tomato Early Blight**
    - **Tomato Late Blight**
    - **Tomato Leaf Mold**
    - **Tomato Septoria Leaf Spot**
    - **Tomato Spider Mites**
    - **Tomato Target Spot**
    - **Tomato Yellow Leaf Curl Virus**
    - **Tomato Mosaic Virus**
    - **Tomato Healthy**
    
    ### 🌾 Rice Disease Model
    - **Rice Bacterial Leaf Blight**
    - **Rice Brown Spot**
    - **Rice Leaf Smut**
    - **Rice Blast**
    - **Rice Tungro**
    - **Rice Healthy**
    
    ### 🌱 General Plant Disease Model
    - **Bacterial Leaf Blight**
    - **Brown Spot**
    - **Leaf Smut**
    
    ## 🔬 Technology Stack
    - **TensorFlow/Keras** for deep learning
    - **Streamlit** for web interface
    - **Computer Vision** for image processing
    - **Convolutional Neural Networks** for classification
    """)

elif app_mode == "POTATO Disease Recognition":
    st.markdown('<div class="potato-header"><h1>🥔 Potato Disease Recognition</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Upload a potato leaf image for disease detection
    Our specialized potato model can identify:
    - **🦠 Pepper Bell Bacterial Spot**
    - **✅ Pepper Bell Healthy**
    - **🍂 Potato Early Blight**
    - **🍃 Potato Late Blight**
    - **💚 Potato Healthy**
    """)
    
    test_image = st.file_uploader("Choose a Potato Leaf Image:", type=['jpg', 'jpeg', 'png'], key="potato_uploader")
    
    if test_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🖼️ Show Image", use_container_width=True, key="potato_show"):
                st.image(test_image, caption="Uploaded Potato Leaf Image", use_column_width=True)
        
        with col2:
            if st.button("🔍 Predict", use_container_width=True, key="potato_predict"):
                with st.spinner('Analyzing potato leaf image...'):
                    try:
                        result_index = model_prediction(test_image, 'newtestmodel1.keras')
                        class_name = ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
                        
                        st.markdown(f"""
                        <div class="potato-success-message">
                            <h3>🎯 Potato Disease Prediction Result</h3>
                            <p>Model is predicting it's a <strong>{class_name[result_index]}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Store prediction result for remedy and Q&A
                        st.session_state.potato_prediction = class_name[result_index]
                        st.session_state.potato_crop_type = "potato"
                        
                    except Exception as e:
                        st.error(f"Error in prediction: {str(e)}")
        
        # Remedy Section
        if 'potato_prediction' in st.session_state:
            st.markdown("""
            <div class="remedy-section">
                <div class="remedy-title">💊 Treatment & Prevention Remedies</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Get Remedy Suggestions", key="potato_remedy_btn", use_container_width=True):
                with st.spinner("Fetching remedy information..."):
                    remedy = get_disease_remedy(st.session_state.potato_prediction, st.session_state.potato_crop_type)
                    st.session_state.potato_remedy = remedy
            
            if 'potato_remedy' in st.session_state:
                st.markdown(f"""
                <div class="remedy-section">
                    <div class="remedy-content">{st.session_state.potato_remedy}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Q&A Section
        if 'potato_prediction' in st.session_state:
            st.markdown("""
            <div class="qa-section">
                <div class="qa-title">❓ Ask Additional Questions</div>
            </div>
            """, unsafe_allow_html=True)
            
            user_question = st.text_area(
                "Ask any question about this disease or potato farming:",
                placeholder="e.g., How can I prevent this disease in the future?",
                key="potato_question"
            )
            
            if st.button("🤔 Get Answer", key="potato_qa_btn", use_container_width=True):
                if user_question.strip():
                    with st.spinner("Processing your question..."):
                        answer = get_qa_response(
                            user_question, 
                            st.session_state.potato_prediction, 
                            st.session_state.potato_crop_type
                        )
                        st.session_state.potato_answer = answer
                else:
                    st.warning("Please enter a question first.")
            
            if 'potato_answer' in st.session_state:
                st.markdown(f"""
                <div class="qa-section">
                    <div class="qa-answer">{st.session_state.potato_answer}</div>
                </div>
                """, unsafe_allow_html=True)

elif app_mode == "RICE Disease Recognition":
    st.markdown('<div class="rice-header"><h1>🌾 Rice Disease Recognition</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Upload a rice leaf image for disease detection
    Our specialized rice model can identify:
    - **🦠 Rice Bacterial Leaf Blight**
    - **🟤 Rice Brown Spot**
    - **🍃 Rice Leaf Smut**
    - **💥 Rice Blast**
    - **🦠 Rice Tungro**
    - **💚 Rice Healthy**
    """)
    
    test_image = st.file_uploader("Choose a Rice Leaf Image:", type=['jpg', 'jpeg', 'png'], key="rice_uploader")
    
    if test_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🖼️ Show Image", use_container_width=True, key="rice_show"):
                st.image(test_image, caption="Uploaded Rice Leaf Image", use_column_width=True)
        
        with col2:
            if st.button("🔍 Predict", use_container_width=True, key="rice_predict"):
                with st.spinner('Analyzing rice leaf image...'):
                    try:
                        result_index = model_prediction(test_image, 'newtestmodel (1).h5')
                        class_name = ['Rice___Bacterial_leaf_blight', 'Rice___Brown_spot', 'Rice___Leaf_smut',
                                    'Rice___Blast', 'Rice___Tungro', 'Rice___healthy']
                        
                        st.markdown(f"""
                        <div class="rice-success-message">
                            <h3>🎯 Rice Disease Prediction Result</h3>
                            <p>Model is predicting it's a <strong>{class_name[result_index]}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Store prediction result for remedy and Q&A
                        st.session_state.rice_prediction = class_name[result_index]
                        st.session_state.rice_crop_type = "rice"
                        
                    except Exception as e:
                        st.error(f"Error in prediction: {str(e)}")
        
        # Remedy Section
        if 'rice_prediction' in st.session_state:
            st.markdown("""
            <div class="remedy-section">
                <div class="remedy-title">💊 Treatment & Prevention Remedies</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Get Remedy Suggestions", key="rice_remedy_btn", use_container_width=True):
                with st.spinner("Fetching remedy information..."):
                    remedy = get_disease_remedy(st.session_state.rice_prediction, st.session_state.rice_crop_type)
                    st.session_state.rice_remedy = remedy
            
            if 'rice_remedy' in st.session_state:
                st.markdown(f"""
                <div class="remedy-section">
                    <div class="remedy-content">{st.session_state.rice_remedy}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Q&A Section
        if 'rice_prediction' in st.session_state:
            st.markdown("""
            <div class="qa-section">
                <div class="qa-title">❓ Ask Additional Questions</div>
            </div>
            """, unsafe_allow_html=True)
            
            user_question = st.text_area(
                "Ask any question about this disease or rice farming:",
                placeholder="e.g., What is the best time to apply treatment?",
                key="rice_question"
            )
            
            if st.button("🤔 Get Answer", key="rice_qa_btn", use_container_width=True):
                if user_question.strip():
                    with st.spinner("Processing your question..."):
                        answer = get_qa_response(
                            user_question, 
                            st.session_state.rice_prediction, 
                            st.session_state.rice_crop_type
                        )
                        st.session_state.rice_answer = answer
                else:
                    st.warning("Please enter a question first.")
            
            if 'rice_answer' in st.session_state:
                st.markdown(f"""
                <div class="qa-section">
                    <div class="qa-answer">{st.session_state.rice_answer}</div>
                </div>
                """, unsafe_allow_html=True)

elif app_mode == "TOMATO Disease Recognition":
    st.markdown('<div class="tomato-header"><h1>🍅 Tomato Disease Recognition</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Upload a tomato leaf image for disease detection
    Our specialized tomato model can identify:
    - **🦠 Tomato Bacterial Spot**
    - **🍂 Tomato Early Blight**
    - **🍃 Tomato Late Blight**
    - **🍄 Tomato Leaf Mold**
    - **🟤 Tomato Septoria Leaf Spot**
    - **🕷️ Tomato Spider Mites**
    - **🎯 Tomato Target Spot**
    - **💛 Tomato Yellow Leaf Curl Virus**
    - **🦠 Tomato Mosaic Virus**
    - **💚 Tomato Healthy**
    """)
    
    test_image = st.file_uploader("Choose a Tomato Leaf Image:", type=['jpg', 'jpeg', 'png'], key="tomato_uploader")
    
    if test_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🖼️ Show Image", use_container_width=True, key="tomato_show"):
                st.image(test_image, caption="Uploaded Tomato Leaf Image", use_column_width=True)
        
        with col2:
            if st.button("🔍 Predict", use_container_width=True, key="tomato_predict"):
                with st.spinner('Analyzing tomato leaf image...'):
                    try:
                        result_index = model_prediction(test_image, 'tomato_dataset.keras')
                        class_name = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
                                    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
                                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                                    'Tomato___healthy']
                        
                        st.markdown(f"""
                        <div class="tomato-success-message">
                            <h3>🎯 Tomato Disease Prediction Result</h3>
                            <p>Model is predicting it's a <strong>{class_name[result_index]}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Store prediction result for remedy and Q&A
                        st.session_state.tomato_prediction = class_name[result_index]
                        st.session_state.tomato_crop_type = "tomato"
                        
                    except Exception as e:
                        st.error(f"Error in prediction: {str(e)}")
        
        # Remedy Section
        if 'tomato_prediction' in st.session_state:
            st.markdown("""
            <div class="remedy-section">
                <div class="remedy-title">💊 Treatment & Prevention Remedies</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Get Remedy Suggestions", key="tomato_remedy_btn", use_container_width=True):
                with st.spinner("Fetching remedy information..."):
                    remedy = get_disease_remedy(st.session_state.tomato_prediction, st.session_state.tomato_crop_type)
                    st.session_state.tomato_remedy = remedy 
            
            if 'tomato_remedy' in st.session_state:
                st.markdown(f"""
                <div class="remedy-section">
                    <div class="remedy-content">{st.session_state.tomato_remedy}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Q&A Section
        if 'tomato_prediction' in st.session_state:
            st.markdown("""
            <div class="qa-section">
                <div class="qa-title">❓ Ask Additional Questions</div>
            </div>
            """, unsafe_allow_html=True)
            
            user_question = st.text_area(
                "Ask any question about this disease or tomato farming:",
                placeholder="e.g., What are the best fungicides for this disease?",
                key="tomato_question"
            )
            
            if st.button("🤔 Get Answer", key="tomato_qa_btn", use_container_width=True):
                if user_question.strip():
                    with st.spinner("Processing your question..."):
                        answer = get_qa_response(
                            user_question,
                            st.session_state.tomato_prediction, 
                            st.session_state.tomato_crop_type
                        )
                        st.session_state.tomato_answer = answer
                else:
                    st.warning("Please enter a question first.")
            
            if 'tomato_answer' in st.session_state:
                st.markdown(f"""
                <div class="qa-section">
                    <div class="qa-answer">{st.session_state.tomato_answer}</div>
                </div>
                """, unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.markdown('<div class="custom-header"><h1>🔬 General Disease Recognition</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Upload a plant leaf image for disease detection
    Our general model can identify:
    - **🦠 Bacterial Leaf Blight**
    - **🟤 Brown Spot**
    - **🍃 Leaf Smut**
    """)
    
    test_image = st.file_uploader("Choose a Plant Leaf Image:", type=['jpg', 'jpeg', 'png'], key="general_uploader")
    
    if test_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🖼️ Show Image", use_container_width=True, key="general_show"):
                st.image(test_image, caption="Uploaded Plant Leaf Image", use_column_width=True)
        
        with col2:
            if st.button("🔍 Predict", use_container_width=True, key="general_predict"):
                with st.spinner('Analyzing plant leaf image...'):
                    try:
                        result_index = model_prediction(test_image)
                        class_name = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
                        
                        st.markdown(f"""
                        <div class="success-message">
                            <h3>🎯 Plant Disease Prediction Result</h3>
                            <p>Model is predicting it's a <strong>{class_name[result_index]}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Store prediction result for remedy and Q&A
                        st.session_state.general_prediction = class_name[result_index]
                        st.session_state.general_crop_type = "plant"
                        
                    except Exception as e:
                        st.error(f"Error in prediction: {str(e)}")
        
        # Remedy Section
        if 'general_prediction' in st.session_state:
            st.markdown("""
            <div class="remedy-section">
                <div class="remedy-title">💊 Treatment & Prevention Remedies</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Get Remedy Suggestions", key="general_remedy_btn", use_container_width=True):
                with st.spinner("Fetching remedy information..."):
                    remedy = get_disease_remedy(st.session_state.general_prediction, st.session_state.general_crop_type)
                    st.session_state.general_remedy = remedy
            
            if 'general_remedy' in st.session_state:
                st.markdown(f"""
                <div class="remedy-section">
                    <div class="remedy-content">{st.session_state.general_remedy}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Q&A Section
        if 'general_prediction' in st.session_state:
            st.markdown("""
            <div class="qa-section">
                <div class="qa-title">❓ Ask Additional Questions</div>
            </div>
            """, unsafe_allow_html=True)
            
            user_question = st.text_area(
                "Ask any question about this disease or plant farming:",
                placeholder="e.g., How to improve soil conditions to prevent this disease?",
                key="general_question"
            )
            
            if st.button("🤔 Get Answer", key="general_qa_btn", use_container_width=True):
                if user_question.strip():
                    with st.spinner("Processing your question..."):
                        answer = get_qa_response(
                            user_question, 
                            st.session_state.general_prediction, 
                            st.session_state.general_crop_type
                        )
                        st.session_state.general_answer = answer
                else:
                    st.warning("Please enter a question first.")
            
            if 'general_answer' in st.session_state:
                st.markdown(f"""
                <div class="qa-section">
                    <div class="qa-answer">{st.session_state.general_answer}</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)