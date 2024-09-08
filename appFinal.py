import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import joblib

# Set directory paths for bone and chest X-ray images
BONE_TRAIN_FRACTURED = "Images_small/train/fractured"
BONE_TRAIN_NOT_FRACTURED = "Images_small/train/not fractured"
CHEST_TRAIN_NORMAL = "chest_xray_small/train/NORMAL"
CHEST_TRAIN_PNEUMONIA = "chest_xray_small/train/PNEUMONIA"

# Load trained models
bone_model_path = 'model2.keras'
bone_model = tf.keras.models.load_model(bone_model_path)

chest_model_path = 'model.h5'
chest_model = tf.keras.models.load_model(chest_model_path)

# Load Heart Disease prediction model and scaler
svc_model = joblib.load('svc_model1.pkl')
scaler = joblib.load('minmax_scaler.pkl')

# Function to preprocess the uploaded image
def preprocess_image(image, target_size=(128, 128)):
    img = Image.open(image)
    img = img.resize(target_size)  # Resize image to match model's expected sizing
    
    # Check if image has 3 channels (RGB)
    if img.mode != 'RGB':
        img = img.convert('RGB')  # Convert to RGB if not already
    
    img = np.asarray(img)  # Convert image to numpy array
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to check if an image is a chest X-ray
def is_chest_xray(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((256, 256))  # Resize image for consistency
        
        # Convert to RGB if not already in RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.asarray(img)
        
        # Check if image is grayscale
        if len(img_array.shape) < 3:
            return False
        
        # Check if image is mostly black and white
        grayscale_img = rgb2gray(img_array)
        thresh = threshold_otsu(grayscale_img)
        binary_img = grayscale_img > thresh
        return np.mean(binary_img) > 0.5  # Adjust threshold based on your images
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False

# Function to display a random image comparison
def display_comparison_images(user_image_path, category):
    # Open user uploaded image
    user_image = Image.open(user_image_path)
    
    if category == 'Bone X-ray':
        # Choose a random image from the fractured and not fractured directories
        random_fractured_image = random.choice(os.listdir(BONE_TRAIN_FRACTURED))
        random_not_fractured_image = random.choice(os.listdir(BONE_TRAIN_NOT_FRACTURED))

        # Open the selected images
        fractured_image = Image.open(os.path.join(BONE_TRAIN_FRACTURED, random_fractured_image))
        not_fractured_image = Image.open(os.path.join(BONE_TRAIN_NOT_FRACTURED, random_not_fractured_image))

        # Create a figure for displaying the images
        figure = plt.figure(figsize=(20, 10))

        # Display the user uploaded image in the first subplot
        subplot1 = figure.add_subplot(1, 3, 1)
        plt.imshow(user_image)
        subplot1.set_title("Uploaded Image")

        # Display the fractured image in the second subplot
        subplot2 = figure.add_subplot(1, 3, 2)
        plt.imshow(fractured_image)
        subplot2.set_title("Fractured")

        # Display the not fractured image in the third subplot
        subplot3 = figure.add_subplot(1, 3, 3)
        plt.imshow(not_fractured_image)
        subplot3.set_title("Not Fractured")

    elif category == 'Chest X-ray':
        # Choose a random image from the normal and pneumonia directories
        random_normal_image = random.choice(os.listdir(CHEST_TRAIN_NORMAL))
        random_pneumonia_image = random.choice(os.listdir(CHEST_TRAIN_PNEUMONIA))

        # Open the selected images
        normal_image = Image.open(os.path.join(CHEST_TRAIN_NORMAL, random_normal_image))
        pneumonia_image = Image.open(os.path.join(CHEST_TRAIN_PNEUMONIA, random_pneumonia_image))

        # Create a figure for displaying the images
        figure = plt.figure(figsize=(20, 10))

        # Display the user uploaded image in the first subplot
        subplot1 = figure.add_subplot(1, 3, 1)
        plt.imshow(user_image)
        subplot1.set_title("Uploaded Image")

        # Display the normal image in the second subplot
        subplot2 = figure.add_subplot(1, 3, 2)
        plt.imshow(normal_image)
        subplot2.set_title("Normal")

        # Display the pneumonia image in the third subplot
        subplot3 = figure.add_subplot(1, 3, 3)
        plt.imshow(pneumonia_image)
        subplot3.set_title("Pneumonia")

    # Show the figure
    st.pyplot(figure)

# Function to predict based on user-uploaded image
def predict_image(image_path, category):
    processed_image = preprocess_image(image_path)

    if category == 'Bone X-ray':
        prediction = bone_model.predict(processed_image)
        return prediction[0][0], "Fracture", "No Fracture"
    elif category == 'Chest X-ray':
        prediction = chest_model.predict(processed_image)
        return prediction[0][0], "Pneumonia", "No Pneumonia"

# Function to predict heart disease
def predict_heart_disease(input_data):
    # Convert input_data to a 2D array for MinMaxScaler
    input_data = np.array(input_data).reshape(1, -1)
    
    # Use the loaded MinMaxScaler to transform the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict and get confidence level
    prediction = svc_model.predict(input_data_scaled)
    prediction_proba = svc_model.predict_proba(input_data_scaled)[0]  # Get the probabilities for each class (0 and 1)
    
    return prediction, prediction_proba

# Main Streamlit app
def main():
    st.sidebar.title('Medical Diagnosis Tool')
    category = st.sidebar.selectbox('Select Analysis Type:', ('Bone X-ray', 'Chest X-ray', 'Heart Disease'))
    
    if category == 'Heart Disease':
        st.title("Heart Disease Prediction")
        
        # Collect user input for all features
        age = st.number_input('Age', min_value=1, max_value=120, value=30)
        sex = st.selectbox('Sex', ['Female', 'Male'])
        resting_blood_pressure = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
        cholesterol = st.number_input('Cholesterol', min_value=100, max_value=400, value=200)
        fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        rest_ecg = st.selectbox('Rest ECG', [0, 1, 2])
        max_heart_rate_achieved = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
        exercise_induced_angina = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        st_depression = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        num_major_vessels = st.selectbox('Number of Major Vessels', [0, 1, 2, 3])
        
        # Chest Pain Type
        st.subheader('Chest Pain Type')
        chest_pain_type_0 = st.checkbox('Type 0')
        chest_pain_type_1 = st.checkbox('Type 1')
        chest_pain_type_2 = st.checkbox('Type 2')
        chest_pain_type_3 = st.checkbox('Type 3')
        
        # Thalassemia
        st.subheader('Thalassemia')
        thalassemia_0 = st.checkbox('Normal')
        thalassemia_1 = st.checkbox('Fixed Defect')
        thalassemia_2 = st.checkbox('Reversible Defect')
        thalassemia_3 = st.checkbox('Other')
        
        # ST Slope
        st.subheader('ST Slope')
        st_slope_0 = st.checkbox('Upsloping')
        st_slope_1 = st.checkbox('Flat')
        st_slope_2 = st.checkbox('Downsloping')
        
        # Map categorical inputs to numerical values
        sex = 1 if sex == 'Male' else 0
        fasting_blood_sugar = 1 if fasting_blood_sugar == 'Yes' else 0
        exercise_induced_angina = 1 if exercise_induced_angina == 'Yes' else 0
        
        # Create the input array
        input_data = [
            age, sex, resting_blood_pressure, cholesterol, fasting_blood_sugar, rest_ecg,
            max_heart_rate_achieved, exercise_induced_angina, st_depression, num_major_vessels,
            int(chest_pain_type_0), int(chest_pain_type_1), int(chest_pain_type_2), int(chest_pain_type_3),
            int(thalassemia_0), int(thalassemia_1), int(thalassemia_2), int(thalassemia_3),
            int(st_slope_0), int(st_slope_1), int(st_slope_2)
        ]
        
        # When the button is pressed, predict
        if st.button("Predict"):
            prediction, prediction_proba = predict_heart_disease(input_data)
            
            if prediction == 0:
                st.success(f"You don't have a possible heart disease. Confidence: {prediction_proba[0] * 100:.2f}%")
                st.write("Your health seems fine but always take precautionary measures to protect yourself.")
                
            else:
                st.error(f"You have a possible heart disease. Confidence: {prediction_proba[1] * 100:.2f}%")
                st.write("Meanwhile, take these precautions:")
                st.write("-Avoid smoking, vaping or using other tobacco products")
                st.write("-Limit alcohol")
                st.write("-Eat heart-healthy foods")
                st.write("-Lower your total cholesterol, LDL (bad) cholesterol and triglyceride levels")
                st.write("-Raise your HDL (good) cholesterol")
                st.write("-Manage high blood pressure")
                st.write("-Manage diabetes")
                st.write("-Keep a weight that’s healthy for you")
                st.write("-Take your medications as prescribed")
                st.write("-Take your medications as prescribed")
                st.write("-Manage your stress level")
                st.write("-Get the sleep you need")
                
    else:
        st.title(f'{category} Detection')
        st.write(f'Upload a {category.lower()} image for analysis')
        
        # File upload
        uploaded_file = st.file_uploader(f"Choose a {category.lower()} image ...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Temporarily save the uploaded image
            user_image_path = './user_uploaded_image.png'
            with open(user_image_path, 'wb') as f:
                f.write(uploaded_file.read())

            if category == 'Chest X-ray' and not is_chest_xray(user_image_path):
                st.error('Please upload a valid chest X-ray image only.')
                return

            # Display the uploaded image
            user_image = Image.open(user_image_path)
            st.image(user_image, caption='Uploaded Image', use_column_width=True)
            
            # Display normal and abnormal images for comparison
            display_comparison_images(user_image_path, category)
            
            # Predict based on the uploaded image
            probability, positive_label, negative_label = predict_image(user_image_path, category)
            
            # Display prediction result with different messages for Pneumonia and Fracture
            if probability > 0.5:
               st.error(f"High probability of {positive_label}. Please consult a doctor for further evaluation.")
    
               if positive_label == "Pneumonia":
                   st.write("Meanwhile, take these precautions for Pneumonia:")
                   st.write("- Choose heart-healthy foods, because good nutrition helps your body recover. Vegetables such as leafy greens (spinach, collard greens, kale, cabbage), broccoli, and carrots. Fruits such as apples, bananas, oranges, pears, grapes, and prunes. Whole grains such as plain oatmeal, brown rice, and whole-grain bread or tortillas. Fat-free or low-fat dairy foods such as milk, cheese, or yogurt. Protein-rich foods such as meat, fish, and eggs.")
                   st.write("- Drink plenty of fluids to help you stay hydrated.")
                   st.write("- Don’t drink alcohol or use illegal drugs.")
                   st.write("- Don’t smoke and avoid secondhand smoke. Breathing in smoke can worsen your pneumonia.")
                   st.write("- Get plenty of sleep. Good quality sleep can help your body rest and improve the response of your immune system.")
                   st.write("- Get light physical activity.")
                   st.write("- Sit upright to help you feel more comfortable and breathe more easily.")
    
               elif positive_label == "Fracture":
                   st.write("Meanwhile, take these precautions for a Fracture:")
                   st.write("- Keep the affected area immobilized to prevent further damage.")
                   st.write("- Apply ice to reduce swelling.")
                   st.write("- Take prescribed pain relievers to manage discomfort.")
                   st.write("- Avoid putting weight or stress on the fractured area until it heals.")
                   st.write("- Follow the doctor’s advice on physical therapy or rehabilitation.")
            else:
                st.success(f"Low probability of {negative_label}. Consider regular check-ups.")

                if negative_label == "No Pneumonia":
                  st.write("Keep a healthy lifestyle and consider regular health check-ups to maintain your well-being.")
                elif negative_label == "No Fracture":
                  st.write("Your bones seem fine, but always be cautious to prevent injury and maintain bone health.")


            # Remove the temporarily saved uploaded image
            os.remove(user_image_path)

# Run the app
if __name__ == '__main__':
    main()
