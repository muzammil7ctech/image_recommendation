import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
# st.set_page_config()

st.header('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('Final_Products_Images',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=50, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices
st.write("                                               ")
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):

        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("Final_Products_Images",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        st.info("")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][1]])
        with col2:
            st.image(filenames[indices[0][2]])
        with col3:
            st.image(filenames[indices[0][3]])
        with col4:
            st.image(filenames[indices[0][4]])
        with col5:
            st.image(filenames[indices[0][5]])


        st.info("                                               ")

        col6, col7, col8, col9, col10 = st.columns(5)


        with col6:
            st.image(filenames[indices[0][6]])
        with col7:
            st.image(filenames[indices[0][7]])
        with col8:
            st.image(filenames[indices[0][8]])
        with col9:
            st.image(filenames[indices[0][9]])
        with col10:
            st.image(filenames[indices[0][10]])
        st.info("                                               ")

        col11, col12, col13, col14, col15 = st.columns(5)

        with col11:
            st.image(filenames[indices[0][11]])
        with col12:
            st.image(filenames[indices[0][12]])
        with col13:
            st.image(filenames[indices[0][15]])
        with col14:
            st.image(filenames[indices[0][13]])
        with col15:
            st.image(filenames[indices[0][14]])
        # Define the number of columns
        num_columns = 5

        # Loop through each set of 5 images
        for i in range(20, 40, num_columns):
            # Create a row of columns for each set of 5 images
            st.info("                                               ")

            cols = st.columns(num_columns)
            
            # Loop through each column in the row
            for j, col in enumerate(cols):
                # Calculate the index of the image to display
                index = i + j
                
                # Check if the index is within the range of the filenames list
                if index < len(filenames):
                    # Display the image in the current column
                    col.image(filenames[indices[0][index]])
                else:
                    # If the index is out of range, display a placeholder
                    col.write("No image available")

    
    else:
        st.header("Some error occured in file upload")

