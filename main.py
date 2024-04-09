import tkinter as tk
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
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

# Read the CSV file
df = pd.read_csv(r"40_threshold_60_product.csv")
df_cnn = pd.read_csv(r"data.csv")

# df_cosin = pd.read_csv("cosine_similarities.csv")

# Define driver as a global variable
driver = None

url = "https://dev.fashionpass.com/searchCollectionById"

# Initialize the WebDriver if it's not already initialized
if driver is None:
    driver = webdriver.Chrome()

driver.get(url)

print(f"<<<< Url : {url} >>>>")
def path_format(path):
    path = path.rsplit(".", 1)[0] + ".jpg"
    return path



# st.set_page_config()




def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    threshold = 0.60
    neighbors = NearestNeighbors(n_neighbors=50, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])

    similar_indices = [indices[distances < threshold]]
    similar_distances = distances[distances < threshold]
    print(similar_distances,similar_indices)
    print(indices,distances)
    return similar_indices
def on_button_click():
    global driver  # Use the global driver variable
    
    # Filter the DataFrame based on the value of "Product_A"
    value1 = int(entry1.get())
    filtered_df = df[df["original_product"] == value1]
    
    # Sort the filtered DataFrame by 'cosin_score'
    sorted_df = filtered_df.sort_values(by='cosin_score', ascending=False)
    print(sorted_df)
    filtered_df = sorted_df[sorted_df["cosin_score"] >= 0.60]
    print(filtered_df)

    # Display the sorted DataFrame (you might want to update this based on how you want to display the data)
    all_similar_ids = filtered_df["matched_product"].iloc[:50].tolist()
    top_ten = filtered_df["matched_product"].iloc[:10].tolist()
    top_ten.insert(0,value1)
    print("top_ten : ",top_ten)
    for i in top_ten:
        try :    
            df_cnn_data = df_cnn[df_cnn["product_id"] == i]
            print(df_cnn_data)
            df_cnn_data["product_link"] = df_cnn_data["product_link"].apply(path_format)


            filtered_df = df_cnn_data[df_cnn_data["product_link"].str.endswith("1.jpg")]

            path = filtered_df["product_link"].iloc[:1].values[0]  # Get the first image path
            print(path)

            model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
            model.trainable = False

            model = tensorflow.keras.Sequential([
                model,
                GlobalMaxPooling2D()
            ])
            feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
            filenames = pickle.load(open('filenames.pkl','rb'))

            features = feature_extraction(os.path.join("Final_Products_Images",path),model)
            indices = recommend(features,feature_list)
            for j in range(1,2):
                indices_image_path = filenames[indices[0][j]]
                base_path = os.path.basename(indices_image_path)
                base_path = base_path.rsplit(".", 1)[0]

                print('-------------------------------------------------')
                print()
                print()
                print(base_path)
                cnn_similar__df= df_cnn[df_cnn['product_link'].str.contains(base_path)]
                similar_cnn_id = cnn_similar__df["product_id"].tolist()
                print(similar_cnn_id)
                if i == value1:
                    similar_cnn_id = int(similar_cnn_id[0])

                    all_similar_ids.insert(j, similar_cnn_id)


                    print(f"if working because of {i} and {value1} is similar")
                else :
                    all_similar_ids.extend(similar_cnn_id)
                    print("else working")
                print()
                print()
        except:
            print("Problem in Cnn work FLow")

        values = ', '.join(map(str, all_similar_ids))
        print(" values: ",values)
    


    
    try:
        print("working")
        input_field = driver.find_element(By.CLASS_NAME, "form-control")
        # Clear any existing text in the input field
        input_field.clear()
        # Enter the desired value into the input field
        input_field.send_keys(values)
        time.sleep(1)
        search_button = driver.find_element(By.CLASS_NAME, "btn-primary")
        search_button.click()
        time.sleep(5)
    except Exception as e:
        print("scrapper not working:", e)

# Create the main window
root = tk.Tk()
root.title("Input GUI")

# Create input fields
entry1 = tk.Entry(root, width=50)
entry1.pack()

# Create a button
button = tk.Button(root, text="Submit", command=on_button_click)
button.pack()

# Create a close button
close_button = tk.Button(root, text="Close Browser", command=driver.quit if driver else None)
close_button.pack()

# Start the GUI event loop
root.mainloop()
