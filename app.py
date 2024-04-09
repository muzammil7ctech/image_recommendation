import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Modify the model to include GlobalMaxPooling2D
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to extract features from an image
def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except FileNotFoundError:
        print(f"File not found: {img_path}")
        return None

# Read the CSV file containing file names and product IDs
df_filenames = pd.read_csv('data.csv')

# Create a DataFrame to store file names, product IDs, and embeddings
df = pd.DataFrame(columns=['filename', 'product_id', 'embedding'])

# Iterate over each row in the original DataFrame
for index, row in df_filenames.iterrows():
    filename_id = row['id']
    product_id = row['product_id']
    file_link = row['product_link']
    base_path = file_link.rsplit(".", 1)[0]
    print(base_path)

    # Extract features from the image
    file_path = os.path.join('Final_Products_Images', str(base_path)+".jpg")
    print(file_path)
    embedding = extract_features(file_path, model)

    # Append the filename, product ID, and embedding to the new DataFrame
    if embedding is not None:
        df = df._append({'filename': filename_id, 'product_id': product_id, 'embedding': embedding}, ignore_index=True)
        
# Calculate cosine similarity between embeddings
embeddings = np.array(df['embedding'].tolist())
cosine_similarities = cosine_similarity(embeddings, embeddings)

# Get the product IDs
product_ids = df['product_id'].tolist()

# Create a list to store the cosine similarity results
similarity_results = []

# Iterate over the cosine similarity matrix and save the results
for i in range(len(product_ids)):
    for j in range(i+1, len(product_ids)):
        similarity_results.append({
            'original_product_id': product_ids[i],
            'match_product_id': product_ids[j],
            'cosine_similarity_score': cosine_similarities[i, j]
        })

# Create a DataFrame from the list
cosine_similarity_df = pd.DataFrame(similarity_results)

# Save the cosine similarities DataFrame to a CSV file
cosine_similarity_df.to_csv('cosine_similarities.csv', index=False)

print("Cosine similarities calculated and saved.")
