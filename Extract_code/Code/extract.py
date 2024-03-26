from Extract_code.lib.libraries  import *

# def Download_images(df = None,image_save_path=None,url= None,col1=None,col2=None):
#     for id , path in zip(df[col1],df[col2]):
#         id = id 
#         path = path 
#         if path.endswith(".png"):
#             path = path.replace(".png", ".jpg")
#         os.makedirs(image_save_path, exist_ok=True)
#         final_url = f"{url}{path}""?profile=a"
#         print(final_url)
#         req = urllib.request.Request(final_url, headers={'User-Agent': 'Mozilla/5.0'})
#         urllib.request.urlretrieve(req, image_save_path)        
#         # Download the image
#         time.sleep(2)

#         print(f"This product_id : {id} <<< {path} >>> downloaded ")
                
def Download_images(df=None, image_save_path=None, url=None, col1=None, col2=None):
    failed_path = ""
    os.makedirs(image_save_path, exist_ok=True)
    for id, path in zip(df[col1], df[col2]):
        path = str(path)
        print(path) # Convert path to string
        if path.endswith(".png"):
            path = path.replace(".png", ".jpg")
        if not os.path.exists(os.path.join(image_save_path, path)):
            final_url = f"{url}{path}?profile=a"
            print(final_url)
            try:
                retries = 0
                file_name, file_extension = os.path.splitext(path)
                if file_name.endswith("1") :
                    print("------------------------------yes ----------------------------")
                    while retries < 2:
                        response = requests.get(final_url, headers={'User-Agent': 'Mozilla/5.0'})
                        
                        if response.status_code == 200:
                            filename = os.path.splitext(path)[0]
                            image_save_full_path = os.path.join(image_save_path, f"{filename}.jpg")
                            with open(image_save_full_path, 'wb') as file:
                                file.write(response.content)
                            print(f"Image saved for product_id: {path}")
                            break  # Break the loop if the image is saved successfully
                        else:
                            print(f"Failed to download image for product_id: {id} - Status code {response.status_code}")
                            retries += 1
                            # time.sleep(1) 
            except Exception as e:
                print(f"Error downloading image for product_id: {id} - {e}")