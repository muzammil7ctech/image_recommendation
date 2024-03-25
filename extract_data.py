from Extract_code.Code.extract import *
from Extract_code.Code.load_csv import *
from Extract_code.Code.filtering import *
from Extract_code.lib.libraries import *

confing_name='config.ini'
configur = ConfigParser() 
configur.read(confing_name)
BASE_PATH=os.getcwd()
print(BASE_PATH)
product_image_path = configur.get('image_url_data_csv','producte_image_path')
product_image_path = f"{BASE_PATH}{product_image_path}"
product_reviews_image_path = configur.get('image_url_data_csv','product_reviews_image_path')
product_reviews_image_path = f"{BASE_PATH}{product_reviews_image_path}"
image_save_path = configur.get('image_save','image_Save_path')
image_save_path =  f"{BASE_PATH}{image_save_path}"
print(image_save_path)
reviewimage_save_path = configur.get('image_save','reviewimage_save_path')
reviewimage_save_path =  f"{BASE_PATH}{reviewimage_save_path}"

productimage_url = configur.get('url','product_url')
productreview_url = configur.get('url','productreview_url')



def main():

    Download_images(filter_column(load_csv(product_image_path),column=['product_id', 'product_image_url']),image_save_path,productimage_url,col1="product_id",col2="product_image_url")


    # Download_images(filter_column(load_csv(product_reviews_image_path),column=['product_id', 'img_link_s3']),reviewimage_save_path,productreview_url,col1="product_id",col2="img_link_s3")





if __name__ == "__main__":
    main()


