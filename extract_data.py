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


def main():

    df = filter_column(load_csv(product_image_path))
    print(df)


if __name__ == "__main__":
    main()


