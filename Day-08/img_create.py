#Creating a dataset by downloading images using bing_image_downloader

from bing_image_downloader import downloader

downloader.download( query="superman", limit=10, output_dir="dataset\train" )