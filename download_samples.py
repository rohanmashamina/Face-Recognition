import os
import requests
from PIL import Image
from io import BytesIO

def download_sample_images():
    # Sample image URLs (these are placeholder URLs - you should replace them with actual face images)
    sample_images = {
        'person1': [
            'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg',
            'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
        ],
        'person2': [
            'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg',
            'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
        ]
    }
    
    for person, urls in sample_images.items():
        person_dir = os.path.join('dataset', person)
        os.makedirs(person_dir, exist_ok=True)
        
        for i, url in enumerate(urls):
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                img.save(os.path.join(person_dir, f'image{i+1}.jpg'))
                print(f"Downloaded image for {person}")
            except Exception as e:
                print(f"Error downloading image for {person}: {str(e)}")

if __name__ == "__main__":
    print("Downloading sample images...")
    download_sample_images()
    print("Done!") 