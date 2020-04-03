import io
import os

from google.cloud import vision
from google.cloud.vision import types

print(vision)

client = vision.ImageAnnotatorClient()
file_name = os.path.abspath('text.jpg')

with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

#response = client.label_detection(image=image)
response = client.text_detection(image=image)

#print(response)

texts = response.face_annotations


likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                   'LIKELY', 'VERY_LIKELY')

