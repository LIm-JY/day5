import io
import os

from google.cloud import vision
from google.cloud.vision import types

print(vision)

client = vision.ImageAnnotatorClient()
file_name = os.path.abspath('face2.jpg')

with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

#response = client.label_detection(image=image)
response = client.face_detection(image=image)

#print(response)

faces = response.face_annotations


likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                   'LIKELY', 'VERY_LIKELY')
print('Faces:')

for face in faces:
    print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
    print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
    print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

    vertices = (['({},{})'.format(vertex.x, vertex.y)
                 for vertex in face.bounding_poly.vertices])

    print('face bounds: {}'.format(','.join(vertices)))

