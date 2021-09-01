
from deepface import DeepFace
from mtcnn.mtcnn import MTCNN
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import uuid

def idcard_face_verify(img, request, backend='mtcnn', pad=10, ):
    base_uuid= f'{uuid.uuid1().hex[:10]}'
    
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    detected_faces = MTCNN().detect_faces(img)
    
    fname = f'upload_files/{base_uuid}.jpg'
    img_pil = Image.fromarray(img)
    img_pil.convert("RGB")
    img_pil.save(fname)
    img_link = f'{request.base_url}{fname}'
    
    faces = []
    for face in detected_faces:
        x,y,w,h = face['box']
        f = img[y-pad:y+h+pad, x-pad:x+w+pad]
        faces.append(f)
    if len(faces)>1:
        verify = DeepFace.verify(faces[0], faces[1], enforce_detection=False, detector_backend = backend)
        # faces = [Image.fromarray(f).convert("RGB").tobytes() for f in faces]
        faces_link = []
        for idx, f in enumerate(faces):
            fname = f'upload_files/{base_uuid}_face{idx+1}.jpg'
            im = Image.fromarray(f)
            im.convert("RGB")
            im.save(fname)
            
            link = f'{request.base_url}{fname}'
            faces_link.append(link)
            
        return verify, faces, faces_link, img, img_link
    else:
        return None