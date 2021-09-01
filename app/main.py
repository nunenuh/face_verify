from typing import Any, Generator, List

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import UUID4
from sqlalchemy.orm import Session
from starlette.status import HTTP_201_CREATED, HTTP_404_NOT_FOUND

from fastapi import FastAPI, File, UploadFile

import io
from starlette.responses import Response


import cv2 as cv
import numpy as np
from PIL import Image

from .face import idcard_face_verify


from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, Depends, HTTPException
from .auth import AuthHandler
from .schemas import AuthDetails

# from . import actions, schemas
# from .db.session import SessionLocal

# Create all tables in database.
# Comment this out if you using migrations.
# models.Base.metadata.create_all(bind=engine)

app = FastAPI()
app.mount("/upload_files", StaticFiles(directory="upload_files"), name="upload_files")

# auth_handler = AuthHandler()
# users = []


# @app.post('/register', status_code=201)
# def register(auth_details: AuthDetails):
#     if any(x['username'] == auth_details.username for x in users):
#         raise HTTPException(status_code=400, detail='Username is taken')
#     hashed_password = auth_handler.get_password_hash(auth_details.password)
#     users.append({
#         'username': auth_details.username,
#         'password': hashed_password    
#     })
#     return

# @app.post('/login')
# def login(auth_details: AuthDetails):
#     user = None
#     for x in users:
#         if x['username'] == auth_details.username:
#             user = x
#             break
    
#     if (user is None) or (not auth_handler.verify_password(auth_details.password, user['password'])):
#         raise HTTPException(status_code=401, detail='Invalid username and/or password')
#     token = auth_handler.encode_token(user['username'])
#     return { 'token': token }


@app.get("/")
def index() -> Any:
    return {"message": "gai"}


@app.post("/upload/")
# async def create_upload_file(file: UploadFile = File(...), username=Depends(auth_handler.auth_wrapper)):
async def create_upload_file(request: Request, file: UploadFile = File(...), ):
    contents = await file.read()
    img = convert_buffer_to_image(contents)
    verify, faces, faces_link, img, img_link = idcard_face_verify(img, request)
    
    return {
        'results': {
            'verification': verify,
            'faces':faces_link,
            'image': img_link,
        },
    }

def convert_buffer_to_image(contents):
    nparr = np.fromstring(contents, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return img