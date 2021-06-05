#!/usr/bin/env python3

import os
import uuid
import shutil
import cv2
import math
import time
import numpy as np
import tensorflow as tf
import keras.backend as K
from fastapi import FastAPI, UploadFile, File
from keras.preprocessing import image
from pyzbar.pyzbar import decode


##################### Utility #####################
def angle_difference(x, y):
    return 180 - abs(abs(x - y) - 180)

def angle_error(y_true, y_pred):
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))

def rotate(image, angle):
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def largest_rotated_rect(w, h, angle):
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def crop_largest_rectangle(image, angle, height, width):
    return crop_around_center(
        image,
        *largest_rotated_rect(
            width,
            height,
            math.radians(angle)
        )
    )

###################################################


def custom_res(condition, mes, data, time_exec):
    return {'success': condition, 'message': mes, 'data': data, 'time_exec': time_exec}


app = FastAPI()
model = tf.keras.models.load_model('rotnet_barcode_view_resnet50_v2.hdf5', custom_objects={'angle_error': angle_error})
angle_class = {k: -k for k in range(360)}

@app.get('/')
def get_root():
    return {'message': 'Welcome to the barcode angle detection API'}

@app.post("/predict")
def read_root(barcode_image: UploadFile = File(...)):
    base_dir = 'tmp'

    start_time = time.time()
    generate_uuid = uuid.uuid4().hex
    file_name_uuid = '{}_{}'.format(generate_uuid, barcode_image.filename)
    path = '{}/{}'.format(base_dir, file_name_uuid)
    with open(path, 'wb') as img:
        shutil.copyfileobj(barcode_image.file, img)

    img = image.load_img(path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    w, h = img_arr.shape[0], img_arr.shape[1]
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.vstack([x])
    result = model.predict(x)
    max_prob = max(result[0])
    class_index = list(result[0]).index(max_prob)
    angle = angle_class[class_index]

    # restore the image
    restored_img = rotate(img_arr, angle)
    restored_img = crop_largest_rectangle(restored_img, angle, w, h)

    exec_time = str(time.time() - start_time) + ' seconds'

    # read the barcode
    raw = decode(restored_img)
    data = None
    if len(raw):
        for el in raw:
            data = el.data.decode('ascii')
        return custom_res(True, file_name_uuid, data, exec_time)
    
    os.remove(path)
    
    return custom_res(False, file_name_uuid, data, exec_time)
