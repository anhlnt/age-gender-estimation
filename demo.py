from pathlib import Path
import cv2
import numpy as np
import argparse
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from src.factory import get_model
import os
import time
import match_face
import json
from datetime import datetime
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.compat.v1.saved_model import tag_constants, signature_constants
from tensorflow.python.framework import convert_to_constants

def get_faceid():
    with open('FACE_ID', 'r') as f:
        return int(f.read())

def write_faceid(FACE_ID):
    with open('FACE_ID', 'w') as f:
        f.write(str(FACE_ID))


FACE_ID = get_faceid()

pretrained_model = "https://github.com/anhlnt/age-gender-estimation/releases/download/0.1/EfficientNetB3_224_weights.26-3.15.hdf5"
modhash = '7f195bc97a0aa9418b4f97fa95a54658'


def convert_to_tensorrt():
    input_saved_model_dir = "pretrained_models/EfficientNetB3_224_weights.26-3.15"
    output_saved_model_dir = "pretrained_models/TensorRT/EfficientNetB3_224_weights.26-3.15"


    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(
        max_workspace_size_bytes=(1<<32))
    conversion_params = conversion_params._replace(precision_mode="FP16")
    conversion_params = conversion_params._replace(
        maximum_cached_engines=100)
    conversion_params = conversion_params._replace(
      max_batch_size=8)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        conversion_params=conversion_params)
    converter.convert()
    print("[LOG] Successfully converted")
    def my_input_fn():
        # Input for a single inference call, for a network that has two input tensors:
        inp1 = np.random.normal(size=(1, 224, 224, 3)).astype(np.float32)
        # inp2 = np.random.normal(size=(8, 16, 16, 3)).astype(np.float32)
        yield inp1
    converter.build(input_fn=my_input_fn)
    print("[LOG] Successfully build")
    converter.save(output_saved_model_dir)
    print("[LOG] Successfully saved TensorRT Model")

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))

class Location():
    def __init__(self, startX, startY, endX, endY):
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY
    
    def top(self):
        return self.startY
    
    def bottom(self):
        return self.endY

    def left(self):
        return self.startX
    
    def right(self):
        return self.endX

    def width(self):
        return self.endX - self.startX
    
    def height(self):
        return self.endY - self.startY

def detect_mask(frame, faceNet, maskNet=None):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                              (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	locs = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
		# if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			locs.append(Location(startX, startY, endX, endY))

	return locs

def write_result(data, json_file):
    if len(data) > 0:
        with open(json_file, 'a') as f:
            f.write(json.dumps(data) + "\n")


def main():
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


    args = get_args()
    weight_file = args.weight_file
    margin = args.margin
    image_dir = args.image_dir

    output_saved_model_dir = "pretrained_models/TensorRT/EfficientNetB3_224_weights.26-3.15"
    
    if not weight_file:
        weight_file = get_file("EfficientNetB3_224_weights.26-3.15.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    print('model_name: ', model_name, 'img_size: ', img_size)
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    # model = get_model(cfg)
    # model.load_weights(weight_file)

    convert_to_tensorrt()
    saved_model_loaded = tf.saved_model.load(
        output_saved_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(
        graph_func)

    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()
    start = time.time()
    hold_face_start = time.time()
    faces_match = []
    face_id = FACE_ID

    for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        detected = detect_mask(img, faceNet)
        faces = np.empty((len(detected), img_size, img_size, 3))
        

        faces_cur = []
        faces_detect = []
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                faces_detect.append(img[y1:y2 + 1, x1:x2 + 1])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

            # predict ages and genders of the detected faces
            # results = model.predict(faces)
            results = frozen_func(tf.constant(faces.astype(np.float32)))

            predicted_genders = results[1].numpy()
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[0].numpy().dot(ages).flatten()

            
            # draw results
            for i, d in enumerate(detected):
                age = int(predicted_ages[i])
                gender = 0 if predicted_genders[i][0] < 0.5 else 1
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "Male" if predicted_genders[i][0] < 0.5 else "Female")
                draw_label(img, (d.left(), d.top()), label)
                
                match_id = match_face.match(faces_detect[i], faces_match, faces_cur, face_id, age, gender)
                if match_id < 0:
                    match_id = face_id
                    face_id += 1
                    write_faceid(face_id)
                draw_label(img, (d.left(), d.bottom()), "id: " + str(match_id))

        result = []
        timestamp = int(time.time())
        for face, idx, age, gender in faces_cur:
            data = {}
            data["user_id"] = idx
            data["user_age"] = age
            data["user_gender"] = gender
            data["start_time"] = timestamp
            data["end_time"] = timestamp
            result.append(data)
        
        if not os.path.exists('result'):
            os.mkdir('result')
        json_file = 'result/' + datetime.fromtimestamp(timestamp).strftime('%Y%m%d') + '.log'
        write_result(result, json_file)
        if time.time() - hold_face_start > 5:
            faces_match = faces_cur
            hold_face_start = time.time()
        else:
            faces_match += faces_cur

        draw_label(img, (50, 50), "{:.2f}fps".format(1.0 / (time.time() - start)))
        cv2.imshow("result", img)
        key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

        if key == 27:  # ESC
            break
        start = time.time()


if __name__ == '__main__':
    main()
