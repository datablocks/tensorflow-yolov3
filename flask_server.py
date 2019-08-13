#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
from flask import Flask, request, Response
from io import BytesIO
import base64
import jsonpickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--frozen", help="input checkpoint file", default="yolov3_test_loss=2.1426.ckpt-lre_1e-7")
parser.add_argument("--classcount", type=int, help="number of classes", default=32)
flag = parser.parse_args()

# Initialize the Flask application
app = Flask(__name__)


return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./" + flag.frozen #"./yolov3_coco_flickr_lre_1e-7.pb"#yolov3_coco_flickr_lre_5e-7_2.pb"
video_path      = "./docs/images/road.mp4"  #/home/andy/tensorjupyter/tensorflow-yolov3
# video_path      = 0
num_classes     = flag.classcount #32
input_size      = 736 #736 #608 #416
noflow = False
try:
    tf
    graph           = tf.Graph()
    return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)
except NameError:
    print("ignoring tf libraries missing")

def readb64(base64_string):
    return Image.open(BytesIO(base64.b64decode(base64_string)))

    # sbuf = StringIO()
    # tstr = base64.b64decode(base64_string)
    # sbuf.write(tstr)
    # return Image.open(sbuf)
    # pimg = Image.open(sbuf)
    # return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

# vid = cv2.VideoCapture(video_path)
# return_value, frame = vid.read()
# print(return_value)
# if return_value:
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image = Image.fromarray(frame)

# print(frame.shape[:2])
# print(type(frame.shape[:2]))
# print(type(frame))

with tf.Session(graph=graph) as sess:

    @app.route('/predict', methods=['POST'])
    def test():


        

        r = request

        jsonData = request.get_json()
        # print(r)
        # print(r.files)
        # print(jsonData)
        if not (jsonData.get('inputs')):
            response = {'message': 'missing inputs'}
            return Response(response=jsonpickle.encode(response), status=200, mimetype="application/json")
        
        inputs = jsonData['inputs']

        outputs = {
            "detection_scores": [],
            "detection_boxes": [],
            "detection_classes": [],
            "coordmode": "actual"

        }
        # outputs = []
        imageIndex = 0
        for ipt in inputs:
            if not (ipt.get('b64')):
                return Response(response=jsonpickle.encode({'message': 'missing b64 in input'}), status=200, mimetype="application/json")
            
            outputs["detection_scores"].append([])
            outputs["detection_boxes"].append([])
            outputs["detection_classes"].append([])

            # print(type(ipt['b64']))
            # print(type('b64'))
            # print(ipt['b64'])
            limage = readb64(ipt['b64'])

            # print('data')
            # print(type(limage))
            frame = np.array(limage)
            # print(type(frame))
            # print(limage.size)
            # print(type(limage.size))
            frame_size = frame.shape[:2]
            image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]
            prev_time = time.time()

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={ return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')  # bboxes, tried 0.35, method='nms'
            oimage = utils.draw_bbox(frame, bboxes)

            #assemble into tensorflow serving output format
            for i, bbox in enumerate(bboxes):
                print(bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],bbox[5])
                # print( type(bbox[0]) )
                # print(bbox[0].dtype)
                outputs["detection_scores"][imageIndex].append(bbox[4].item())
                outputs["detection_boxes"][imageIndex].append([bbox[1].item(),bbox[0].item(),bbox[3].item(),bbox[2].item()])
                outputs["detection_classes"][imageIndex].append(bbox[5].item())
                

                
            timage = Image.fromarray(oimage)
            timage.save( 'toutput.png', 'PNG' )
            # outputs += bboxes
            # change format to tensorflow serving style

            imageIndex = imageIndex+1
            

        response = {'message': 'success', 'outputs':outputs}

        # encode response using jsonpickle
        return Response(response=jsonpickle.encode(response), status=200, mimetype="application/json")

# with tf.Session(graph=graph) as sess:
#     vid = cv2.VideoCapture(video_path)
#     while True:
#         return_value, frame = vid.read()
#         if return_value:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(frame)
#         else:
#             raise ValueError("No image!")
#         frame_size = frame.shape[:2]
#         image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
#         image_data = image_data[np.newaxis, ...]
#         prev_time = time.time()

#         pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
#             [return_tensors[1], return_tensors[2], return_tensors[3]],
#                     feed_dict={ return_tensors[0]: image_data})

#         pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
#                                     np.reshape(pred_mbbox, (-1, 5 + num_classes)),
#                                     np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

#         bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
#         bboxes = utils.nms(bboxes, 0.45, method='nms')
#         image = utils.draw_bbox(frame, bboxes)

#         curr_time = time.time()
#         exec_time = curr_time - prev_time
#         result = np.asarray(image)
#         info = "time: %.2f ms" %(1000*exec_time)
#         cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
#         result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         cv2.imshow("result", result)
#         if cv2.waitKey(1) & 0xFF == ord('q'): break





# start flask app
    app.run(host="0.0.0.0", port=8501)