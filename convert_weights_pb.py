# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import yolo_v4
import yolo_v4_tiny

from utils import load_weights, load_coco_names, detections_boxes, freeze_graph

FLAGS = tf.compat.v1.app.flags.FLAGS

tf.compat.v1.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.compat.v1.app.flags.DEFINE_string(
    'weights_file', 'yolov4.weights', 'Binary file with detector weights')
tf.compat.v1.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.compat.v1.app.flags.DEFINE_string(
    'output_graph', 'frozen_darknet_yolov4_model.pb', 'Frozen tensorflow protobuf model output path')

tf.compat.v1.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv4')
tf.compat.v1.app.flags.DEFINE_integer(
    'size', 416, 'Image size')



def main(argv=None):
    if FLAGS.tiny:
        model = yolo_v4_tiny.yolo_v4_tiny
    else:
        model = yolo_v4.yolo_v4

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    inputs = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3], "inputs")

    with tf.compat.v1.variable_scope('detector'):
        detections = model(inputs, len(classes), data_format=FLAGS.data_format)
        load_ops = load_weights(tf.compat.v1.global_variables(scope='detector'), FLAGS.weights_file)

    # Sets the output nodes in the current session
    boxes = detections_boxes(detections)

    with tf.compat.v1.Session() as sess:
        sess.run(load_ops)
        freeze_graph(sess, FLAGS.output_graph)

if __name__ == '__main__':
    tf.compat.v1.app.run()
