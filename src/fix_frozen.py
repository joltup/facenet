"""
Convert model.ckpt to model.pb
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

model_filename = '/Users/tyler/face-it/facenet-models/frozen_models/transformed_4_24_2018_mobilenet.pb'


with tf.Session() as sess:
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def)

        # fix batch norm nodes
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
    tf.train.write_graph(graph_def,
                         '/Users/tyler/face-it/facenet-models/frozen_models/',
                         'fixed_transformed_4_24_2018_mobilenet.pb', as_text=False)
