from keras.models import load_model
from tensorflow.python.framework import graph_util
import tensorflow as tf
import numpy as np
import os
import re
from tensorflow.python.platform import gfile
import time
import pprint
import tensorflow.contrib.slim as slim
import models.mobilenet


pp = pprint.PrettyPrinter(indent=4)

image_size = 160


# python3 ~/python-venvs/gpu-tensorflow/tensorflow/tensorflow/python/tools/freeze_graph.py \
# --input_graph=/Users/tyler/face-it/models/sn-20180204-160909/20180204-16090.pb \
# --input_checkpoint=/Users/tyler/face-it/models/sn-20180204-160909/model-20180204-160909.ckpt-266000 \
# --output_graph=/tmp/frozen_graph.pb \
# --output_node_names=embeddings
def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        start = time.time()
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
        end = time.time()
        print('model restored in: %0.2f', (end - start))

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def clean_graph_for_freezing(graph):
    graph_def = graph.as_graph_def()
    for node in graph_def.node:
        print('node name: {}', node.name)
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
    return graph_def


if __name__ == '__main__':

    embedding_size = 256
    keep_probability = 0.8
    weight_decay = .00005

    session = tf.InteractiveSession()
    load_model('/Users/tyler/face-it/facenet-models/20180423-173901/')
    # layers = tf.get_default_graph().get_operations()

    original_graph = tf.get_default_graph()
    original_variables = tf.trainable_variables()
    original_variable_values = session.run(original_variables)
    original_variable_names = map(lambda x: x.name, original_variables)
    original_variable_dict = dict(zip(original_variable_names, original_variable_values))

    new_graph = tf.Graph()
    with new_graph.as_default() as save_graph:
        input_image = tf.placeholder(tf.float32, shape=(1, 160, 160, 3), name="input")
        phase_train_placeholder = tf.constant(False, dtype=tf.bool, name='phase_train')
        prelogits, _ = models.mobilenet.inference(
                                           input_image,
                                           0.2,
                                           phase_train=phase_train_placeholder,
                                           bottleneck_layer_size=embedding_size,
                                           weight_decay=weight_decay)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        new_variables = tf.trainable_variables()
        for new_variable in new_variables:
            print('setting {}', new_variable.name)
            tf.assign(new_variable, original_variable_dict[new_variable.name])

        # whitelist = [
        #     'phase_train',
        #     'input',
        #     'embeddings/Square',
        #     'embeddings/Sum/reduction_indices',
        #     'embeddings/Sum',
        #     'embeddings/Maximum/y',
        #     'embeddings/Maximum',
        #     'embeddings/Rsqrt',
        #     'embeddings',
        # ]
        save_graph_def = clean_graph_for_freezing(save_graph)
        output_graph_def = graph_util.convert_variables_to_constants(
            session,
            save_graph_def,
            ['embeddings'],
            # variable_names_whitelist=whitelist
        )

        output_file = './clean_mobilenet_no_batching.pb'

        with tf.gfile.GFile(output_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node), output_file))

        # tf.train.write_graph(save_graph_def, '/Users/tyler/face-it/facenet-models/frozen_models/4_24_2018_mobilenet_checkpointed/', 'mobilenet.pb', as_text=False)

        # all_saver = tf.train.Saver()
        # all_saver.save(session, '/Users/tyler/face-it/facenet-models/frozen_models/4_24_2018_mobilenet_checkpointed/mobilenet', global_step=1, write_meta_graph=True)

        # log summary of graph
        # LOGDIR='/tmp/tensorboard-new-input'
        # writer = tf.summary.FileWriter(LOGDIR, save_graph)
        # writer.close()

    print('completed')
