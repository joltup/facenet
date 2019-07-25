import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename ='./clean_mobilenet.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

    LOGDIR='/tmp/tensorboard-cleanmobilefail'
    train_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
    train_writer.add_graph(sess.graph)
print('done?')
