from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import listdir
from os.path import join
import cv2
import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


def encoding():
    meta_file = './data/meta/classes.txt'
    mapping = {}
    with open(meta_file, 'r') as txt:
        classes = [l.strip() for l in txt.readlines()]
    for i, class_name in enumerate(classes):
        mapping[class_name] = i
    print(mapping)
    return mapping


def load_images(path):
    all_imgs = []
    all_classes = []
    for i, subdir in enumerate(listdir(path)):
        imgs = listdir(join(path, subdir))
        print(subdir)
        for img_name in imgs:
            img = cv2.imread(join(path, subdir, img_name), cv2.IMREAD_COLOR)
            norm_image = img
            norm_image = cv2.normalize(img, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            all_imgs.append(norm_image)
            all_classes.append(mapping[subdir])
    return np.array(all_imgs, dtype=np.float32), np.array(all_classes, dtype=np.int32)


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    norm1 = tf.nn.local_response_normalization(input=pool1, depth_radius=5, alpha=0.0001, beta=0.75)

    conv2 = tf.layers.conv2d(
        inputs=norm1,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    norm2 = tf.nn.local_response_normalization(input=pool2, depth_radius=5, alpha=0.0001, beta=0.75)

    conv3 = tf.layers.conv2d(
        inputs=norm2,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1)

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1)

    # pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        strides=1)

    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool5, [-1, 16 * 16 * 256])

    dense = tf.layers.dense(inputs=pool2_flat, units=4096, activation=tf.nn.relu)

    dense2 = tf.layers.dense(inputs=dense, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense3 = tf.layers.dense(inputs=dropout, units=100, activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(inputs=dense3, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout2, units=25)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilites": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=25)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)


def main(unused_agv):
    x_test, y_test = load_images('./data/test')
    print(x_test.shape)
    print(y_test.shape)
    x_train, y_train = load_images('./data/train')
    groc_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./temp/groc_covnet_model1")
    tensors_to_log = {"probabilites": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1000)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=y_train,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    groc_classifier.train(
        input_fn=train_input_fn,
        steps=25000,
        hooks=[logging_hook])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    eval_results = groc_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    mapping = encoding()
    tf.app.run()
