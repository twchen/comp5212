import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from tensorflow.contrib.layers import xavier_initializer


import shutil
import os

from sklearn.model_selection import train_test_split
import time

NUM_LABELS = 47
rnd = np.random.RandomState(123)
tf.set_random_seed(123)



def cnn_model_fn(features, labels, mode, params):
    if params == None:
        learning_rate = 0.001
        momentum = 0.9
    else:
        learning_rate = params['learning_rate']
        momentum = params['momentum']
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
    conv_layers = [
        [32, 3, 1],
        [32, 5, 2],
        [64, 3, 1],
        [64, 5, 2]
    ]
    curr_layer = input_layer
    for i, [f, k, s] in enumerate(conv_layers, start=1):
        curr_layer = tf.layers.conv2d(
            inputs=curr_layer,
            filters=f,
            kernel_size=k,
            strides=s,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=xavier_initializer(seed=i),
            name=f'conv{i}'
        )
    conv_flat = tf.reshape(curr_layer, [-1, np.prod(curr_layer.shape[1:])])
    dense = tf.layers.dense(
        inputs=conv_flat,
        units=1024,
        activation=tf.nn.relu
    )
    logits = tf.layers.dense(
        inputs=dense,
        units=NUM_LABELS
    )
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # ================== my momentum optimizer ===============
        xs = tf.trainable_variables()
        prev_deltas = [
            tf.Variable(initial_value=tf.zeros(shape=x.shape, dtype=x.dtype), trainable=False)
            for x in xs
        ]
        grads = tf.gradients(loss, xs)
        train_ops = []
        for x, grad, prev_delta in zip(xs, grads, prev_deltas):
            curr_delta = momentum * prev_delta - learning_rate * grad
            train_ops.append(tf.assign_add(x, curr_delta))
            train_ops.append(tf.assign(prev_delta, curr_delta))
        global_step = tf.train.get_global_step()
        train_ops.append(tf.assign_add(global_step, tf.constant(1, dtype=global_step.dtype)))
        train_op = tf.group(train_ops)
        # ==============================================================
        # tensorflow momentum optimizer
#         optimizer = tf.train.MomentumOptimizer(momentum=momentum, learning_rate=learning_rate)
#         train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def train_eval_cnn(X_train, y_train, X_eval, y_eval,
                   learning_rate, momentum, batch_size=64, steps=10000):
    model_dir = f'cnn_model_{momentum}_{learning_rate}'
    # remove old models
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    params = {
        'learning_rate': learning_rate,
        'momentum': momentum
    }
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_dir,
        params=params)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': X_train},
        y=y_train,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': X_eval},
        y=y_eval,
        num_epochs=1,
        shuffle=False)
    trainset_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': X_train},
        y=y_train,
        num_epochs=1,
        shuffle=False)
    start_ts = time.time()
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=steps)
    end_ts = time.time()
    trainset_score = mnist_classifier.evaluate(input_fn=trainset_eval_input_fn)
    eval_score = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(f'Training time in seconds: {end_ts - start_ts:.4f}')
    print('Performance on training set: ')
    print(trainset_score)
    print('Performance on validation/test set: ')
    print(eval_score)
    return mnist_classifier, eval_score['accuracy']

def cae_model_fn(features, labels, mode, params):
    if params == None:
        learning_rate = 0.001
    else:
        learning_rate = params['learning_rate']
    X_origin = tf.reshape(features['x'], [-1, 28, 28, 1])
    
    conv1 = tf.layers.conv2d(
        inputs=X_origin,
        filters=32,
        kernel_size=5,
        strides=2,
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=xavier_initializer(seed=0),
        name='conv1'
    )
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=5,
        strides=2,
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=xavier_initializer(seed=1)
    )
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=2,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=xavier_initializer(seed=2)
    )
    deconv1 = tf.layers.conv2d_transpose(
        inputs=conv3,
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=xavier_initializer(seed=3)
    )
    deconv2 = tf.layers.conv2d_transpose(
        inputs=deconv1,
        filters=32,
        kernel_size=5,
        strides=2,
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=xavier_initializer(seed=4)
    )
    deconv3 = tf.layers.conv2d_transpose(
        inputs=deconv2,
        filters=1,
        kernel_size=5,
        strides=2,
        padding='same',
        kernel_initializer=xavier_initializer(seed=5)
    )
    
    X_reconstruct = deconv3
    predictions = {
        'predictions': X_reconstruct
    }
    loss = tf.losses.mean_squared_error(X_origin, X_reconstruct)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #optimizer = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

def train_eval_cae(X_train, X_eval, learning_rate, batch_size=64, steps=20000):
    model_dir = f'cae_model_{learning_rate}'
    # remove old models
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    
    mnist_cae = tf.estimator.Estimator(
        model_fn=cae_model_fn,
        model_dir=model_dir,
        params={'learning_rate': learning_rate})
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': X_train},
        batch_size=batch_size,
        num_epochs=None,
        shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': X_eval},
        num_epochs=1,
        shuffle=False)
    trainset_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': X_train},
        num_epochs=1,
        shuffle=False)
    start_ts = time.time()
    mnist_cae.train(
        input_fn=train_input_fn,
        steps=steps)
    end_ts = time.time()
    trainset_score = mnist_cae.evaluate(input_fn=trainset_eval_input_fn)
    eval_score = mnist_cae.evaluate(input_fn=eval_input_fn)
    print(f'Training time (seconds): {end_ts - start_ts:.4f}')
    print('Performance on training set: ')
    print(trainset_score)
    print('Performance on validation/test set: ')
    print(eval_score)
    return mnist_cae, eval_score['loss']

def test_cnn(X_test, y_test, momentum=0.9, learning_rate=0.001):
    # TODO: implement CNN testing
    model_dir = f'cnn_model_{momentum}_{learning_rate}'
    params = {
        'learning_rate': learning_rate,
        'momentum': momentum
    }
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_dir,
        params=params)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': X_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    score = mnist_classifier.evaluate(input_fn=test_input_fn)
    return score['accuracy']

def plot_feature_maps(kernels):
    fig, axes = plt.subplots(4, 8)
    fig.suptitle('Feature maps of the 1st layer')
    for i in range(0, 4):
        for j in range(0, 8):
            kernel = kernels[:,:,0, 8*i+j].reshape([5, 5])
            axes[i, j].imshow(kernel)
            axes[i, j].axis('off')
    plt.show()

def plot_reconstructed_images(X_eval, X_pred, n):
    fig, axes = plt.subplots(n, 4)
    fig.suptitle('Column 1,3: original images. Column 2, 4: reconstructed images')
    for i in range(0, n):
        axes[i, 0].imshow(X_eval[2*i], cmap='gray')
        axes[i, 1].imshow(X_pred[2*i], cmap='gray')
        axes[i, 2].imshow(X_eval[2*i+1], cmap='gray')
        axes[i, 3].imshow(X_pred[2*i+1], cmap='gray')
        for j in range(0, 4):
            axes[i, j].axis('off')
    plt.show()

def evaluate_ae(X_eval, learning_rate=0.001):
    model_dir = f'cae_model_{learning_rate}'
    model = tf.estimator.Estimator(
        model_fn=cae_model_fn,
        model_dir=model_dir,
        params={'learning_rate': learning_rate})
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': X_eval},
        num_epochs=1,
        shuffle=False)
    eval_res = model.evaluate(input_fn=eval_input_fn)
    print(f"Loss on the evaluation set: {eval_res['loss']:.4f}")
    pred_res = model.predict(input_fn=eval_input_fn)
    X_pred = [x['predictions'].reshape([28,28]) for x in pred_res]
    kernels = model.get_variable_value('conv1/kernel')
    plot_feature_maps(kernels)
    plot_reconstructed_images(X_eval, X_pred, 5)

def main():
    parser = argparse.ArgumentParser(description='COMP5212 Programming Project 2')
    parser.add_argument('--task', default="train_cnn", type=str,
                        help='Select the task, train_cnn, test_cnn, '
                             'train_ae, evaluate_ae, ')
    parser.add_argument('--datapath',default="./data",type=str, required=False,
                        help='Select the path to the data directory')
    args = parser.parse_args()
    datapath = args.datapath
    with tf.variable_scope("placeholders"):
        img_var = tf.placeholder(tf.uint8, shape=(None, 28, 28), name="img")
        label_var = tf.placeholder(tf.int32, shape=(None,), name="true_label")

    if args.task == "train_cnn":
        train_data = np.load('datasets/data_classifier_train.npz')
        test_data = np.load('datasets/data_classifier_test.npz')
        X_train = np.asarray(train_data['x_train'], dtype=np.float32) / 255
        y_train = np.asarray(train_data['y_train'], dtype=np.int32)
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test = np.asarray(test_data['x_test'], dtype=np.float32) / 255
        y_test = np.asarray(test_data['y_test'], dtype=np.int32)
        X_test, y_test = shuffle(X_test, y_test, random_state=1)
        # the test set is not used for training.
        # it is passed for evaluating the model on test set after training is completed.
        train_eval_cnn(X_train, y_train, X_test, y_test, learning_rate=0.01, momentum=0.9)

    elif args.task == "test_cnn":
        test_data = np.load('datasets/data_classifier_test.npz')
        X_test = np.asarray(test_data['x_test'], dtype=np.float32) / 255
        y_test = np.asarray(test_data['y_test'], dtype=np.int32)
        X_test, y_test = shuffle(X_test, y_test, random_state=1)
        accuracy = test_cnn(X_test, y_test, learning_rate=0.01, momentum=0.9)
        print("accuracy = {:.4f}\n".format(accuracy))

    elif args.task == "train_ae":
        train_data = np.load('datasets/data_autoencoder_train.npz')
        eval_data = np.load('datasets/data_autoencoder_eval.npz')
        RANDOM_STATE = 0
        X_train = np.asarray(shuffle(train_data['x_ae_train'], random_state=RANDOM_STATE), dtype=np.float32) / 255
        X_eval = np.asarray(shuffle(eval_data['x_ae_eval'], random_state=RANDOM_STATE), dtype=np.float32) / 255
        # the evaluation set is not used for training
        # it is passed to evaluate the model after the training is completed
        train_eval_cae(X_train, X_eval, 0.001)

    elif args.task == "evaluate_ae":
        eval_data = np.load('datasets/data_autoencoder_eval.npz')
        RANDOM_STATE = 0
        X_eval = np.asarray(shuffle(eval_data['x_ae_eval'], random_state=RANDOM_STATE), dtype=np.float32) / 255
        evaluate_ae(X_eval, learning_rate=0.001)

if __name__ == "__main__":
    main()
