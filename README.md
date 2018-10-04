# Deep Learning To Mobile
### Curated way to convert deep learning model to mobile. 

This repository will show you how to put your own model directly into mobile(iOS/Android) with basic example. First part is about **deep learning model to mobile machine learning framwork**, and second part is about **deep learning framwork to mobile machine learning framwork**

## Intro

#### Part 1. Deep learning model to mobile machine learning framwork

| Neural Network | CoreML | TensorFlow Mobile | Tensorflow Lite |
| :-: | :---: | :---------------: | :-------------: |
| Feedforward NN | ✔️ | ✔️ | ✔️ |
| Convolutional NN | ✔️ | ✔️ | ✔️ |
| Recurrent NN | ✔️ | ✔️ | ❗️ |

#### Part 2. Deep learning framework to mobile machine learning framework
| Framework | CoreML | TensorFlow Mobile | Tensorflow Lite |
| :-------: | :----: | :---------------: | :-------------: |
| Tensorflow | `tf-coreml` | `tensorflow` | `tensorflow` |
| Pytorch | `onnx` | ← | ← |
| Keras | `coremltools` | `tensorflow backend` | ← |
| Caffe | `coremltools` | `caffe-tensorflow` | ←  |


# Part 0. 
### Basic FFNN example
I'll use Golbin code in this [TensorFlow-Tutorials](https://github.com/golbin/TensorFlow-Tutorials/blob/master/04%20-%20Neural%20Network%20Basic/02%20-%20Deep%20NN.py)

```python
import tensorflow as tf
import numpy as np

x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

y_data = np.array([
    [1, 0, 0],  
    [0, 1, 0],  
    [0, 0, 1],  
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

global_step = tf.Variable(0, trainable=False, name='global_step')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
b = tf.Variable(tf.zeros([3]))

L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)

model = tf.nn.softmax(L)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables())
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
        tf.train.write_graph(sess.graph_def, '.', './model/FFNN.pbtxt')  
        saver.save(sess, './model/FFNN.ckpt', global_step=global_step)
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
```

# Part 1. 
### Deep learning model to mobile machine learning framework
## CoreML

![CoreML](https://github.com/younatics/DeepLearningToMobile/blob/master/img/coreml.png)

- ML Framework supported by Apple, using .mlmodel extension
- Automatically generated wrapper for iOS(Swift or Objective-C)
- | Neural Network | CoreML |
  | :-: | :---: |
  | Feedforward NN | ✔️ |
  | Convolutional NN | ✔️ |
  | Recurrent NN | ✔️ |

### REFERENCE
- [Core ML](https://developer.apple.com/documentation/coreml)
- [Converting Trained Models to Core ML](https://developer.apple.com/documentation/coreml/converting_trained_models_to_core_ml)

## TensorFlow Mobile🔒
### TensorFlow Mobile is now deprecated
![tensorflowmobile](https://github.com/younatics/DeepLearningToMobile/blob/master/img/tensorflowmobile.png)

- ML Framework supported by Google, using .pb extension
- Support Java for Android, Objective-C++ for iOS
- | Neural Network | TensorFlow Mobile |
  | :-: | :---: |
  | Feedforward NN | ✔️ |
  | Convolutional NN | ✔️ |
  | Recurrent NN | ✔️ |
  
### Reference
- [TensorFlow Mobile](https://www.tensorflow.org/lite/tfmobile/)
- [TensorFlow on Mobile: Tutorial](https://towardsdatascience.com/tensorflow-on-mobile-tutorial-1-744703297267)

## TensorFlow Lite

- ML Framework supported by Google, using .tflite extension
- Support Java for Android, Objective-C++ for iOS
- **Recommand way** by Google to use tensorflow in Mobile
- | Neural Network | TensorFlow Mobile |
  | :-: | :---: |
  | Feedforward NN | ✔️ |
  | Convolutional NN | ✔️ |
  | Recurrent NN | RNN is not supported see more information in [this link](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/tf_ops_compatibility.md) |

### Reference
- [TensorFlow Lite](https://www.tensorflow.org/lite/)
- [TensorFlow Lite & TensorFlow Compatibility Guide](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/tf_ops_compatibility.md)

# Part 2. 
### Deep learning framework to mobile machine learning framwork

## TensorFlow to Tensorflow Mobile
We get `FFNN.pbtxt`and `FFNN.ckpt-90` in Part 0 code.
Freeze graph using `freeze_graph` from `tensorflow.python.tools`

```python
from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph("model/FFNN.pbtxt", "",
                          "", "model/FFNN.ckpt-90", "Output",
                          "", "",
                          "FFNN_frozen_graph.pb", True, "")
```
Now you can use `FFNN_frozen_graph.pb` in TensorFlow Mobile!



