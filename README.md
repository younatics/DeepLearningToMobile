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
X = tf.placeholder(tf.float32, name='Input')
Y = tf.placeholder(tf.float32, name='Output')

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name='W1')
    b1 = tf.Variable(tf.zeros([10]), name='b1')
    L1 = tf.add(tf.matmul(X, W1), b1, name='L1')
    L1 = tf.nn.relu(L1)
    
with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.), name='W2')
    b2 = tf.Variable(tf.zeros([3]), name='b2')
    model = tf.add(tf.matmul(L1, W2), b2, name='model')
    prediction = tf.argmax(model, 1, name='prediction')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model), name='cost')
optimizer = tf.train.AdamOptimizer(learning_rate=0.01, name='optimizer')
train_op = optimizer.minimize(cost, global_step=global_step)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver(tf.global_variables())
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

for step in range(30):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 30 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
        tf.train.write_graph(sess.graph_def, '.', './model/FFNN.pbtxt')  
        saver.save(sess, './model/FFNN.ckpt', global_step=global_step)
        break
        
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

## Check your Tensor graph
You have to check frozen tensor graph

```python
def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph
    
graph = load_graph('FFNN_frozen_graph.pb')

for op in graph.get_operations():
    print(op.name)
```

### Reference
- [Graphs and Sessions](https://www.tensorflow.org/guide/graphs)

## TensorFlow to CoreML (iOS)
### `tf-coreml` is the recommended way from Apple to convert tensorflow to CoreML

```python
import tfcoreml

mlmodel = tfcoreml.convert(
        tf_model_path = 'FFNN_frozen_graph.pb',
        mlmodel_path = 'FFNN.mlmodel',
        output_feature_names = ['layer2/prediction:0'],
        input_name_shape_dict = {'Input:0': [1, 2]})
```
Now you can use `FFNN.mlmodel` in iOS project! 

### Reference
- [tf-coreml](https://github.com/tf-coreml/tf-coreml)

## TensorFlow to TensorFlow Lite (Android)
### `toco` is the recommended way from Google to convert TensorFlow to TensorFlow Lite

```python
import tensorflow as tf

graph_def_file = "FFNN_frozen_graph.pb"
input_arrays = ["Input"]
output_arrays = ["layer2/prediction"]

converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("FFNN.tflite", "wb").write(tflite_model)
```
Now you can use `FFNN.tflite` in Android project! 

### Reference
- [Toco](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/toco)
- [Intro to Machine Learning on Android — How to convert a custom model to TensorFlow Lite](https://heartbeat.fritz.ai/intro-to-machine-learning-on-android-how-to-convert-a-custom-model-to-tensorflow-lite-e07d2d9d50e3)

