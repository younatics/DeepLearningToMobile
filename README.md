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

#### Part 2. Deep learning framwork to mobile machine learning framwork
| Framework | CoreML | TensorFlow Mobile | Tensorflow Lite |
| :-------: | :----: | :---------------: | :-------------: |
| Tensorflow | `tf-coreml` | `tensorflow` | `tensorflow` |
| Pytorch | `onnx` | ← | ← |
| Keras | `coremltools` | `tensorflow backend` | ← |
| Caffe | `coremltools` | `caffe-tensorflow` | ←  |


# Part 0. 
### Basic FFNN example
I'll use Golbin code in this [TensorFlow-Tutorials](https://github.com/golbin/TensorFlow-Tutorials/blob/master/04%20-%20Neural%20Network%20Basic/01%20-%20Classification.py)

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

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
b = tf.Variable(tf.zeros([3]))

L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)

model = tf.nn.softmax(L)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
```

# Part 1. 
### Deep learning model to mobile machine learning framwork
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

## TensorFlow Mobile

![tensorflowmobile](https://github.com/younatics/DeepLearningToMobile/blob/master/img/tensorflowmobile.png)

- ML Framework supported by Google, using .pb extension
- Support Java for Android, Objective-C++ for iOS




