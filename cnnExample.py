import tensorflow as tf

# x1 = tf.constant([5,6,7])
# x2 = tf.constant([6,7,8])
#
# result = x1 * x2
#
# with tf.Session() as sess:
#     print (sess.run(result))

#Session object encapsulates
'''
input > weight>hidden layer 1 (activation function) > weights > hidden l 2
(activation function) > weights > output layer

compare output to intended output > cost or loss function (cross entropy)
optimization function (optimizer) > minimize cost (Adam Optimizer ... SGD, AdaGrad)

backpropagation
feed forward + backprop = epoch
'''

from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


mnist = input_data.read_data_sets("/tmp/data", one_hot=True)



n_classes = 10
batch_size = 128

# height x width
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, shape=[None, n_classes])

def conv2d(x,W):
    return tf.nn.conv2d(x, W , strides=[1,1,1,1],padding = "SAME")

def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding = "SAME")

def convolutional_neural_network_model(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([7*7*64,1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'B_conv1': tf.Variable(tf.random_normal([ 32])),
               'B_conv2': tf.Variable(tf.random_normal([64])),
               'B_fc': tf.Variable(tf.random_normal([1024])),
               'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x,shape=[-1,28,28,1])

    conv1 = tf.nn.relu(conv2d(x,weights['W_conv1']) + biases['B_conv1'])

    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['B_conv2'])

    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1,7*7*64])
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['B_fc'])

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output



def train_neural_network(x):
    prediction = convolutional_neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # Cycles of feed forward and back propagation
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x ,epoch_y = mnist.train.next_batch(batch_size)
                _, c =sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

train_neural_network(x)




