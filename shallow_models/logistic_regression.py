"""
A logistic regression algorithm implementation class
library: tensorflow
data set: mnist
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class LR_tf:

    def __init__(self):
        self.mnist = input_data.read_data_sets('/home/qwe/PycharmProjects/cg2vec/data/', one_hot=True)
        self.learning_rate = 0.01
        self.training_epochs = 25
        self.batch_size = 100
        self.display_step = 1

    def init_graph(self):
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])
        self.x = x
        self.y = y

        W = tf.Variable(tf.zeros([784,10]))
        b = tf.Variable(tf.zeros([10]))

        # construct model
        pred = tf.nn.softmax(tf.matmul(x, W) + b)
        self.pred = pred

        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        self.optimizer = optimizer
        self.cost = cost

        init = tf.global_variables_initializer()
        return init

    def train(self):
        with tf.Session() as sess:
            init = self.init_graph()
            sess.run(init)

            for epoch in range(self.training_epochs):
                avg_cost = 0
                total_batch = int(self.mnist.train.num_examples / self.batch_size)

                for i in range(total_batch):
                    batch_xs, batch_ys = self.mnist.train.next_batch(self.batch_size)
                    _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x:batch_xs, self.y:batch_ys})
                    avg_cost += c / total_batch

                # display
                if (epoch+1) % self.display_step == 0:
                    print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))

            print('Optimization Finished')

    def test(self):
        with tf.Session() as sess:
            correct_prediction = tf.equal(tf.arg_max(self.pred, 1), tf.arg_max(self.y, 1))
            # caculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print('Accuracy:', accuracy.eval({self.x:self.mnist.test.images, self.y:self.mnist.test.labels}))

def main():
    lr = LR_tf()
    lr.init_graph()
    lr.train()
    lr.test()

main()