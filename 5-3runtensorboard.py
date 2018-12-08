import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


batch_size = 128
n_batch = mnist.train.num_examples // batch_size

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(var-mean))))
        tf.summary.scalar('stddev',stddev) #标准差
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
#命名空间
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None,784],name='x-input')
    y = tf.placeholder(tf.float32, [None, 10],name='y-input')

with tf.name_scope('layer'):
# 创建神经网络
    with tf.name_scope('weights'):
        W1 = tf.Variable(tf.truncated_normal([784,10],stddev=0.1))
        variable_summaries(W1)
    with tf.name_scope('biases'):
        b1 = tf.Variable(tf.zeros([1, 10]))
        variable_summaries(b1)
    with tf.name_scope('sigmoid'):
        prediction = tf.nn.sigmoid(tf.matmul(x,W1) + b1)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss',loss)
    # 梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()
with tf.name_scope('accuracy'):
# 计算准确率
    prediction_2 = tf.nn.softmax(prediction)
# 得到一个布尔型列表，存放结果是否正确
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction_2,1)) #argmax 返回一维张量中最大值索引

# 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 把布尔值转换为浮点型求平均数
    tf.summary.scalar('accuracy',accuracy)

#合并所有summary
merged = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(31):
        for batch in range(n_batch):
            # 获得批次数据
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary,_ =sess.run([merged,train_step], feed_dict={x:batch_xs, y:batch_ys,})

        writer.add_summary(summary,epoch)
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels} )
        print("Iter " + str(epoch) + " Testing Accuracy: " + str(acc))