import tensorflow as tf
from model import *
from stream import *
from config import *

class Predicter():
    def __init__(self, stream, model):
        self.stream=stream
        self.model=model
    
    def predict(self):
        feature, label = self.stream.get_data_batch()
        
        Yp = self.model.build_graph(feature)
        
        correct_prediction = tf.equal(tf.argmax(Yp,1), tf.argmax(label,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        saver=tf.train.Saver(max_to_keep=5)
        max_acc=0
        
        np.set_printoptions(threshold='nan') 
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            saver.restore(sess,'ckpt/mnist.ckpt-13544')
            coord=tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            i=0
            try:
                while not coord.should_stop():
                    Yp_r, accuracy_r = sess.run([Yp, accuracy])
                    print(i, accuracy_r)
                    i+=1
            except tf.errors.OutOfRangeError:
                print('done! now lets kill all the threads')
            finally:
                coord.request_stop()
                print('all threads are asked to stop!')
            coord.join(threads)
            print('all threads are stopped!')
            print(i)

if __name__ == '__main__':
    data_path='../data/validation.tfrecord'
    config = Config(data_path)
    config.num_epochs=1
    config.batch_size=1
    stream = Stream(config)
    model = Model()
    predicter = Predicter(stream, model)
    predicter.predict()
