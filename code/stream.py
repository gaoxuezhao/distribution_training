# -*- coding: UTF-8 -*-
import tensorflow as tf

class Stream:
    def __init__(self,config):
        self.data_path=config.data_path.split(',')
        self.num_epochs=config.num_epochs
        self.batch_size=config.batch_size

    def read_data(self):
        filename_queue = tf.train.string_input_producer(self.data_path,num_epochs=self.num_epochs, shuffle=True)
        print(self.data_path[0])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
	
        features = tf.parse_single_example(serialized_example,
		features={
                        'feature':tf.FixedLenFeature([],tf.string),
			'label': tf.FixedLenFeature([], tf.float32),
			'height': tf.FixedLenFeature([], tf.int64),
			'width': tf.FixedLenFeature([], tf.int64)})
        feature=tf.decode_raw(features['feature'], tf.float32)
        feature=tf.reshape(feature,[784])
        label = tf.cast(features['label'], tf.float32)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        
        return  feature, label, height, width
    
    def get_data_batch(self):
        feature, label, height, width=self.read_data()
        feature, label, height, width = tf.train.shuffle_batch([feature, label, height, width], batch_size=self.batch_size, num_threads=2, capacity=1000, min_after_dequeue=300,allow_smaller_final_batch=True)
        feature = tf.cast(feature, tf.float32)
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, 10)
        
        return self.rescaler(feature), label

    def rescaler(self, feature):
       return (feature - 127.5)/255.0
