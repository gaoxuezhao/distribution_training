import tensorflow as tf
import numpy as np

class MLP():
    def __init__(self,name, input_shape, neural_nums_of_hidden_layers):
        self.W=[]
        self.biases=[]
        with tf.variable_scope('MLP'):
            for i in range(len(neural_nums_of_hidden_layers)):
                name_w=name+'w'+str(i)
                if i == 0:
                    w=tf.get_variable(name_w, shape=[input_shape[1], neural_nums_of_hidden_layers[i]],initializer=tf.constant_initializer(0))
                else :
                    w=tf.get_variable(name_w, shape=[neural_nums_of_hidden_layers[i-1], neural_nums_of_hidden_layers[i]],initializer=tf.constant_initializer(0))
                self.W.append(w)

                name_b=name+'b'+str(i)
                biase=tf.get_variable(name_b, shape=[neural_nums_of_hidden_layers[i]], initializer=tf.constant_initializer(0))
                self.biases.append(biase)

    def apply(self, input_tensor):
        result=input_tensor
        for i in range(len(self.W)):
            result=tf.sigmoid(tf.matmul(result, self.W[i]) + self.biases[i])
        return result
        

class Factory():
    
    @staticmethod
    def get_MLP(name, input_shape, neural_nums_of_hidden_layers):
        return MLP(name, input_shape, neural_nums_of_hidden_layers)
        

if __name__ == '__main__':
    input_np=np.random.randint(1, 10, size=(2, 3))
    print(input_np)
    input_placehold = tf.placeholder("float", [None, 3])
    
    layers=[2,3,2]

    test_mlp=Factory.get_MLP('test', input_placehold.get_shape(), layers)
    result=test_mlp.apply(input_placehold)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        print(sess.run(result, feed_dict={input_placehold:input_np})) 
