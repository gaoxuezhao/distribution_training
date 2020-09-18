import tensorflow as tf
from module import *

class Model():
    def __init__(self):
        print('init')
        
    def build_graph(self,input_tensor):
        
        mlp_layers=[10]
        mlp=Factory.get_MLP('mlp', input_tensor.get_shape(), mlp_layers)
        Yp=tf.nn.softmax(mlp.apply(input_tensor))
        return Yp
