import numpy as np
import tensorflow as tf
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

class serveLSTM:
    def __init__(self):
        graph = tf.Graph()
        self.sess = tf.Session(graph=graph)
        tf.saved_model.loader.load(self.sess,{'train'},'./SavedModel/')
        self.pred = graph.get_tensor_by_name('Variable_3:0')
        self.x = graph.get_tensor_by_name('Variable_2:0')

    def runInference(self,input):
        y =self.sess.run([self.pred],feed_dict={'x:0':input.reshape(1,128,9)})
        result = list(y[0])
        return LABELS[result.index(max(result))]


    def close(self):
        self.sess.close()


import pickle
sample_input = pickle.load(open('input_sample','rb'))
lstm = serveLSTM()
result = lstm.runInference(sample_input)
print(result)