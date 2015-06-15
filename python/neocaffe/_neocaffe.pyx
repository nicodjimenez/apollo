#from caffe_pb2 import NetParameter, LayerParameter, DataParameter, SolverParameter, ParamSpec
from caffe_pb2 import LayerParameter, DataParameter
from libcpp.string cimport string
cdef extern from "caffe/neocaffe.hpp" namespace "caffe":
    int foo(string)
cdef extern from "caffe/neonet.hpp" namespace "caffe":
    cdef cppclass NeoNet[T]:
        void Init()
        void ForwardLayer(string)
def bar():
    print 3

cdef NeoNet[float] net

class Layer(object):
    pass
class DataLayer(Layer):
    def __init__(self):
        self.param = LayerParameter()
        self.param.type = "Data"
        self.param.name = "data"
        self.param.top.append("image")
        self.param.data_param.source = 'examples/language_model/lm_train_db'
        self.param.data_param.backend = DataParameter.LMDB
        self.param.data_param.batch_size = 1

class ReluLayer(Layer):
    def __init__(self):
        self.param = LayerParameter()
        self.param.type = "ReLU"
        self.param.name = "relu"
        self.param.bottom.append("image")
        self.param.top.append("relu")

#cdef string cpp_string = data_layer.SerializeToString()

net.ForwardLayer(DataLayer().param.SerializeToString())
net.ForwardLayer(ReluLayer().param.SerializeToString())
#print 'hello world'
