
from contextlib import contextmanager

import numpy as np

import onnx
from onnx import AttributeProto, TensorProto, GraphProto

type_map = {
    np.array([1.]).dtype: TensorProto.FLOAT, 
    np.array([1]).dtype: TensorProto.INT64
}

default_namespace_stack = []
default_extern_stack = []
default_env = None

def node(op_type:str, inputs:list, outputs:list, **attributes):
    env = get_env()
    
    _inputs = []
    for inp in inputs:
        if isinstance(inp, str):
            # _inputs.append(inp)
            name = get_namespace_name(inp)
        else:
            name = env.get_constant_name()
            name = get_namespace_name(name)
            
            typ = type_map[np.array(inp).dtype]
            shape = np.array(inp).shape
            if len(shape) == 0:
                shape = [1]
                inp = [inp]
            tensor = onnx.helper.make_tensor(name, typ, shape, inp)
            env.initializer.append(tensor)
        _inputs.append(name)
    
    _outputs = [get_namespace_name(o) for o in outputs]
    
    _node = onnx.helper.make_node(
        op_type,
        _inputs,
        _outputs,
        **attributes
    )
    
    env.node.append(_node)
    
    return _node

def input(name, elem_type, shape):
    env = get_env()
    
    _name = get_namespace_name(name)
    _inp = onnx.helper.make_tensor_value_info(_name, elem_type, shape)
    env.input.append(_inp)
    return _inp
    

def output(name, elem_type, shape):
    env = get_env()
    
    _name = get_namespace_name(name)
    _inp = onnx.helper.make_tensor_value_info(_name, elem_type, shape)
    env.output.append(_inp)
    return _inp


class Env:
    def __init__(self, model=None, append=True):
        self.model = model
        self.append = append
        
        self.node = []
        self.initializer = []
        self.input = []
        self.output = []
        
        self.constant_count = 0
        
    def get_constant_name(self):
        name = "__constant_" + str(self.constant_count)
        self.constant_count += 1
        return name
        
    def show_debug(self):
        print("node:")
        print(self.node)
        print("initializer:")
        print(self.initializer)
        print("input:")
        print(self.input)
        print("output:")
        print(self.output)
        
    def do(self):
        if self.append:
            for node in self.node:
                self.model.graph.node.append(node)
        else:
            for node in self.node[::-1]:
                self.model.graph.node.insert(0, node)
        for initializer in self.initializer:
            self.model.graph.initializer.append(initializer)
        for input in self.input:
            self.model.graph.input.append(input)
        for output in self.output:
            self.model.graph.output.append(output)


@contextmanager
def env(model=None, append=True, *, debug=False):
    global default_env
    assert default_env is None
    default_env = Env(model, append=append)
    try:
        yield
    finally:
        _default_env = default_env
        default_env = None
        if debug:
            _default_env.show_debug()
        else:
            _default_env.do()

def get_env():
    global default_env
    assert default_env is not None
    return default_env



@contextmanager
def namespace(s, *, extern=[]):
    assert not isinstance(extern, str)
    
    default_namespace_stack.append(s)
    default_extern_stack.append(set(extern))
    
    try:
        yield
    finally:
        default_namespace_stack.pop()
        default_extern_stack.pop()


def get_namespace_name(s):
    sl = [s]
    is_extern = True
    for ns, extern_set in zip(default_namespace_stack[::-1], default_extern_stack[::-1]):
        if not is_extern:
            sl.append(ns)
        else:
            is_extern = s in extern_set
            if not is_extern:
                sl.append(ns)
        
    return "/".join(sl[::-1])


