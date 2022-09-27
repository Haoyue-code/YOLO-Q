from collections import OrderedDict, namedtuple
from ..utils.torch_utils import select_device
import torch
import numpy as np
import tensorrt as trt


class TRT_Model:
    def __init__(self, engine_file_path, device=0) -> None:
        """Tensorrt engine inference. 
        Input ndoe:`images`.
        Output ndoe:`output`.
        """
        self.device = select_device(device)
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_file_path, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            # 不需要额外给`images`创建data
            if name in ['images']:
                continue
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = model.create_execution_context()
        # self.batch_size = self.bindings['images'].shape[0]

    def __call__(self, img):
        # assert img.shape == self.bindings['images'].shape, (img.shape, self.bindings['images'].shape)
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2([self.binding_addrs['images'], self.binding_addrs['output0']])
        # self.context.execute_v2(list(self.binding_addrs.values()))
        # 没有拷贝到cpu，数据依旧在cuda, 可供torch操作
        y = self.bindings['output0'].data
        return y
