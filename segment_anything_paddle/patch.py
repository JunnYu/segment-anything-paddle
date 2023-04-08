# nn.Parameter
import numpy as np

# load_torch
import io
import math
import os
import paddle
import pickle
from _io import BufferedReader
from functools import lru_cache
from typing import Union
from zipfile import ZipFile


def Parameter(data: paddle.Tensor, requires_grad=True):
    tensor = paddle.create_parameter(
        data.shape, dtype=data.dtype, default_initializer=paddle.nn.initializer.Assign(data)
    )
    if not requires_grad:
        tensor.stop_gradient = True
    return tensor


paddle.nn.Parameter = Parameter

# F.scaled_dot_product_attention_

try:
    from paddle.incubate.nn.memory_efficient_attention import memory_efficient_attention
    from paddle.nn.functional.flash_attention import flash_attention

    def scaled_dot_product_attention_(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        training=False,
        attention_op="cutlass",
    ):
        if attention_op is None or attention_op == "cutlass" or training:
            if scale is None:
                scale = 1 / math.sqrt(query.shape[-1])
            # support fp32, fp16, bfp16
            output = memory_efficient_attention(
                query,
                key,
                value,
                attn_mask,
                p=dropout_p,
                scale=scale,
                training=training,
            )
        elif attention_op == "flash":
            raw_dtype = query.dtype
            if raw_dtype == paddle.float32:
                query, key, value = (
                    query.cast(paddle.float16),
                    key.cast(paddle.float16),
                    value.cast(paddle.float16),
                )
            output = flash_attention(
                query, key, value, dropout=dropout_p, causal=is_causal, return_softmax=False
            )[0]
            if raw_dtype == paddle.float32:
                output = output.cast(raw_dtype)
        else:
            raise ValueError("ppxformers's attention_op shoulde be in ['cutlass', 'flash']")
        return output

    paddle.nn.functional.scaled_dot_product_attention_ = scaled_dot_product_attention_
except ImportError:
    pass


MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30


class SerializationError(Exception):
    """Exception for serialization"""

    pass


def seek_by_string(file_handler: BufferedReader, string: str, file_size: int) -> int:
    """seek the index of file-handler with target words
    Args:
        file_handler (BufferedReader): file handler
        string (str): the specific string in the file
        file_size (int): size of file
    Returns:
        int: end index of target string
    """
    word_index = 0
    word_bytes = string.encode("latin")
    empty_byte = "".encode("latin")

    while word_index < len(string) and file_handler.tell() < file_size:
        content = file_handler.read(1)
        if content == empty_byte:
            break

        if word_bytes[word_index] == content[0]:
            word_index += 1
        else:
            word_index = 0

    if file_handler.tell() >= file_size - 1:
        raise SerializationError(f"can't find the find the target string<{string}> in the file")
    return file_handler.tell()


def read_prefix_key(path):
    file_size = os.stat(path).st_size
    with open(path, "rb") as file_handler:
        end_index = seek_by_string(file_handler, "data.pkl", file_size)
        file_handler.seek(MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
        prefix_key = file_handler.read(end_index - MZ_ZIP_LOCAL_DIR_HEADER_SIZE - len("/data.pkl"))
    return prefix_key.decode("latin")


def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
    if isinstance(bytes_str, bytes):
        return bytes_str.decode("ascii")
    return bytes_str


@lru_cache(maxsize=None)
def _storage_type_to_dtype_to_map():
    """convert storage type to numpy dtype"""
    return {
        "DoubleStorage": np.double,
        "FloatStorage": np.float32,
        "HalfStorage": np.half,
        "LongStorage": np.int64,
        "IntStorage": np.int32,
        "ShortStorage": np.int16,
        "CharStorage": np.int8,
        "ByteStorage": np.uint8,
        "BoolStorage": np.bool_,
        "ComplexDoubleStorage": np.cdouble,
        "ComplexFloatStorage": np.cfloat,
        "BFloat16Storage": np.uint16,  # support bf16
    }


class StorageType:
    """Temp Class for Storage Type"""

    def __init__(self, name):
        self.dtype = _storage_type_to_dtype_to_map()[name]

    def __str__(self):
        return f"StorageType(dtype={self.dtype})"


def _element_size(dtype: str) -> int:
    """
    Returns the element size for a dtype, in bytes
    """
    if dtype in [np.float16, np.float32, np.float64]:
        return np.finfo(dtype).bits >> 3
    elif dtype == np.bool_:
        return 1
    else:
        return np.iinfo(dtype).bits >> 3


class UnpicklerWrapperStage(pickle.Unpickler):
    def find_class(self, mod_name, name):
        if type(name) is str and "Storage" in name:
            try:
                return StorageType(name)
            except KeyError:
                pass

        if mod_name == "torch._utils":
            # rebuild torch.nn.Papameter
            if name == "_rebuild_parameter":
                return _rebuild_parameter
            # rebuild torch.nn.Papameter with state
            if name == "_rebuild_parameter_with_state":
                return _rebuild_parameter_with_state
            # rebuild torch.Tensor
            return _rebuild_tensor_stage

        # pytorch_lightning tensor builder
        if "pytorch_lightning" in mod_name:
            return dumpy
        return super().find_class(mod_name, name)


def _rebuild_tensor_stage(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    # if a tensor has shape [M, N] and stride is [1, N], it's column-wise / fortran-style
    # if a tensor has shape [M, N] and stride is [M, 1], it's row-wise / C-style
    # defautls to C-style
    if stride is not None and len(stride) > 1 and stride[0] == 1 and stride[1] > 1:
        order = "F"
    else:
        order = "C"
    # fix this storage_offset
    return storage[storage_offset : storage_offset + np.prod(size)].reshape(size, order=order)


def _rebuild_parameter(data, requires_grad, backward_hooks):
    return data


def _rebuild_parameter_with_state(data, requires_grad, backward_hooks, state):
    return data


def dumpy(*args, **kwarsg):
    return None


def load_torch(path: str, **pickle_load_args):
    """
    load torch weight file with the following steps:
    1. load the structure of pytorch weight file
    2. read the tensor data and re-construct the state-dict
    Args:
        path: the path of pytorch weight file
        **pickle_load_args: args of pickle module
    Returns:
    """
    pickle_load_args.update({"encoding": "utf-8"})

    prefix_key = read_prefix_key(path)

    torch_zip = ZipFile(path, "r")
    loaded_storages = {}

    def load_tensor(dtype, numel, key, location):
        name = f"{prefix_key}/data/{key}"
        typed_storage = np.frombuffer(torch_zip.open(name).read()[:numel], dtype=dtype)
        return typed_storage

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        assert (
            typename == "storage"
        ), f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        storage_type, key, location, numel = data
        dtype = storage_type.dtype

        if key in loaded_storages:
            typed_storage = loaded_storages[key]
        else:
            nbytes = numel * _element_size(dtype)
            typed_storage = load_tensor(dtype, nbytes, key, _maybe_decode_ascii(location))
            loaded_storages[key] = typed_storage

        return typed_storage

    data_iostream = torch_zip.open(f"{prefix_key}/data.pkl").read()
    unpickler_stage = UnpicklerWrapperStage(io.BytesIO(data_iostream), **pickle_load_args)
    unpickler_stage.persistent_load = persistent_load
    state_dict = unpickler_stage.load()
    torch_zip.close()
    return state_dict
