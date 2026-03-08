# -*- coding: utf-8 -*-
# 简化的 model_full_param_pb2.py
# 由于原始 proto 文件太复杂，这里创建一个简化版本

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2

_sym_db = _symbol_database.Default()

# 这里只是一个简化的版本，实际使用时应该用编译的版本
# 定义所有消息类型的占位符
class ModelFullParam(_message.Message):
    __metaclass__ = _reflection.GeneratedProtocolMessageType
    
class ModelInputDTO(_message.Message):
    __metaclass__ = _reflection.GeneratedProtocolMessageType
    
class StringList(_message.Message):
    __metaclass__ = _reflection.GeneratedProtocolMessageType
    
class FloatList(_message.Message):
    __metaclass__ = _reflection.GeneratedProtocolMessageType
    
class StationPredictDTO(_message.Message):
    __metaclass__ = _reflection.GeneratedProtocolMessageType
    
class ModelOptimizeParamDTO(_message.Message):
    __metaclass__ = _reflection.GeneratedProtocolMessageType
    
class ModelUpdateParamList(_message.Message):
    __metaclass__ = _reflection.GeneratedProtocolMessageType
    
class StringStringMap(_message.Message):
    __metaclass__ = _reflection.GeneratedProtocolMessageType

print("⚠️  注意: 这是简化的 pb2 文件，仅用于测试")
