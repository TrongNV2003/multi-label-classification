from enum import Enum


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
