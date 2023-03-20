from typing import Type
from ..tokenizer import Tokenizer
from .svd_tokenizer import SVDTokenizer

def get_tokenizer_class(classname) -> Type[Tokenizer]:
    if classname == "svd":
        return SVDTokenizer