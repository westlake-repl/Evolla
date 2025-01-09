import os

def register_llm(cls):
    global now_cls
    now_cls = cls
    return cls

class LLMInterface:
    @classmethod
    def init_llm(cls, model_py_path, **kwargs):
        """
        Initialize model from python file.
        Args:
            model_py_path: Path to model python file. e.g. model/transformer.py
            **kwargs: Kwargs for model initialization

        Returns:
            Initialized model
        """
        sub_dirs = model_py_path.split(os.sep)
        cmd = f"from {'.'.join(sub_dirs[:-1])} import {sub_dirs[-1].split('.')[0]}"
        exec(cmd)
        return now_cls(**kwargs)
