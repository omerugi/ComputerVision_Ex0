# -*- coding: utf-8 -*-

"""
High-energy physics objects.
"""


__all__ = ["LorentVector"]


from numphy.core import Wrapper


class LorentVector(Wrapper):

    # def __init__(self, data=None, trace=None, attrs=None):
    def __init__(self, *args, **kwargs):
        super(LorentVector, self).__init__(*args, **kwargs)

        # 
