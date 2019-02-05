from abc import ABCMeta, abstractmethod


class Detector():

    __metaclass__ = ABCMeta

    def __init__(self, profiler):
        self.profiler = profiler
        self.dataEngine = self.profiler.dataEngine
        self.field = self.dataEngine.field
        try:
            self.key = self.profiler.key
        except:
            self.key = 'id'




