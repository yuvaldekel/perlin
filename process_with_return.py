from multiprocessing import Process


class ProcessWithReturn(Process):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Process.__init__(self, group, target, name, args, kwargs)
        self._return = None


    def run(self):
        if self._target is not None:
            self._return = self._target(self._args[0])


    def join(self, *args):
        Process.join(self, *args)
        return self._return