from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

class SimpleClass(object):
    def __init__(self, num):
        self.var = num

    def set(self, value):
        print(self.var)
        self.var = value

    def get(self):
        return self.var
        

def change_obj_value(obj):
    obj.set(100)


if __name__ == '__main__':
    BaseManager.register('SimpleClass', SimpleClass)
    manager = BaseManager()
    manager.start()
    inst = manager.SimpleClass(10)

    p = Process(target=change_obj_value, args=[inst])
    p.start()
    p.join()

    print(inst)                    # <__main__.SimpleClass object at 0x10cf82350>
    print(inst.get())              # 100