import os

def cnt(n):
    for i in range(n):
        yield i

class A(object):
    def __init__(self):
        super(A).__init__()

    def cnt(self):
        for i in range(10):
            yield i

    def pr(self):
        for i in self.cnt():
            print(i)

a = A()
a.pr()

