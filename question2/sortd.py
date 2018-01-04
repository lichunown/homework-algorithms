import random
import numpy as np
from time import time
# import unittest
from memory_profiler import profile

defaultfunc = lambda x,y:x<y

def exch(data,i,j):
    a = data[i]
    data[i] = data[j]
    data[j] = a

def isSorted(data,func =defaultfunc):
    N = len(data)
    for i in range(1,N):
        if not func(data[i-1],data[i]):
            return False
    return True

# Insertion Sort
def ISsort(data,func=defaultfunc):
    N = len(data)
    for i in range(1,N):
        for j in range(i,0,-1):
            if func(data[j-1],data[j]):
                break
            exch(data,j,j-1) 

# TDM
def TDMsort(data,func=defaultfunc):
    N = len(data)
    aux = data.copy()
    _TDMsort(data,aux,0,N-1,func)

def _Merge(data, aux, lo:int, mid:int, hi:int, func=defaultfunc):
    for k in range(lo, hi + 1):
        aux[k] = data[k]
    i, j = lo, mid + 1
    for k in range(lo, hi + 1):
        if i > mid:
            data[k] = aux[j]
            j += 1
        elif j > hi:
            data[k] = aux[i]
            i += 1
        elif func(aux[j],aux[i]):
            data[k] = aux[j]
            j += 1
        else:
            data[k] = aux[i]
            i += 1

def _TDMsort(data, aux, lo:int, hi:int, func=defaultfunc):
    if lo >= hi:
        return
    mid = int(lo + (hi-lo)/2)
    _TDMsort(data,aux,lo,mid,func)
    _TDMsort(data,aux,mid+1,hi,func)
    _Merge(data,aux,lo,mid,hi,func)

# BUM
def BUMsort(data,func = defaultfunc):
    aux = data.copy()
    N = len(data)
    sz = 1
    while sz < N:
        lo = 0
        while lo < N - sz:
            _Merge(data,aux,lo,lo+sz-1,min(lo+sz+sz-1,N-1),func)
            lo += sz + sz
        sz += sz



# RQ
def _partition(data,lo:int,hi:int,func = defaultfunc):
    i,j,v = lo,hi+1,data[lo]
    while True:
        i += 1
        while func(data[i],v):
            if i==hi:
                break
            i += 1
        j -= 1
        while func(v,data[j]):
            if j==lo:
                break
            j -= 1
        if(i>=j):
            break
        exch(data,i,j)
    exch(data,lo,j)
    return j


def _RQsort(data,lo:int,hi:int,func = defaultfunc):
    if(hi<=lo):
        return 
    cut = _partition(data,lo,hi,func)
    _RQsort(data,lo,cut-1)   
    _RQsort(data,cut+1,hi) 

def RQsort(data,func = defaultfunc):
    random.shuffle(data)
    _RQsort(data,0,len(data)-1,func)


# QD3P
def QD3Psort(data,func = defaultfunc):
    random.shuffle(data)
    _QD3Psort(data,0,len(data)-1,func)

def _QD3Psort(data,lo:int,hi:int,func = defaultfunc):
    if(hi<=lo):
        return 
    lt,i,gt,v = lo,lo + 1,hi,data[lo]
    while i<=gt:
        if func(data[i],v):
            exch(data,lt,i)
            lt += 1
            i += 1
        elif func(v,data[i]):
            exch(data,i,gt)
            gt -= 1
        else:
            i += 1
    _QD3Psort(data,lo,lt-1,func)
    _QD3Psort(data,gt+1,hi,func)     


# test
class TestSort():
    def __init__(self):       
        self.sortfunc = {
            'ISsort':ISsort,
            'TDMsort':TDMsort,
            'BUMsort':BUMsort,
            'RQsort':RQsort,
            'QD3Psort':QD3Psort,
            # 'sorted':sorted,
        } 
        self.isSorted = isSorted     

    def testAll(self, num = 1500,randomfunc = np.random.random):
        data = randomfunc(num)
        print("Test Sort   create:{}    num:{} ".format(randomfunc.__name__,num))
        r = []
        for funcname in self.sortfunc:
            r.append(self.test(data,self.sortfunc[funcname]))
        return r
           
    def test(self,data,func):
        data = data.copy()
        time1 = time()
        func(data)
        time2 = time()
        print("{}: Run time: {} \t {}".format(func.__name__, time2-time1,self.isSorted(data)))
        return time2-time1

    def test_for10(self):
        r = []
        for i in range(10):
            r.append(self.testAll())
        r = np.array(r)
        print(r)
        print(np.mean(r,0))
        return r
    
test = TestSort()
test.testAll(num = 2000,randomfunc=np.random.random)

