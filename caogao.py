from collections import deque
import numpy as np
import threading
import multiprocessing
import time
from queue import Queue # generally slower than list for iteration
# because it includes additional overhead for managing thread safety and synchronization, which is not present in a regular list
from collections import deque
import pathlib
import random
from matplotlib import pyplot as plt
import matplotlib
from torchvision import datasets

def try_deque():
    deq = deque([1, 2])

    for i in range(3):
        deq.append(i)

    print(len(deq))

    for element in deq:
        print(element)
    deq = np.array(deq)
    print(deq.shape)
# try_deque()

def queue_vs_list():
    start_time = time.time()
    q = Queue(maxsize = 99999)
    for i in range(99999):
        q.put(i)

    for i in q.queue:
        continue
    end_time = time.time()
    execution_time = end_time - start_time
    print('time spent: ', round(execution_time, 4))

    start_time = time.time()
    lst = []
    for i in range(99999):
        lst.append(99999)

    for i in lst:
        continue
    end_time = time.time()
    execution_time = end_time - start_time
    print('time spent: ', round(execution_time, 4))
# queue_vs_list()

# def print_cube(num):
#     # function to print cube of given num
#     for i in range(999999999):
#         continue
#     print("Cube: {}" .format(num * num * num))
 
# def print_square(num):
#     # function to print square of given num
#     for i in range(999999999):
#         continue
#     print("Square: {}" .format(num * num))

# def run_with_thread():
#     # creating thread
#     t1 = threading.Thread(target=print_square, args=(10,))
#     t2 = threading.Thread(target=print_cube, args=(10,))

#     # starting thread 1
#     t1.start()
#     # starting thread 2
#     t2.start()

#     # wait until thread 1 is completely executed
#     t1.join()
#     # wait until thread 2 is completely executed
#     t2.join()

# def run_with_multiprocess():
#     t1 = multiprocessing.Process(target=print_square, args=(10,))
#     t2 = multiprocessing.Process(target=print_cube, args=(10,))

#     t1.start()
#     t2.start()

#     t1.join()
#     t2.join()

# def run_without_thread():
#     print_cube(10)
#     print_square(10)

# if __name__ == "__main__":
#     start_time = time.time()

#     # run_with_thread() # 20.9 s
#     # run_without_thread() # 21.55 s
#     run_with_multiprocess() # 11.0284

#     end_time = time.time()
#     execution_time = end_time - start_time
#     print('time spent: ', round(execution_time, 4))

import torch
import torch.nn as nn

# # Correct way to use CrossEntropyLoss
# loss = nn.CrossEntropyLoss()
# input = torch.randn(1, 5)
# target = torch.empty(1, dtype=torch.long).random_(5)
# output = loss(input, target)

# print(input)
# print(target)
# print(output)
# print('\n')

# input = torch.randn(3, 5, requires_grad=True)
# target = torch.tensor([1, 0, 4])
# output = nn.functional.nll_loss(nn.functional.log_softmax(input, dim=1), target)
# print(input)
# print(target)
# print(output)

# m = nn.LogSoftmax(dim=1)
# loss = nn.NLLLoss()
# # input is of size N x C = 3 x 5
# input = torch.randn(1, 5, requires_grad=True)
# # each element in target has to have 0 <= value < C
# # target = torch.tensor([1, 0, 4])
# target = torch.empty(1, dtype=torch.long).random_(5)
# softmax = m(input)
# output = loss(softmax, target)

# print(input)
# print(softmax)
# print(target)
# print(output, '\n')

# softmax = torch.tensor([0.006354291, 0.028920725, 1.0, 0.00805, 0.00046146192], requires_grad=True)
# input = torch.log(softmax)
# target = torch.tensor(2, dtype=torch.long)
# output = loss(input, target)

# print(softmax)
# print(input)
# print(target)
# print(output)

# random_index = list(range(5))
# for i in range(5):
#     random.shuffle(random_index)
#     print('random numbers', random_index)

if __name__ == '__main__':
    lst = [1,2,3,4,5,6,7,8]
    matplotlib.use('Agg')

    # Plot the list data
    fig = plt.plot(lst)
    print(fig)

    # Show the plot
    fig = plt.show()
