from collections import deque
import numpy as np
import threading
import time


def print_cube(num):
    # function to print cube of given num
    for i in range(999999999):
        continue
    print("Cube: {}" .format(num * num * num))
 
def print_square(num):
    # function to print square of given num
    for i in range(999999999):
        continue
    print("Square: {}" .format(num * num))

def run_with_thread():
    # creating thread
    t1 = threading.Thread(target=print_square, args=(10,))
    t2 = threading.Thread(target=print_cube, args=(10,))

    # starting thread 1
    t1.start()
    # starting thread 2
    t2.start()

    # wait until thread 1 is completely executed
    t1.join()
    # wait until thread 2 is completely executed
    t2.join()

def run_without_thread():
    print_cube(10)
    print_square

start_time = time.time()

run_with_thread()
# run_without_thread()

end_time = time.time()
execution_time = end_time - start_time
print('time spent: ', round(execution_time, 4))