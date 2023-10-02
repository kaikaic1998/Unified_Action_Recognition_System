from collections import deque
import numpy as np
import threading
import multiprocessing
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

def run_with_multiprocess():
    t1 = multiprocessing.Process(target=print_square, args=(10,))
    t2 = multiprocessing.Process(target=print_cube, args=(10,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

def run_without_thread():
    print_cube(10)
    print_square(10)

if __name__ == "__main__":
    start_time = time.time()

    # run_with_thread() # 20.9 s
    # run_without_thread() # 21.55 s
    run_with_multiprocess() # 11.0284

    end_time = time.time()
    execution_time = end_time - start_time
    print('time spent: ', round(execution_time, 4))