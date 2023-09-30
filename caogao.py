from collections import deque
import numpy as np

# q = deque

dic = dict()
lst1 = [1,2,3]
lst2 = [4,5,6]

dic['key1'] = np.array([lst1])

dic['key1'].append(np.array(lst2))

print(dic)
