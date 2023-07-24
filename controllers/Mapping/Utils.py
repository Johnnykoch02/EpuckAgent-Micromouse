import numpy as np

def get_dTheta(ti, tf):
    positiveDTheta = ((tf+360)-ti)%360
    negativeDTheta = -360 + positiveDTheta
    return positiveDTheta if abs(positiveDTheta) <= abs(negativeDTheta) else negativeDTheta
# print(get_dTheta(90, -93)
def get_closestCardinalDirection(theta):
    thetas = np.array([0, 90, 180, 270, 360])
    theta_idx = np.argmin(np.abs(thetas - theta))
    return theta_idx % 4

class HeapQueue:
    def __init__(self, min_heap=True):
        self.heap = []
        self.min_heap = min_heap

    def size(self):
        return len(self.heap)

    def push(self, item):
        self.heap.append(item)

        position = len(self.heap) - 1
        parent = (position - 1) // 2

        if self.min_heap:
            while parent >= 0 and self.heap[position] < self.heap[parent]:
                self.heap[parent], self.heap[position] = self.heap[position], self.heap[parent]
                position, parent = parent, (parent - 1) // 2
        else:
            while parent >= 0 and self.heap[position] > self.heap[parent]:
                self.heap[parent], self.heap[position] = self.heap[position], self.heap[parent]
                position, parent = parent, (parent - 1) // 2

    def pop(self):
        if not self.heap:
            return None
        result = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        position = 0
        left = 1
        right = 2

        if self.min_heap:
            while left < len(self.heap):
                min_child = left
                if right < len(self.heap) and self.heap[right] < self.heap[left]:
                    min_child = right

                if self.heap[position] > self.heap[min_child]:
                    self.heap[position], self.heap[min_child] = self.heap[min_child], self.heap[position]
                    position = min_child
                    left = position * 2 + 1
                    right = position * 2 + 2
                else:
                    break
        else:
            while left < len(self.heap):
                max_child = left
                if right < len(self.heap) and self.heap[right] > self.heap[left]:
                    max_child = right

                if self.heap[position] < self.heap[max_child]:
                    self.heap[position], self.heap[max_child] = self.heap[max_child], self.heap[position]
                    position = max_child
                    left = position * 2 + 1
                    right = position * 2 + 2
                else:
                    break
        return result
    
    def remove_worst(self):
        self.heap.pop(0)