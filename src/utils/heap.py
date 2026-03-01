class MinHeap:
    def __init__(self, k, cmp):
        self.k = k
        self.cmp = cmp
        self.heap = []

    def _push_heap(self, val):
        self.heap.append(val)
        idx = len(self.heap) - 1
        parent = (idx - 1) // 2
        while idx > 0 and self.cmp(self.heap[parent], self.heap[idx]): 
            # if parent is greater -> swap
            self.heap[parent], self.heap[idx] = self.heap[idx], self.heap[parent]
            idx = parent
            parent = (idx - 1) // 2

    def _pop_heap(self):
        if not self.heap:
            return None
        top = self.heap[0] # the minimum value in the current minheap
        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            n = len(self.heap)
            idx = 0
            while True: # adjust the heap from top to the bottom
                left = 2 * idx + 1
                right = 2 * idx + 2
                candidate = idx

                # if parent is greater -> swap
                if left < n and self.cmp(self.heap[candidate], self.heap[left]):
                    candidate = left
                if right < n and self.cmp(self.heap[candidate], self.heap[right]):
                    candidate = right
                if candidate == idx:
                    break
                self.heap[idx], self.heap[candidate] = self.heap[candidate], self.heap[idx]
                idx = candidate
        return top

    def add(self, val):
        if len(self.heap) < self.k:
            self._push_heap(val)
        else:
            # the new value is greater than the current top-k -> replace
            if self.cmp(val, self.heap[0]):
                self._pop_heap()
                self._push_heap(val)

    def get_topk(self):
        def quick_sort(arr, cmp):
            if len(arr) <= 1:
                return arr
            
            pivot = arr[len(arr) // 2]  
            # cmp(pivot, x) returns 1 if pivot > x
            left = []
            right = []
            for x in arr:
                if x == pivot:
                    continue
                elif cmp(x, pivot):
                    left.append(x)
                else:
                    right.append(x) 
            return quick_sort(left, cmp) + [pivot] + quick_sort(right, cmp)

        return quick_sort(self.heap, self.cmp)


class MaxHeap:
    def __init__(self, k, cmp):
        self.k = k
        self.cmp = cmp  
        self.heap = []

    def _push_heap(self, val):
        self.heap.append(val)
        idx = len(self.heap) - 1
        parent = (idx - 1) // 2
        while idx > 0 and self.cmp(self.heap[parent], self.heap[idx]): 
            self.heap[parent], self.heap[idx] = self.heap[idx], self.heap[parent]
            idx = parent
            parent = (idx - 1) // 2

    def _pop_heap(self):
        if not self.heap:
            return None
        top = self.heap[0] 
        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            n = len(self.heap)
            idx = 0
            while True:
                left = 2 * idx + 1
                right = 2 * idx + 2
                candidate = idx

                if left < n and self.cmp(self.heap[candidate], self.heap[left]):
                    candidate = left
                if right < n and self.cmp(self.heap[candidate], self.heap[right]):
                    candidate = right
                if candidate == idx:
                    break
                self.heap[idx], self.heap[candidate] = self.heap[candidate], self.heap[idx]
                idx = candidate
        return top

    def add(self, val):
        if len(self.heap) < self.k:
            self._push_heap(val)
        else:
            if self.cmp(val, self.heap[0]):
                self._pop_heap()
                self._push_heap(val)

    def get_topk(self):
        def quick_sort(arr, cmp):
            if len(arr) <= 1:
                return arr
            
            pivot = arr[len(arr) // 2]
            # cmp(x, pivot) returns 1 if pivot > x
            left = []
            right = []
            for x in arr:
                if x == pivot:
                    continue
                elif cmp(x, pivot):
                    left.append(x)
                else:
                    right.append(x)
            return quick_sort(left, cmp) + [pivot] + quick_sort(right, cmp)

        return quick_sort(self.heap, self.cmp)
