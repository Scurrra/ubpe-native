import heapq


class TopElements[T]:
    """
    Heap wrapper for more efficient finding of top *MAX* n elements in a stream.
    """

    n: int
    _data: list[T]

    def __init__(self, n: int):
        """
        Initialize the TopElements object with a maximum size of n.
        """

        self.n = n
        self._data = []

    def push(self, element: T):
        """
        Push an element into the heap.
        """
        if len(self._data) < self.n:
            heapq.heappush(self._data, element)  # type: ignore
        elif element > self._data[0]:  # type: ignore
            heapq.heapreplace(self._data, element)  # type: ignore

    def pop(self):
        """
        Pop an element from the heap.
        """
        heapq.heappop(self._data)  # type: ignore

    def empty(self) -> bool:
        """
        Check if the heap is empty.
        """
        return len(self._data) == 0

    def top(self) -> T | None:
        """
        Get the top element from the heap.
        """
        if self.empty():
            return None
        return self._data[0]

    def sorted(self) -> list[T]:
        """
        Get sorted top `n` elements.
        """
        return sorted(self._data, reverse=True)  # type: ignore
