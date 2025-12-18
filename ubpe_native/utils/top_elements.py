import heapq


class TopElements[T]:
    """
    Heap wrapper for more efficient finding of top *MAX* n elements in a stream.
    """

    n: int
    _data: list[T]

    def __init__(self, n: int, data: list[T] | None = None):
        self.n = n

        if data is None:
            self._data = []
        else:
            self._data = data
            heapq.heapify(self._data)  # type: ignore
            while len(self._data) > n:
                _ = heapq.heappop(self._data)  # type: ignore

    def push(self, element: T):
        if len(self._data) < self.n:
            heapq.heappush(self._data, element)  # type: ignore
        elif element > self._data[0]:  # type: ignore
            heapq.heappushpop(self._data, element)  # type: ignore

    def pop(self):
        heapq.heappop(self._data)  # type: ignore

    def empty(self) -> bool:
        return len(self._data) == 0

    def top(self) -> T | None:
        if self.empty():
            return None
        return self._data[0]

    def sorted(self) -> list[T]:
        """
        Get sorted top `n` elements.
        """
        return sorted(self._data, reverse=True)  # type: ignore
