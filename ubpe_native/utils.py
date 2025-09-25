def copy[T](smth: str | tuple[T, ...] | list[T]) -> str | tuple[T, ...] | list[T]:
    """
    Universal function for copying. Supports strings, tuples, and lists.
    """
    return (
        smth
        if isinstance(smth, str)
        else tuple(smth)
        if isinstance(smth, tuple)
        else smth.copy()
    )


def join[T](
    *smth: str | tuple[T, ...] | list[T],
) -> str | tuple[T, ...] | list[T] | None:
    """
    Join a sequence of sequences (same type) into a single sequence. Supports strings, tuples, and lists.
    """
    if len(smth) == 0:
        return None

    eltype = type(smth[0])
    # ensure that the type of each argument is the same
    for i in range(1, len(smth)):
        # if not, return `None`
        if eltype is not type(smth[i]):
            return None

    # strings are simply joined
    if eltype is str:
        return "".join(smth)  # type: ignore

    # if arguments are not of type the function was created for
    if eltype is not tuple and eltype is not list:
        # just return a tuple from `*smth`
        return tuple(smth)  # type: ignore

    # for effective copying:
    # 1. find the length of result as sum of length of all the elements
    length = 0
    for i in range(len(smth)):
        length += len(smth[0])
    # 2. construct list of `None`s
    result = [None] * length
    # 3. `smth` by `smth` initialize `result`
    start = 0
    for i in range(len(smth)):
        result[start : (start + len(smth[i]))] = smth[i]  # type: ignore (lengths of a sublist and `smth[i]` are guaranteed to be the same)
        start += len(smth[i])
    # 4. keep `result` a list or convert it to a tuple
    return result if eltype is list else tuple(result)  # type: ignore (`None` was already return at this point)


class Node[K: str | tuple[int, ...] | list[int], V]:
    """
    Node of a radix tree.
    """

    key: K
    value: V | None  # `None` only in splits
    children: list["Node[K, V]"]

    def __init__(self, key: K, value: V):
        self.key = key
        self.value = value
        self.children = []

    def __add__(self, element: tuple[K, V]):
        """
        Add new entry to the tree that starts with `Node`.
        """
        (key, value) = element

        i = 0
        max_len = min(len(self.key), len(key))
        while i < max_len and self.key[i] == key[i]:
            i += 1

        # key to insert is in the tree
        if i == len(key):
            # equal keys
            if i == len(self.key):
                if self.value is None:
                    self.value = value
                return self.value == value

            # split vertex in two
            split = Node[K, V](self.key[i:], self.value)  # type: ignore (no `None` here)
            split.children = self.children
            self.children = [split]
            self.key = key  # same as self.key[:i]
            self.value = value

        # part of a key is in the tree
        else:
            key = key[i:]

            # the new key starts with the old one
            if i == len(self.key):
                is_new = True
                for child in self.children:
                    if child.key[0] == key[0]:
                        _ = child + (key, value)  # type: ignore
                        is_new = False
                        break
                if is_new:
                    self.children.append(Node[K, V](key, value))  # type: ignore (no `None` here)

            # the new and the old keys have common first i elements
            else:
                split = Node[K, V](self.key[i:], self.value)  # type: ignore (no `None` here)
                split.children = self.children
                self.children = [split, Node[K, V](key, value)]  # type: ignore (no `None` here)
                self.key = self.key[:i]  # type: ignore
                self.value = None

    def __getitem__(self, key: K) -> V | None:
        """
        Get the value from the tree for the provided key. If not found, `None` is returned.
        """
        if key == self.key:
            return self.value
        if key[: len(self.key)] == self.key:
            key = key[len(self.key) :]  # type: ignore
            for child in self.children:
                if child.key[0] == key[0]:
                    return child[key]
        return None

    def __call__(
        self, key: K, stack: list[tuple[K, V | None]], start: int = 0
    ) -> tuple[K, V | None]:
        """
        Trace `key` by the tree. Finds all entries `(k, v)`, where `key` starts with `k` and `v` is not `None`.
        """
        if key[start : (start + len(self.key))] == self.key:
            stack.append((self.key, self.value))
            start += len(self.key)
            if start >= len(key):
                return stack[-1]
            for child in self.children:
                if child.key[0] == key[start]:
                    _ = child(key, stack, start)
        return stack[-1]


class Root[K: str | tuple[int, ...] | list[int], V]:
    """
    Root of a radix tree.
    """

    children: list["Node[K, V]"]

    def __init__(self):
        self.children = []

    def __add__(self, element: tuple[K, V]):
        """
        Add new entry to the tree.

        Function searches for the elder child subtree (of type `Node[K, V]`) and adds the entry to this subtree.
        If subtree is not found, the new one is created.
        """
        i = 0
        while i < len(self.children):
            if self.children[i].key[0] == element[0][0]:
                _ = self.children[i] + element
                break
            i += 1
        if i == len(self.children):
            self.children.append(Node(*element))

        return True

    def __getitem__(self, key: K) -> V | None:
        """
        Get the value from the tree for the provided key. If not found, `None` is returned.
        """
        i = 0
        while i < len(self.children):
            if self.children[i].key[0] == key[0]:
                return self.children[i][key]
            i += 1
        if i == len(self.children):
            return None

    def __call__(self, key: K, start: int = 0) -> list[tuple[K, V]]:
        """
        Trace `key` by the tree. Finds all entries `(k, v)`, where `key` starts with `k` and `v` is not `None`.
        """
        i = 0
        while i < len(self.children):
            if self.children[i].key[0] == key[start]:
                stack: list[tuple[K, V | None]] = []
                _ = self.children[i](key, stack, start)
                if len(stack) > 0:
                    sub_key: K = copy(stack[0][0])  # type: ignore
                    for j in range(1, len(stack)):
                        sub_key += stack[j][0]  # type: ignore
                        stack[j] = (  # type: ignore
                            (copy(sub_key), None)
                            if stack[j][1] is None
                            else (copy(sub_key), stack[j][1])
                        )
                return [s for s in stack if s[1] is not None]  # type: ignore (no `None` here)
            i += 1
        return []
