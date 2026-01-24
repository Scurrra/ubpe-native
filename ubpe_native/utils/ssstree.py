from .utils import copy


class SSSTreeNode[K: str | tuple[int, ...] | list[int], V]:
    """
    Node of a radix tree.
    """

    key: K
    value: V | None  # `None` only in splits
    children: list["SSSTreeNode[K, V]"]

    def __init__(self, key: K, value: V):
        self.key = key
        self.value = value
        self.children = []

    def __add__(self, element: tuple[K, V]):
        """
        Add new entry to the tree that starts with the current node.
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
            split = SSSTreeNode[K, V](self.key[i:], self.value)  # type: ignore (no `None` here)
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
                    self.children.append(SSSTreeNode[K, V](key, value))  # type: ignore (no `None` here)

            # the new and the old keys have common first i elements
            else:
                split = SSSTreeNode[K, V](self.key[i:], self.value)  # type: ignore (no `None` here)
                split.children = self.children
                self.children = [split, SSSTreeNode[K, V](key, value)]  # type: ignore (no `None` here)
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
        self, key: K, stack: list[tuple[K, V | None]] = None, start: int = 0
    ) -> tuple[K, V | None]:
        """
        Trace `key` by the tree. Finds all entries `(k, v)`, where `key` starts with `k` and `v` is not `None`.
        """
        if stack is None:
            stack = []
        if start + len(self.key) > len(key):
            return stack
        if key[start : (start + len(self.key))] == self.key:
            stack.append((self.key, self.value))
            start += len(self.key)
            if start == len(key):
                return stack
            for child in self.children:
                if child.key[0] == key[start]:
                    child(key, stack, start)
        return stack


class SSSTree[K: str | tuple[int, ...] | list[int], V]:
    """
        SubSequence Search Tree.

    Well, it's a version of an optimized trie but with an efficient search operator `()`
    which return not the full match for the `key`, but all non-null entries
    which keys are prefixes in the `key`.
    """

    children: list["SSSTreeNode[K, V]"]

    def __init__(self):
        self.children = []

    def __add__(self, element: tuple[K, V]):
        """
        Add new entry to the tree.

        Function searches for the elder child subtree (of type `SSSTreeNode[K, V]`) and adds the entry to this subtree.
        If subtree is not found, the new one is created.
        """
        i = 0
        while i < len(self.children):
            if self.children[i].key[0] == element[0][0]:
                _ = self.children[i] + element
                break
            i += 1
        if i == len(self.children):
            self.children.append(SSSTreeNode(*element))

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
        return None

    def __call__(self, key: K, start: int = 0) -> list[tuple[int, V]]:
        """
        Trace `key` by the tree. Finds all entries `(k, v)`, where `key` starts with `k` and `v` is not `None`.
        """
        for child in self.children:
            if child.key[0] != key[start]:
                continue
            
            stack = child(key, start=start)

            if not stack:
                return []

            res = []
            sub_key_len = 0
            for i in range(len(stack)):
                sub_key_len += len(stack[j][0])  # type: ignore
                if stack[i][1] is not None:
                    res.append((sub_key_len, stack[i][1]))    
            return res
        return []
