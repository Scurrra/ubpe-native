from collections import Counter
from itertools import pairwise


class PairCounter:
    _pairs_counter: Counter[tuple[int, int]]
    _docs_counter: Counter[tuple[int, int]]

    def __init__(self, corpus: list[list[int]] | list[int] | None = None) -> None:
        self._docs_counter = Counter()
        self._pairs_counter = Counter()

        if corpus is None:
            return

        assert isinstance(
            corpus, list
        ), "`corpus` should be a list of documents or a docment (list of tokens) itself"

        if len(corpus) == 0:
            return

        if isinstance(corpus[0], list):
            for doc in corpus:
                self.update(doc)  # type: ignore
        else:
            self.update(corpus)  # type: ignore

    def update(self, doc: list[int]):
        self._pairs_counter.update(pairwise(doc))
        self._docs_counter.update(set(pairwise(doc)))

    def most_common(self, n: int) -> list[tuple[tuple[int, int], int]]:
        return self._pairs_counter.most_common(n)

    def __call__(self, pair: tuple[int, int]) -> tuple[int, int]:
        return (self._docs_counter[pair], self._pairs_counter[pair])
