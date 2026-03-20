from collections import Counter
from heapq import nlargest
from itertools import pairwise


class PairCounter:
    _pairs_counter: Counter[tuple[int, int]]
    _docs_counter: Counter[tuple[int, int]]

    def __init__(
        self, corpus: list[list[list[int]]] | list[list[int]] | list[int] | None = None
    ) -> None:
        self._docs_counter = Counter()
        self._pairs_counter = Counter()

        if corpus is None:
            return

        if not isinstance(corpus, list):
            raise ValueError(
                "`corpus` should be a list of documents or a docment (list of tokens) itself"
            )

        if len(corpus) == 0:
            return

        if isinstance(corpus[0], list):
            for doc in corpus:
                self.update(doc)  # type: ignore
        else:
            self.update(corpus)  # type: ignore

    def update(self, doc: list[list[int]] | list[int]):
        if not isinstance(doc, list):
            raise ValueError("`doc` should be a list of tokens or a list of documents")

        if len(doc) == 0:
            return

        if isinstance(doc[0], list):
            unique_pairs = set()
            for part in doc:
                self._pairs_counter.update(pairwise(part))  # type: ignore
                unique_pairs.update(pairwise(part))  # type: ignore
            self._docs_counter.update(unique_pairs)
        else:
            # old logic, each document is a list of tokens
            self._pairs_counter.update(pairwise(doc))  # type: ignore
            self._docs_counter.update(set(pairwise(doc)))  # type: ignore

    def most_common(self, n: int) -> list[tuple[tuple[int, int], int]]:
        return nlargest(
            n,
            self._pairs_counter.items(),
            key=lambda pair_count: (
                (pair_count[1], -self._docs_counter[pair_count[0]]),
                pair_count[0],
            ),
        )

    def __call__(self, pair: tuple[int, int]) -> tuple[int, int]:
        return (self._docs_counter[pair], self._pairs_counter[pair])
