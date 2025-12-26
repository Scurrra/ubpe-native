from collections import Counter
from math import log

from .ubpe_base import UBPEBase
from .utils import SSSTree, TopElements, PairCounter

try:
    from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]
except ImportError:
    _has_tqdm = False
else:
    _has_tqdm = True


class EncodingCandidate:
    weight: float
    sequence: list[int]
    counter: Counter[int]

    def __init__(
        self,
        weight: float = 0.0,
        sequence: list[int] | None = None,
        counter: Counter[int] | None = None,
    ):
        self.weight = weight

        if sequence is None:
            self.sequence = []
        else:
            self.sequence = sequence

        if counter is None:
            self.counter = Counter(self.sequence)
        else:
            self.counter = counter.copy()

    def __lt__(self, rhs: "EncodingCandidate") -> bool:
        if self.weight == rhs.weight:
            return len(self.sequence) > len(rhs.sequence)
        return self.weight < rhs.weight

    def __le__(self, rhs: "EncodingCandidate") -> bool:
        if self.weight == rhs.weight:
            return len(self.sequence) >= len(rhs.sequence)
        return self.weight <= rhs.weight

    def __gt__(self, rhs: "EncodingCandidate") -> bool:
        if self.weight == rhs.weight:
            return len(self.sequence) < len(rhs.sequence)
        return self.weight > rhs.weight

    def __ge__(self, rhs: "EncodingCandidate") -> bool:
        if self.weight == rhs.weight:
            return len(self.sequence) <= len(rhs.sequence)
        return self.weight >= rhs.weight

    def __call__(self) -> tuple[list[int], float]:
        return (self.sequence, self.weight)


class UBPE[T](UBPEBase[T]):
    _lookup: SSSTree[tuple[int, ...], int]

    def __init__(
        self,
        alphabet_size: int | None = None,
        alphabet: dict[T, int] | None = None,
        n_tokens: int = 2**10,
    ):
        super().__init__(
            alphabet_size=alphabet_size, alphabet=alphabet, n_tokens=n_tokens
        )

    def fit(
        self,
        corpus: list[str | list[T] | tuple[T]],  # pyright: ignore[reportRedeclaration]
        n_candidates: int = 50,
        rearrange_tokens: bool = True,
        use_tqdm: bool = _has_tqdm,
    ):
        """
        Fit tokenizer with `corpus`.

        On each step top `n_candidates` pairs of adjacent tokens are filtered into a list of pairs of tokens,
        where each token is unique. The method does not preserve token's frequency in the corpus and creates
        usually more than `self.n_tokens`, so if `rearrange_tokens` is set to True, tokens are rearanged (tokens
        with lowest numbers are more valueable according to idf metric) and the vocabulary is trimmed to have size `self.n_tokens`.

        Note: this tokenizer differs from `UBPEClassic` in the way the vocabulary is stored. Instead of recursively
        substituting a pair of tokens with another one, a sequence of initial tokens are substituded with the new token.
        """
        if not isinstance(corpus[0], list):
            corpus: list[list[T]] = [list(corpus[i]) for i in range(len(corpus))]  # pyright: ignore[reportAssignmentType, reportRedeclaration]
        corpus: list[list[int]] = [
            [self.alphabet[s] for s in doc]  # pyright: ignore[reportArgumentType]
            for doc in corpus  # pyright: ignore[reportArgumentType]
        ]

        self.tokens_mapper = {
            # subsequences of tokens to a single token
            "forward": dict(),
            # single token to a subsequence of tokens
            "backward": dict(),
        }
        # number of occurrences of each token
        self.tokens_weights = dict()
        # the first token to be added to the mapping minus one
        max_token = self.alphabet_size - 1

        if use_tqdm:
            progress = tqdm(total=self.n_tokens, initial=max_token - 1)  # pyright: ignore[reportPossiblyUnboundVariable, reportUnknownVariableType]
        while max_token < self.n_tokens:
            # compute all bytepairs
            pairs_counter = PairCounter(corpus)
            # find most frequent bytepairs, a.k.a. candidates
            mc = pairs_counter.most_common(n_candidates)
            if len(mc) == 0:
                break

            # find a banch of new tokens
            ## first candidate is always added
            token_pairs = [mc[0]]
            ## each old token may occure only in one candidate
            current_set = set(mc[0][0])
            for i in range(1, len(mc)):
                if len(current_set.intersection(mc[i][0])) != 0:
                    continue

                # check that border pairs are not better
                (l2, r2), n2 = mc[i]
                good_to_add = True
                for (l1, r1), _ in token_pairs:
                    good_to_add = (
                        pairs_counter((r2, l1))[1] < n2
                        and pairs_counter((r1, l2))[1] < n2
                    )
                    if not good_to_add:
                        break

                # finally add candidate if it is good
                if good_to_add:
                    token_pairs.append(mc[i])
                    current_set.update(mc[i][0])

            # merge subsequences for each pair of tokens and add it to the mapings
            mini_mapping: dict[int, tuple[int, list[int]]] = dict()
            for tokens_map, _ in token_pairs:
                (t1, t2) = tokens_map
                max_token += 1
                self.tokens_weights[max_token] = log(
                    (1 + len(corpus)) / (1 + pairs_counter(tokens_map)[0])
                )
                tokens_map: tuple[int, ...] = self.tokens_mapper["backward"].get(  # type: ignore
                    t1, (t1,)
                ) + self.tokens_mapper["backward"].get(t2, (t2,))  # pyright: ignore[reportAssignmentType, reportOperatorIssue]
                self.tokens_mapper["backward"][max_token] = tokens_map
                mini_mapping[t1] = (t2, [max_token])

            corpus = [
                self._replace_token_pairs(corpus[i], mini_mapping)
                for i in range(len(corpus))
            ]

            if use_tqdm:
                progress.update(len(token_pairs))  # pyright: ignore[reportPossiblyUnboundVariable]
        if use_tqdm:
            progress.close()  # pyright: ignore[reportPossiblyUnboundVariable]

        if rearrange_tokens:
            self._rearrange_tokens_by_weight()

        self.tokens_mapper["forward"] = {
            seq: token for token, seq in self.tokens_mapper["backward"].items()
        }

        self._lookup = SSSTree[tuple[int], int]()
        for key in self.inverse_alphabet.keys():
            _ = self._lookup + ((key,), key)
        for key, value in self.tokens_mapper["forward"].items():
            _ = self._lookup + (key, value)  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]

    def encode(
        self,
        doc: str | list[T] | tuple[T],  # pyright: ignore[reportRedeclaration]
        top_n: int = 1,
    ) -> list[tuple[list[int], float]]:
        """
        Encode `doc` with fitted tokenizer.

        Note: "classic" approach is much simpler for encoding but can produce only one variant of the
        encoded sequence. This implementation allows to select `top_n` code candidates according to the
        tf-idf metric.
        """
        assert self._lookup is not None, "Tokenizer is not fitted"
        assert isinstance(doc, str) or isinstance(
            doc, list
        ), "Data can only be a list or a string"
        doc: tuple[int, ...] = tuple(self.alphabet[token] for token in doc)  # pyright: ignore[reportArgumentType]

        # build initial stack
        start: int = 0
        stacks: list[tuple[int, list[tuple[tuple[int, ...], int]]]] = []
        while start < len(doc):
            stack = self._lookup(doc, start)
            stacks.append((start, stack))
            start += len(stack[-1][0])

        # build SSSTreeNodes
        SSSTreeNodes: dict[int, dict[tuple[int, ...], tuple[int, int]]] = dict()
        while len(stacks) != 0:
            start, stack = stacks.pop()
            next: dict[tuple[int, ...], tuple[int, int]] = dict()
            for key, value in stack:
                next_key_start = start + len(key)
                next[key] = (value, next_key_start)
                if next_key_start != len(doc) and next_key_start not in SSSTreeNodes:
                    stacks.append((next_key_start, self._lookup(doc, next_key_start)))
            SSSTreeNodes[start] = next

        ## clean hanging SSSTreeNodes
        ## redundant step
        # SSSTreeNodes_to_delete: list[int] = []
        # for SSSTreeNode_start, SSSTreeNode in SSSTreeNodes.items():
        #     keys_to_delete: list[tuple[int, ...]] = []
        #     for key, (_, start) in SSSTreeNode.items():
        #         if start != len(doc) and start not in SSSTreeNodes:
        #             keys_to_delete.append(key)
        #     for key in keys_to_delete:
        #         del SSSTreeNode[key]
        #     if len(SSSTreeNode) == 0:
        #         SSSTreeNodes_to_delete.append(SSSTreeNode_start)
        # for start in SSSTreeNodes_to_delete:
        #     del SSSTreeNodes[start]

        starts = sorted(SSSTreeNodes.keys(), reverse=True)
        tails: dict[int, list[EncodingCandidate]] = {len(doc): [EncodingCandidate()]}
        for start in starts:
            buf = TopElements[EncodingCandidate](top_n)
            for token, next_start in SSSTreeNodes[start].values():
                for candidate in tails[next_start]:
                    buf_element = [token] + candidate.sequence.copy()
                    buf_counter = candidate.counter.copy()
                    buf_counter.update([token])
                    buf_weight: float = sum(
                        (1 + log(frequency)) * self.tokens_weights.get(token, 0.0)
                        for token, frequency in buf_counter.items()
                    )
                    buf.push(EncodingCandidate(buf_weight, buf_element, buf_counter))
            tails[start] = buf.sorted()
        candidates = tails[0]

        return [candidate() for candidate in candidates]

    def decode(self, tokens: list[int]):
        """
        Decode a list of `tokens` with the fitted tokenizer.
        """
        assert self._lookup is not None, "Tokenizer is not fitted"
        result: list[int] = []
        for token in tokens:
            if token in self.tokens_mapper["backward"]:
                result.extend(self.tokens_mapper["backward"][token])  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
            else:
                result.append(token)  # pyright: ignore[reportUnknownMemberType]
        return [self.inverse_alphabet[token] for token in result]

    @classmethod
    def loads(cls, dump: str, token_type: type = int):
        """
        Load a tokenizer model from a json-serialized string.
        """
        inst = super().loads(dump, token_type=token_type)

        inst._lookup = SSSTree[tuple[int], int]()
        for key in inst.inverse_alphabet.keys():
            _ = inst._lookup + ((key,), key)
        for key, value in inst.tokens_mapper["forward"].items():
            _ = inst._lookup + (key, value)  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]

        return inst
