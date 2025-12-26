from collections import Counter
from itertools import pairwise
from math import log

from .ubpe_base import UBPEBase
from .utils import PairCounter

try:
    from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]
except ImportError:
    _has_tqdm = False
else:
    _has_tqdm = True


class UBPEClassic[T](UBPEBase[T]):
    _pairs: list[tuple[int, int]]

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

        Note: "classic" means that the vocabulary maps a pair of tokens to a new token.
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

            # add new pair mapping
            mini_mapping: dict[int, tuple[int, list[int]]] = dict()
            for tokens_map, _ in token_pairs:
                max_token += 1
                self.tokens_weights[max_token] = log(
                    (1 + len(corpus)) / (1 + pairs_counter(tokens_map)[0])
                )
                self.tokens_mapper["backward"][max_token] = tokens_map  # pyright: ignore[reportArgumentType]
                mini_mapping[tokens_map[0]] = (tokens_map[1], [max_token])

            corpus = [
                self._replace_token_pairs(corpus[i], mini_mapping)
                for i in range(len(corpus))
            ]

            del pairs_counter

            if use_tqdm:
                progress.update(len(token_pairs))  # pyright: ignore[reportPossiblyUnboundVariable]

        if use_tqdm:
            progress.close()  # pyright: ignore[reportPossiblyUnboundVariable]

        if rearrange_tokens:
            self._rearrange_tokens_by_weight()

        self.tokens_mapper["forward"] = {
            seq: token for token, seq in self.tokens_mapper["backward"].items()
        }

        self._pairs = list(self.tokens_mapper["forward"].keys())  # type: ignore

    def encode(self, doc: str | list[T] | tuple[T]) -> list[tuple[list[int], float]]:  # pyright: ignore[reportRedeclaration]
        """
        Encode `doc` with fitted tokenizer.

        Note: on each step instead of substituting a single pair of tokens, a list of pairs of tokens
        from the vocabulary that can be substituded independently is selected and used.
        """
        assert self._pairs is not None, "Tokenizer is not fitted"
        assert isinstance(doc, str) or isinstance(
            doc, list
        ), "Data can only be a list or a string"
        doc: list[int] = [self.alphabet[token] for token in doc]  # pyright: ignore[reportArgumentType]

        while True:
            pairs = set(pairwise(doc))

            i = 0
            while i < len(self._pairs) and self._pairs[i] not in pairs:
                i += 1
            if i == len(self._pairs):
                break
            tokens = [self._pairs[i]]
            current_set = set(tokens[-1])

            for j in range(i + 1, len(self._pairs)):
                if len(current_set.intersection(self._pairs[j])) != 0:
                    break
                # if self._pairs[j] not in pairs:    break
                if self._pairs[j] in pairs:
                    tokens.append(self._pairs[j])
                    current_set.update(self._pairs[j])

            mini_mapping: dict[int, tuple[int, list[int]]] = {
                pair[0]: (pair[1], [self.tokens_mapper["forward"][pair]])
                for pair in tokens
            }  # pyright: ignore[reportAssignmentType]
            doc = self._replace_token_pairs(doc, mini_mapping)

        counter = Counter(doc)
        weight = sum(
            (1 + log(quantity)) * self.tokens_weights.get(token, 0.0)
            for token, quantity in counter.items()
        )

        return [(doc, weight)]

    def decode(self, tokens: list[int]):
        """
        Decode a list of `tokens` with the fitted tokenizer.
        """
        assert self._pairs is not None, "Tokenizer is not fitted"
        i = 0
        while i < len(tokens):
            if tokens[i] in self.tokens_mapper["backward"]:
                tokens[i : i + 1] = self.tokens_mapper["backward"][tokens[i]]  # type: ignore
            else:
                i += 1
        return [self.inverse_alphabet[token] for token in tokens]

    @classmethod
    def loads(cls, dump: str, token_type: type = int):
        """
        Load a tokenizer model from a json-serialized string.
        """
        inst = super().loads(dump, token_type=token_type)

        inst._pairs = list(inst.tokens_mapper["forward"].keys())  # type: ignore

        return inst
