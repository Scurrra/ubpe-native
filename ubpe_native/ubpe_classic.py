from collections import Counter
from itertools import pairwise
from math import log

from .ubpe_base import UBPEBase
from .utils import Logger, PairCounter, SplitMode


class UBPEClassic[T](UBPEBase[T]):
    _pairs: list[tuple[int, int]]

    def __init__(
        self,
        *,
        alphabet: dict[T, int] | list[T] | set[T] | None = None,
        n_tokens: int = 2**10,
        known_words: list | dict | None = None,
        break_tokens: set | list | None = None,
        regex_str: str | None = None,
        stop_tokens: set | list | None = None,
    ):
        super().__init__(
            alphabet=alphabet,
            n_tokens=n_tokens,
            known_words=known_words,
            break_tokens=break_tokens,
            regex_str=regex_str,
            stop_tokens=stop_tokens,
        )

    def fit(
        self,
        corpus: list[str | list[T] | tuple[T, ...]],  # pyright: ignore[reportRedeclaration]
        *,
        n_candidates: int = 50,
        rearrange_tokens: bool = True,
        quiet: bool = False,
        split_mode: SplitMode = SplitMode.FULL,
    ):
        """
        Fit tokenizer with `corpus`.

        On each step top `n_candidates` pairs of adjacent tokens are filtered into a list of pairs of tokens,
        where each token is unique. The method does not preserve token's frequency in the corpus and creates
        usually more than `self.n_tokens`, so if `rearrange_tokens` is set to True, tokens are rearanged (tokens
        with lowest numbers are more valueable according to idf metric) and the vocabulary is trimmed to have size `self.n_tokens`.

        Note: "classic" means that the vocabulary maps a pair of tokens to a new token.
        """
        if n_candidates < 1:
            raise ValueError("`n_candidates` should be greater than 0")

        logger = Logger(scope="UBPEClassic.fit", quiet=quiet, unit="token")
        logger.info("Starting fitting process")

        corpus: list[list[list[int]]] = [
            self.split_pipeline(doc, mode=split_mode, leave_separators=False)
            for doc in corpus
        ]
        logger.info("Loaded the corpus")

        self.tokens_mapper = {
            # subsequences of tokens to a single token
            "forward": dict(),
            # single token to a subsequence of tokens
            "backward": dict(),
        }
        # number of occurrences of each token
        self.tokens_weights = dict()
        # the first token to be added to the mapping minus one
        max_token = (
            len(self.alphabet)
            + (len(self.known_words) if self.known_words is not None else 0)
            - 1
        )

        logger.info("Starting token building")
        logger.progress(total=self.n_tokens, initial=max_token + 1)
        logger.progress.run()
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
                self._replace_token_pairs(corpus[i], mini_mapping)  # type: ignore
                for i in range(len(corpus))
            ]

            logger.progress.update(len(token_pairs))
        logger.progress.stop()
        logger.info(f"Built {len(self.tokens_mapper['backward'])} artificial tokens")

        if rearrange_tokens:
            self._rearrange_tokens_by_weight(is_classic=True)
            logger.info(
                f"Rearranged artificial tokens: {len(self.tokens_mapper['backward'])} left"
            )
        self.n_tokens = len(self.alphabet) + len(self.tokens_weights)
        if self.known_words is not None:
            self.n_tokens += len(self.known_words)

        self.tokens_mapper["forward"] = {
            seq: token for token, seq in self.tokens_mapper["backward"].items()
        }

        self._pairs = list(self.tokens_mapper["forward"].keys())  # type: ignore
        logger.info("Cached pairs for faster encoding")

    def encode(
        self,
        doc: str | list[T] | tuple[T, ...],  # pyright: ignore[reportRedeclaration]
        *,
        split_mode: SplitMode = SplitMode.FULL,
    ) -> list[tuple[list[int], float]]:
        """
        Encode `doc` with fitted tokenizer.

        Note: on each step instead of substituting a single pair of tokens, a list of pairs of tokens
        from the vocabulary that can be substituded independently is selected and used.
        """
        if self._pairs is None:
            raise ValueError("Tokenizer is not fitted")
        if not isinstance(doc, (str, list, tuple)):
            raise ValueError("Data can only be a list or a string")

        parts = self.split_pipeline(doc, mode=split_mode)

        if len(parts) == 1:
            return self._encode_word(parts[0])

        doc: list[int] = []
        weight: float = 0.0
        for part in parts:
            if len(part) == 1:
                doc.append(part[0])
                # known words, break and stop tokens have weight 0.0
                # so `weight += 0.0` is not needed
            else:
                encoding = self._encode_word(part)[0]
                doc.extend(encoding[0])
                weight += encoding[1]
        return [(doc, weight)]

    def _encode_word(
        self,
        word: list[int],  # pyright: ignore[reportRedeclaration]
    ) -> list[tuple[list[int], float]]:
        """
        Encode `word` with fitted tokenizer.

        Note: on each step instead of substituting a single pair of tokens, a list of pairs of tokens
        from the vocabulary that can be substituded independently is selected and used.
        """
        if self._pairs is None:
            raise ValueError("Tokenizer is not fitted")
        if not isinstance(word, list):
            raise ValueError("Data can only be a list")

        while True:
            pairs = set(pairwise(word))

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
            }  # type: ignore
            word = self._replace_token_pairs(word, mini_mapping)  # type: ignore

        counter = Counter(word)
        weight: float = sum(
            (1 + log(quantity)) * self.tokens_weights.get(token, 0.0)  # type: ignore
            for token, quantity in counter.items()
        )

        return [(word, weight)]  # type: ignore

    def decode(self, tokens: list[int]) -> list[T] | T:
        """
        Decode a list of `tokens` with the fitted tokenizer.
        """
        if self._pairs is None:
            raise ValueError("Tokenizer is not fitted")

        result = tokens.copy()

        i = 0
        while i < len(result):
            if result[i] in self.tokens_mapper["backward"]:
                result[i : i + 1] = self.tokens_mapper["backward"][result[i]]  # type: ignore
            else:
                i += 1

        doc: list[T] = []
        if self.inverse_known_words is not None:
            for token in result:
                if token in self.inverse_alphabet:
                    doc.append(self.inverse_alphabet[token])
                elif token in self.inverse_known_words:
                    doc.extend(self.inverse_known_words[token])  # type: ignore
                else:
                    raise ValueError(f"Unknown token {token}")
        else:
            for token in result:
                if token in self.inverse_alphabet:
                    doc.append(self.inverse_alphabet[token])
                else:
                    raise ValueError(f"Unknown token {token}")

        if isinstance(doc[0], str):
            return "".join(doc)  # type: ignore
        return doc

    @classmethod
    def loads(cls, dump: str, token_type: type = int):
        """
        Load a tokenizer model from a json-serialized string.
        """
        inst = super().loads(dump, token_type=token_type)

        inst._pairs = list(inst.tokens_mapper["forward"].keys())  # type: ignore

        return inst
