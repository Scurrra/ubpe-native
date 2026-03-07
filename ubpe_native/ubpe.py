from collections import Counter
from math import log

from .ubpe_base import UBPEBase
from .utils import Logger, PairCounter, SplitMode, SSSTree, TopElements


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
        *,
        alphabet_size: int | None = None,
        alphabet: dict[T, int] | None = None,
        n_tokens: int = 2**10,
        known_words: list | dict | None = None,
        break_tokens: set | list | None = None,
        regex_str: str | None = None,
        stop_tokens: set | list | None = None,
    ):
        super().__init__(
            alphabet_size=alphabet_size,
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

        Note: this tokenizer differs from `UBPEClassic` in the way the vocabulary is stored. Instead of recursively
        substituting a pair of tokens with another one, a sequence of initial tokens are substituded with the new token.
        """
        logger = Logger(scope="UBPE.fit", quiet=quiet, unit="token")
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
        max_token = self.alphabet_size - 1

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
                self._replace_token_pairs(corpus[i], mini_mapping)  # type: ignore
                for i in range(len(corpus))
            ]

            logger.progress.update(len(token_pairs))
        logger.progress.stop()
        logger.info(f"Built {len(self.tokens_mapper['backward'])} artificial tokens")

        if rearrange_tokens:
            self._rearrange_tokens_by_weight()
            logger.info(
                f"Rearranged artificial tokens: {len(self.tokens_mapper['backward'])} left"
            )

        self.tokens_mapper["forward"] = {
            seq: token for token, seq in self.tokens_mapper["backward"].items()
        }

        self._lookup = SSSTree[tuple[int], int]()
        for key in self.inverse_alphabet.keys():
            _ = self._lookup + ((key,), key)
        for key, value in self.tokens_mapper["forward"].items():
            _ = self._lookup + (key, value)  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]
        logger.info("Built the lookup tree")

    def encode(
        self,
        doc: str | list[T] | tuple[T, ...],  # pyright: ignore[reportRedeclaration]
        *,
        split_mode: SplitMode = SplitMode.FULL,
        top_n: int = 1,
    ) -> list[tuple[list[int], float]]:
        """
        Encode `doc` with fitted tokenizer.

        Note: "classic" approach is much simpler for encoding but can produce only one variant of the
        encoded sequence. This implementation allows to select `top_n` code candidates according to the
        tf-idf metric.
        """
        if self._lookup is None:
            raise ValueError("Tokenizer is not fitted")
        if not (isinstance(doc, str) or isinstance(doc, list)):
            raise ValueError("Data can only be a list or a string")
        if top_n < 1:
            raise ValueError("`top_n` must be at least one")

        parts = self.split_pipeline(doc, mode=split_mode)

        if len(parts) == 1:
            return self._encode_word(parts[0], top_n=top_n)

        if top_n == 1:
            doc: list[int] = []
            weight: float = 0.0
            # as we select only the best encoding, we should just concatenate the encodings of the parts
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

        # general case
        tails: list[tuple[list[int], float]] = [([], 0.0)]

        # iterate backwards
        for part in parts[::-1]:
            if len(part) == 1:
                tails = [(part + tail, weight) for tail, weight in tails]
            else:
                candidates = self._encode_word(part, top_n=top_n)
                ti, ci = 0, 0

                new_tails: list[tuple[list[int], float]] = []
                while (
                    ci < len(candidates) and ti < len(tails) and len(new_tails) < top_n
                ):
                    new_tails.append(
                        (
                            candidates[ci][0] + tails[ti][0],
                            candidates[ci][1] + tails[ti][1],
                        )
                    )

                    if len(new_tails) == top_n:
                        break

                    if ci == len(candidates) - 1 and ti == len(tails) - 1:
                        break

                    if ci == len(candidates) - 1 and ti < len(tails) - 1:
                        ti += 1
                    elif ti == len(tails) - 1 and ci < len(candidates) - 1:
                        ci += 1
                    else:
                        # here we use `>` to prioritize new candidates, i.e. to enhance diversity
                        if (
                            tails[ti + 1][1] + candidates[ci][1],
                            -len(tails[ti + 1][0]) - len(candidates[ci][0]),
                        ) > (
                            tails[ti][1] + candidates[ci + 1][1],
                            -len(tails[ti][0]) - len(candidates[ci + 1][0]),
                        ):
                            ti += 1
                        else:
                            ci += 1

                tails = new_tails

        return tails

    def _encode_word(
        self,
        word: list[int] | tuple[int, ...],  # pyright: ignore[reportRedeclaration]
        *,
        top_n: int = 1,
    ) -> list[tuple[list[int], float]]:
        """
        Encode `word` with fitted tokenizer.

        Note: "classic" approach is much simpler for encoding but can produce only one variant of the
        encoded sequence. This implementation allows to select `top_n` code candidates according to the
        tf-idf metric.
        """
        if self._lookup is None:
            raise ValueError("Tokenizer is not fitted")
        if not isinstance(word, (list, tuple)):
            raise ValueError("Data can only be a tuple")
        word = tuple(word)

        # build initial stack
        start: int = 0
        stacks: list[tuple[int, list[tuple[int, int]]]] = []
        while start < len(word):
            stack = self._lookup(word, start, fast=True)
            stacks.append((start, stack))  # type: ignore
            start += stack[-1][0]  # type: ignore

        # build SSSTreeNodes
        SSSTreeNodes: dict[int, dict[int, tuple[int, int]]] = dict()
        while len(stacks) != 0:
            start, stack = stacks.pop()
            next: dict[int, tuple[int, int]] = dict()
            for key_len, value in stack:
                next_key_start = start + key_len
                next[key_len] = (value, next_key_start)
                if next_key_start != len(word) and next_key_start not in SSSTreeNodes:
                    stacks.append(
                        (next_key_start, self._lookup(word, next_key_start, fast=True))  # type: ignore
                    )
            SSSTreeNodes[start] = next

        ## clean hanging SSSTreeNodes
        ## redundant step
        # SSSTreeNodes_to_delete: list[int] = []
        # for SSSTreeNode_start, SSSTreeNode in SSSTreeNodes.items():
        #     keys_to_delete: list[tuple[int, ...]] = []
        #     for key, (_, start) in SSSTreeNode.items():
        #         if start != len(word) and start not in SSSTreeNodes:
        #             keys_to_delete.append(key)
        #     for key in keys_to_delete:
        #         del SSSTreeNode[key]
        #     if len(SSSTreeNode) == 0:
        #         SSSTreeNodes_to_delete.append(SSSTreeNode_start)
        # for start in SSSTreeNodes_to_delete:
        #     del SSSTreeNodes[start]

        starts = sorted(SSSTreeNodes.keys(), reverse=True)
        tails: dict[int, list[EncodingCandidate]] = {len(word): [EncodingCandidate()]}
        if top_n == 1:
            for start in starts:
                best: EncodingCandidate | None = None
                for token, next_start in SSSTreeNodes[start].values():
                    candidate = tails[next_start][0]
                    buf_element = [token] + candidate.sequence.copy()
                    buf_counter = candidate.counter.copy()
                    buf_counter.update([token])
                    buf_weight = sum(
                        (1 + log(frequency)) * self.tokens_weights.get(token, 0.0)
                        for token, frequency in buf_counter.items()
                    )

                    if best is None:
                        best = EncodingCandidate(buf_weight, buf_element, buf_counter)
                    else:
                        if (
                            best.weight == buf_weight
                            and len(best.sequence) > len(buf_element)
                        ) or best.weight < buf_weight:
                            best = EncodingCandidate(
                                buf_weight, buf_element, buf_counter
                            )
                tails[start] = [best]  # type: ignore
        else:
            for start in starts:
                buf = TopElements[EncodingCandidate](top_n)
                for token, next_start in SSSTreeNodes[start].values():
                    for candidate in tails[next_start]:
                        buf_element = [token] + candidate.sequence.copy()
                        buf_counter = candidate.counter.copy()
                        buf_counter.update([token])
                        buf_weight = sum(
                            (1 + log(frequency)) * self.tokens_weights.get(token, 0.0)
                            for token, frequency in buf_counter.items()
                        )

                        buf.push(
                            EncodingCandidate(buf_weight, buf_element, buf_counter)
                        )
                tails[start] = buf.sorted()
        candidates = tails[0]

        return [candidate() for candidate in candidates]

    def decode(self, tokens: list[int]) -> list[T] | T:
        """
        Decode a list of `tokens` with the fitted tokenizer.
        """
        if self._lookup is None:
            raise ValueError("Tokenizer is not fitted")

        result: list[int] = []
        for token in tokens:
            if token in self.tokens_mapper["backward"]:
                result.extend(self.tokens_mapper["backward"][token])  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
            else:
                result.append(token)  # pyright: ignore[reportUnknownMemberType]

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

        inst._lookup = SSSTree[tuple[int], int]()
        for key in inst.inverse_alphabet.keys():
            _ = inst._lookup + ((key,), key)
        for key, value in inst.tokens_mapper["forward"].items():
            _ = inst._lookup + (key, value)  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]

        return inst
