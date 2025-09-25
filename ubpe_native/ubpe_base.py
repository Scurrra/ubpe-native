import json


class UBPEBase[T]:
    n_tokens: int
    alphabet_size: int
    alphabet: dict[T, int]
    inverse_alphabet: dict[int, T]
    tokens_mapper: dict[str, dict[int | tuple[int, ...], tuple[int, ...] | int]]
    tokens_weights: dict[int, float]

    def __init__(
        self,
        alphabet_size: int | None = None,
        alphabet: dict[T, int] | None = None,
        n_tokens: int = 2**10,
    ):
        if alphabet is None and alphabet_size is None:
            print(
                "Either `alphabet_size` or `alphabet` must be specified, or model should be load from json string"
            )
            return

        # if `alphabet_size` is provided and `alphabet` is not, `T` is assumed to be `int`
        if alphabet is None:
            alphabet = {i: i for i in range(alphabet_size)}  # type: ignore

        # ensure that `alphabet` is a dict
        else:
            assert isinstance(
                alphabet, dict
            ), "If `alphabet` is provided, it must be a dict"

        if alphabet_size is None:
            alphabet_size = len(alphabet)  # type: ignore (`alphabet` could not be `None` till here)

        self.alphabet_size = alphabet_size
        self.alphabet = alphabet  # type: ignore (`alphabet` could not be `None` till here)
        self.inverse_alphabet = {value: key for key, value in self.alphabet.items()}
        self.n_tokens = n_tokens

    def _replace_token_pairs(self, l: list[int], sub: dict[int, tuple[int, list[int]]]):  # noqa: E741
        """
        Function for replacing pair of adjacent tokens in a list witha new one.

        Args:
        - `l (list)`: A list to be transformed.
        - `sub (dict[int, tuple[int, list[int]]])`: A substitution map, where keys
        are first tokens in the pairs, and the values are pair of the second token
        and the new one wrapped in a list.
        """
        is_not_start = {key: False for key in list(sub.keys())}
        i = -1
        while i < len(l) - 2:
            i += 1
            if is_not_start.get(l[i], True):
                continue
            start = l[i]
            if l[i + 1] == sub[start][0]:
                l[i : i + 2] = sub[start][1]
        return l

    def _rearrange_tokens_by_weight(self):
        """
        Function that rearranges found tokens according to their weights and trims
        dictionary of the tokenizer to be not greater than `self.n_tokens`.
        """
        assert self.tokens_weights is not None, "Tokenizer is not fitted"
        buf = sorted(
            list(self.tokens_mapper["backward"].items()),
            key=lambda item: self.tokens_weights[item[0]],  # type: ignore (`item[0]` is guaranteed to be of type int)
        )

        to_delete: list[int] = []
        for i in range(len(buf)):
            if i in to_delete:
                continue
            if (
                len(to_delete)
                >= len(self.tokens_weights) - self.n_tokens + self.alphabet_size
            ):
                break
            to_delete.append(i)
            token = buf[i][0]
            for j in range(i + 1, len(buf)):
                if token in buf[j][1]:  # type: ignore (`buf[_][1]` is guaranteed to be of type `tuple[int]`)
                    to_delete.append(j)
        to_delete = [buf[i][0] for i in to_delete]  # type: ignore (`buf[_][0]` is guaranteed to be of type `int`)
        buf = buf[::-1]

        transformer = {buf[i][0]: self.alphabet_size + i for i in range(len(buf))}

        self.tokens_weights = {
            transformer[pair[0]]: self.tokens_weights[pair[0]]  # type: ignore (`pair[0]` is guaranteed to be of type int)
            for pair in buf
            if pair[0] not in to_delete
        }

        self.tokens_mapper = {  # type: ignore
            "forward": dict(
                sorted(
                    [
                        (
                            tuple(transformer.get(t, t) for t in seq),  # type: ignore (`seq` is guaranteed to be of type `tuple[int]`)
                            transformer.get(token, token),
                        )
                        for seq, token in self.tokens_mapper["forward"].items()
                        if token not in to_delete
                    ],
                    key=lambda item: item[1],
                )
            ),
            "backward": dict(
                sorted(
                    [
                        (
                            transformer.get(token, token),
                            tuple(transformer.get(t, t) for t in seq),  # type: ignore (`seq` is guaranteed to be of type `tuple[int]`)
                        )
                        for token, seq in self.tokens_mapper["backward"].items()
                        if token not in to_delete
                    ],
                    key=lambda item: item[0],
                )
            ),
        }

    def dumps(self) -> str:
        """
        Dumps model to a string.
        """
        return json.dumps(
            {
                "n_tokens": self.n_tokens,
                "alphabet": self.alphabet,
                "mapper": self.tokens_mapper["backward"],
                "weights": self.tokens_weights,
            }
        )

    def loads(self, dump: str):
        """
        Load a tokenizer model from a json-serialized string.
        """
        model = json.loads(dump)

        self.n_tokens = model["n_tokens"]

        self.alphabet = model["alphabet"]
        self.inverse_alphabet = {value: key for key, value in model["alphabet"].items()}
        self.alphabet_size = len(model["alphabet"])

        self.tokens_mapper = {
            "backward": {token: tuple(seq) for token, seq in model["mapper"].items()},
            "forward": {tuple(seq): token for token, seq in model["mapper"].items()},
        }
        self.tokens_weights = model["weights"]

        return self
