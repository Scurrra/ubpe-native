import json


class UBPEBase[T]:
    n_tokens: int
    alphabet_size: int
    alphabet: dict[T, int]
    inverse_alphabet: dict[int, T]
    tokens_mapper: dict[str, dict[int | tuple[int, ...], tuple[int, ...] | int]] = {
        "backward": dict(),
        "forward": dict(),
    }
    tokens_weights: dict[int, float] = dict()

    def __init__(
        self,
        alphabet_size: int | None = None,
        alphabet: dict[T, int] | None = None,
        n_tokens: int = 2**10,
    ):
        assert not (
            alphabet is None and alphabet_size is None
        ), "Either `alphabet_size` or `alphabet` must be specified, or model should be load from json string"

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
        Function for replacing pair of adjacent tokens in a list with a new one.

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
        assert len(self.tokens_weights) != 0, "Tokenizer is not fitted"

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

        # the old approach could produce out-of-bounds token ids
        # transformer = {buf[i][0]: self.alphabet_size + i for i in range(len(buf))}
        transformer = dict[int | tuple[int, ...], int]()
        offset = 0
        for i in range(len(buf) - len(to_delete)):
            while buf[i + offset][0] in to_delete:
                offset += 1
            transformer[buf[i + offset][0]] = self.alphabet_size + i

        self.tokens_weights = {
            mapper[1]: self.tokens_weights[mapper[0]]  # type: ignore (`mapper[0]`, i.e. key in `transformer`, or the old artificial token, is guaranteed to be of type `int`)
            for mapper in transformer.items()
        }

        # old approach sorted tokens before constructing a dict, but in the new one `transformer.items()` returns an already sorted by token weights list of mappings
        self.tokens_mapper = {  # type: ignore
            "backward": {
                new_token: tuple(
                    transformer.get(token, token)  # type: ignore (`token` is an element of `tuple[int, ...]`)
                    for token in self.tokens_mapper["backward"][old_token]  # type: ignore (the collection here is quaranteed to be of type `tuple[int, ...]`)
                )
                for old_token, new_token in transformer.items()
            }
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

    @classmethod
    def loads(cls, dump: str, token_type: type = int):
        """
        Load a tokenizer model from a json-serialized string.
        """
        model = json.loads(dump)

        inst = cls(n_tokens=model["n_tokens"], alphabet_size=len(model["alphabet"]))

        for key, value in model["alphabet"].items():
            key = token_type(key)
            inst.alphabet[key] = value
            inst.inverse_alphabet[value] = key

        for token, seq in model["mapper"].items():
            token = token_type(token)
            seq = tuple(token_type(_) for _ in seq)
            inst.tokens_mapper["backward"][token] = seq
            inst.tokens_mapper["forward"][seq] = token

        inst.tokens_weights = {
            token_type(token): float(weight)
            for token, weight in model["weights"].items()
        }

        return inst
