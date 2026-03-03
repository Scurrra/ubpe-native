import json

from .utils import SplitPipeline


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

    known_words: dict[T, int] | dict[tuple[T, ...], int] | None = None
    inverse_known_words: dict[int, str] | dict[int, tuple[T, ...]] | None = None
    break_tokens: set[T] | None = None
    regex_str: str | None = None
    stop_tokens: set[T] | None = None
    split_pipeline: SplitPipeline[T]  # type: ignore

    def __init__(
        self,
        *,
        alphabet_size: int | None = None,
        alphabet: dict[T, int] | list[T] | set[T] | None = None,
        n_tokens: int = 2**10,
        known_words: list | dict | None = None,
        break_tokens: set | list | None = None,
        regex_str: str | None = None,
        stop_tokens: set | list | None = None,
    ):
        if alphabet is None and alphabet_size is None:
            raise TypeError(
                "Either `alphabet_size` or `alphabet` must be specified, or model should be load from json string"
            )

        if known_words is not None and not isinstance(known_words, (list, dict)):
            raise TypeError("If `known_words` is provided, it must be a list or dict")

        # if `alphabet_size` is provided and `alphabet` is not, `T` is assumed to be `int`
        if alphabet is None:
            if known_words is None:
                alphabet = {i: i for i in range(alphabet_size)}  # type: ignore
            else:
                alphabet = {i: i for i in range(alphabet_size - len(known_words))}  # type: ignore

        # ensure that `alphabet` is a dict
        else:
            key_types = set(type(key) for key in alphabet)
            if len(key_types) > 1:
                raise TypeError(
                    "If `alphabet` is a `dict` its keys must must have the same type"
                )

            if isinstance(alphabet, (list, set)):
                alphabet = {token: i for i, token in enumerate(alphabet)}  # type: ignore

            if not isinstance(alphabet, dict):
                raise TypeError(
                    "If `alphabet` is provided, it must be a dict, a list, or a set"
                )

            if sorted(alphabet.values()) != list(range(len(alphabet))):
                raise ValueError(
                    "Values in `alphabet` should form a sequence `0..len(alphabet)`"
                )

        if known_words is not None:
            if isinstance(known_words, list):
                if len(known_words) == 0:
                    known_words = None
                else:
                    if isinstance(known_words[0], str):
                        known_words = {
                            word: i
                            for i, word in enumerate(known_words, start=len(alphabet))  # type: ignore
                        }
                    elif isinstance(known_words[0], list):
                        known_words = {
                            tuple(word): i
                            for i, word in enumerate(known_words, start=len(alphabet))  # type: ignore
                        }
                    elif isinstance(known_words[0], tuple):
                        known_words = {
                            word: i
                            for i, word in enumerate(known_words, start=len(alphabet))  # type: ignore
                        }
                    else:
                        raise TypeError(
                            "If `known_words` is provided, it must be a list of strings or lists/tuples"
                        )
            else:
                key_types = set(type(key) for key in known_words)
                if len(key_types) > 1:
                    raise TypeError(
                        "If `known_words` is a `dict` its keys must must have the same type"
                    )
                key_type = key_types.pop()
                if key_type not in (str, tuple):
                    raise TypeError(
                        "If `known_words` is a `dict` its keys must be strings or tuples"
                    )

                if max(alphabet.values()) != min(known_words.values()) - 1:  # type: ignore
                    raise ValueError(
                        "Minimal token value in `known_words` must be one more than the maximal token value in `alphabet`"
                    )
                if sorted(known_words.values()) != list(
                    range(len(alphabet), len(alphabet) + len(known_words))  # type: ignore
                ):
                    raise ValueError(
                        "Values in `known_words` should form a sequence `len(alphabet)..len(alphabet)+len(known_words)`"
                    )

        if alphabet_size is None:
            alphabet_size = len(alphabet)  # type: ignore (`alphabet` could not be `None` till here)
            if known_words is not None:
                alphabet_size += len(known_words)
        elif isinstance(alphabet_size, int):
            if alphabet_size != len(alphabet) + (  # type: ignore
                0 if known_words is None else len(known_words)
            ):
                raise ValueError(
                    "If `alphabet_size` is provided, it must be equal to the length of `alphabet` plus the length of `known_words`"
                )
        else:
            raise TypeError("If `alphabet_size` is provided, it must be an integer")

        self.alphabet_size = alphabet_size
        self.alphabet = alphabet  # type: ignore (`alphabet` could not be `None` till here)
        self.inverse_alphabet = {value: key for key, value in self.alphabet.items()}
        self.n_tokens = n_tokens

        self.known_words = known_words
        self.inverse_known_words = (
            {value: key for key, value in self.known_words.items()}  # type: ignore
            if known_words is not None
            else None
        )

        if break_tokens is not None and (
            isinstance(break_tokens, set)
            or isinstance(break_tokens, list)
            or isinstance(break_tokens, str)
            or isinstance(break_tokens, tuple)
        ):
            self.break_tokens = set(
                token for token in break_tokens if token in self.alphabet
            )
            if len(self.break_tokens) == 0:
                self.break_tokens = None

        if regex_str is not None and isinstance(regex_str, str) and len(regex_str) > 0:
            self.regex_str = regex_str

        if stop_tokens is not None and (
            isinstance(stop_tokens, set)
            or isinstance(stop_tokens, list)
            or isinstance(stop_tokens, str)
            or isinstance(stop_tokens, tuple)
        ):
            self.stop_tokens = set(
                token for token in stop_tokens if token in self.alphabet
            )
            if len(self.stop_tokens) == 0:
                self.stop_tokens = None

        self.split_pipeline = SplitPipeline(
            alphabet=self.alphabet,  # type: ignore
            known_words=self.known_words,  # type: ignore
            break_tokens=self.break_tokens,  # type: ignore
            stop_tokens=self.stop_tokens,  # type: ignore
            regex_str=self.regex_str,
        )

    def _replace_token_pairs(
        self,
        l: list[int] | list[list[int]],  # noqa: E741
        sub: dict[int, tuple[int, list[int]]],
    ) -> list[int] | list[list[int]]:
        """
        Function for replacing pair of adjacent tokens in a list with a new one.

        Args:
        - `l (list)`: A list to be transformed.
        - `sub (dict[int, tuple[int, list[int]]])`: A substitution map, where keys
        are first tokens in the pairs, and the values are pair of the second token
        and the new one wrapped in a list.
        """
        if isinstance(l, list) and len(l) != 0:
            if isinstance(l[0], list):
                return [self._replace_token_pairs(sub=sub, l=part) for part in l]  # type: ignore
            elif isinstance(l[0], int):
                is_not_start = {key: False for key in list(sub.keys())}
                i = -1
                while i < len(l) - 2:
                    i += 1
                    if is_not_start.get(l[i], True):  # type: ignore
                        continue
                    start: int = l[i]  # type: ignore
                    if l[i + 1] == sub[start][0]:
                        l[i : i + 2] = sub[start][1]  # type: ignore
                return l
            else:
                raise TypeError("Invalid type of list elements")
        raise ValueError("Invalid list arguments")

    def _rearrange_tokens_by_weight(self):
        """
        Function that rearranges found tokens according to their weights and trims
        dictionary of the tokenizer to be not greater than `self.n_tokens`.
        """
        if len(self.tokens_weights) == 0:
            raise ValueError("Tokenizer is not fitted")

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
        inst = {
            "n_tokens": self.n_tokens,
            "alphabet": self.alphabet,
            "known_words": self.inverse_known_words,
            "break_tokens": self.break_tokens,
            "regex_str": self.regex_str,
            "stop_tokens": self.stop_tokens,
            "mapper": self.tokens_mapper["backward"],
            "weights": self.tokens_weights,
        }
        return json.dumps(
            {field: value for field, value in inst.items() if value is not None}
        )

    @classmethod
    def loads(cls, dump: str, token_type: type = int):
        """
        Load a tokenizer model from a json-serialized string.

        None: `.dumps()` guarantees that the `None` values are not included in the serialized model.
        """
        model = json.loads(dump)

        alphabet = dict()
        inverse_alphabet = dict()
        for key, value in model["alphabet"].items():
            key = token_type(key)
            value = int(value)
            alphabet[key] = value
            inverse_alphabet[value] = key

        inst = cls(
            n_tokens=int(model["n_tokens"]),
            alphabet_size=len(model["alphabet"]),
            alphabet=alphabet,
        )
        inst.inverse_alphabet = inverse_alphabet

        if "known_words" in model:
            inst.known_words = dict()
            inst.inverse_known_words = dict()
            for value, key in model["known_words"].items():
                value = int(value)
                key = key if token_type is str else tuple(key)
                inst.known_words[key] = value  # type: ignore
                inst.inverse_known_words[value] = key  # type: ignore

        if "break_tokens" in model:
            inst.break_tokens = set(
                token_type(token)
                for token in model["break_tokens"]  # type: ignore
            )

        if "regex_str" in model:
            inst.regex_str = model["regex_str"]

        if "stop_tokens" in model:
            inst.stop_tokens = set(token_type(token) for token in model["stop_tokens"])  # type: ignore

        inst.split_pipeline = SplitPipeline(
            alphabet=inst.alphabet,  # type: ignore
            known_words=inst.known_words,  # type: ignore
            break_tokens=inst.break_tokens,  # type: ignore
            stop_tokens=inst.stop_tokens,  # type: ignore
            regex_str=inst.regex_str,
        )

        for token, seq in model["mapper"].items():
            token = int(token)
            seq = tuple(int(_) for _ in seq)
            inst.tokens_mapper["backward"][token] = seq
            inst.tokens_mapper["forward"][seq] = token

        inst.tokens_weights = {
            int(token): float(weight) for token, weight in model["weights"].items()
        }

        return inst
