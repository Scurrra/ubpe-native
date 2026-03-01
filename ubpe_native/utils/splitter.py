import re
from enum import Flag, auto

from ssstree import SSSTree


class SplitMode(Flag):
    """SplitMode enum

    Options:
        NONE: No splitting.
        KNOWN_WORDS: Split on known words.
        BREAK_TOKENS: Split on break tokens.
        REGEX: Split on regex.
        STOP_TOKENS: Split on stop tokens.
        FULL: Split on all options.
    """

    NONE = auto()
    KNOWN_WORDS = auto()
    BREAK_TOKENS = auto()
    REGEX = auto()
    STOP_TOKENS = auto()
    FULL = KNOWN_WORDS | BREAK_TOKENS | REGEX | STOP_TOKENS


class SplitPipeline[T]:
    """SplitPipeline class"""

    alphabet: dict
    known_words: dict[T, int] | None = None
    break_tokens: set | None = None
    stop_tokens: set | None = None
    regex_str: str | None = None

    _regex = None
    kw_ssstree: SSSTree[T, int] | None = None  # type: ignore

    def __init__(
        self,
        alphabet: list | set | str | dict,
        *,
        known_words: list | dict | None = None,
        break_tokens: set | list | None = None,
        stop_tokens: set | list | None = None,
        regex_str: str | None = None,
    ):
        """Initialize SplitPipeline class

        Args:
            alphabet (list | set | str | dict): The alphabet to use for splitting.
            known_words (list | dict, optional): Known words to use for splitting. Defaults to None.
            break_tokens (set | list, optional): Break tokens to use for splitting. Defaults to None.
            stop_tokens (set | list, optional): Stop tokens to use for splitting. Defaults to None.
            regex_str (str, optional): Regex string to use for splitting. Defaults to None.
        """

        if not alphabet:
            raise Exception("`alphabet` must be provided")
        if (
            isinstance(alphabet, list)
            or isinstance(alphabet, set)
            or isinstance(alphabet, str)
        ):
            self.alphabet = {token: i for i, token in enumerate(alphabet)}
        elif isinstance(alphabet, dict):
            self.alphabet = alphabet
        else:
            raise Exception(
                "`alphabet` must be either `list` | `set` | `str` | `dict`"
            )

        if known_words:
            if isinstance(known_words, list):
                self.known_words = {
                    word: i
                    for i, word in enumerate(
                        known_words, start=len(self.alphabet)
                    )
                }
                self.kw_ssstree = SSSTree[T, int]()  # type: ignore
                for kw in self.known_words.items():
                    _ = self.kw_ssstree + kw
            elif isinstance(known_words, dict):
                kw_tokens = sorted(known_words.values())
                if len(alphabet) >= kw_tokens[0]:
                    raise Exception(
                        "The minimal token of known words shouldn't be greater than the number of tokens in the alphabet"
                    )
                self.known_words = known_words
                self.kw_ssstree = SSSTree[T, int]()  # type: ignore
                for kw in self.known_words.items():
                    _ = self.kw_ssstree + kw

        if break_tokens and (
            isinstance(break_tokens, set)
            or isinstance(break_tokens, list)
            or isinstance(break_tokens, str)
            or isinstance(break_tokens, tuple)
        ):
            self.break_tokens = set(
                token for token in break_tokens if token in self.alphabet
            )

        if stop_tokens and (
            isinstance(stop_tokens, set)
            or isinstance(stop_tokens, list)
            or isinstance(stop_tokens, str)
            or isinstance(stop_tokens, tuple)
        ):
            self.stop_tokens = set(
                token for token in stop_tokens if token in self.alphabet
            )

        if regex_str and isinstance(regex_str, str):
            self.regex_str = regex_str
            self._regex = re.compile(regex_str)

    def __call__(
        self,
        doc: T,
        *,
        mode: SplitMode = SplitMode.FULL,
        leave_separators: bool = True,
    ):
        """Split a document into parts based on known words and stop tokens.

        Args:
            doc: The document to split.
            mode: The splitting mode.
            leave_separators: Whether to leave separators in the parts.

        Returns:
            A list of parts.
        """

        parts = []
        if SplitMode.KNOWN_WORDS in mode and self.kw_ssstree:
            part_start = 0
            si = 0
            while si < len(doc):  # type: ignore
                kw_candidates: list[tuple[int, int]] = self.kw_ssstree(
                    doc, start=si, fast=True
                )  # type: ignore
                if len(kw_candidates) == 0:
                    si += 1
                    continue
                if si != part_start:
                    parts.extend(
                        [self.alphabet[token] for token in part]
                        for part in self._split_part(
                            doc[part_start:si],  # type: ignore
                            mode=mode,
                            leave_separators=leave_separators,
                        )
                    )
                if leave_separators:
                    parts.append([kw_candidates[-1][1]])
                part_start = si + kw_candidates[-1][0]
                si = part_start
            if part_start < len(doc):  # type: ignore
                parts.extend(
                    [self.alphabet[token] for token in part]
                    for part in self._split_part(
                        doc[part_start:],  # type: ignore
                        mode=mode,
                        leave_separators=leave_separators,
                    )
                )
        else:
            parts = [doc]
        return parts

    def _split_part(
        self,
        part,
        *,
        mode: SplitMode = SplitMode.FULL,
        leave_separators: bool = True,
    ):
        parts = [part]
        if SplitMode.BREAK_TOKENS in mode:
            splits = []
            for part in parts:
                splits.extend(
                    self._split_part_by_break_tokens(part, leave_separators)
                )
            parts = splits
        if SplitMode.REGEX in mode:
            splits = []
            for part in parts:
                splits.extend(self._split_part_by_regex(part))
            parts = splits
        if SplitMode.STOP_TOKENS in mode:
            splits = []
            for part in parts:
                splits.extend(
                    self._split_part_by_stop_tokens(part, leave_separators)
                )
            parts = splits
        return parts

    def _split_part_by_break_tokens(
        self, part: T, leave_separators: bool = True
    ) -> list[T]:
        return (
            self._split_part_by_tokens(
                part, self.break_tokens, leave_separators
            )
            if self.break_tokens
            else [part]
        )

    def _split_part_by_regex(self, part: str) -> list[str]:
        if not isinstance(part, str):
            return [part]
        if not self._regex:
            return [part]
        return self._regex.findall(part)

    def _split_part_by_stop_tokens(
        self, part: T, leave_separators: bool = True
    ) -> list[T]:
        return (
            self._split_part_by_tokens(part, self.stop_tokens, leave_separators)
            if self.stop_tokens
            else [part]
        )

    @staticmethod
    def _split_part_by_tokens(
        part: T, tokens: set, leave_separators: bool = True
    ) -> list[T]:
        parts = []
        part_start = 0

        for ti, token in enumerate(part):  # type: ignore
            if token not in tokens:
                continue
            if ti != part_start:
                parts.append(part[part_start:ti])  # type: ignore
            if leave_separators:
                parts.append(part[ti : ti + 1])  # type: ignore
            part_start = ti + 1

        if part_start < len(part):  # type: ignore
            parts.append(part[part_start:])  # type: ignore
        return parts
