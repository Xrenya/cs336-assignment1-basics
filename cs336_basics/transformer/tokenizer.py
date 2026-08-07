from __future__ import annotations
import os

import json
import unicodedata
import heapq
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

Pair: TypeAlias = tuple[bytes, bytes]


# class PreTokenizer:
#     def __init__(
#         self,
#         special_tokens: Iterable[str] = ("<|endoftext|>",),
#     ) -> None:
#         self.word_pattern = re.compile(
#             r"'(?:[sdmt]|ll|ve|re)"
#             r"| ?\p{L}+"
#             r"| ?\p{N}+"
#             r"| ?[^\s\p{L}\p{N}]+"
#             r"|\s+(?!\S)"
#             r"|\s+"
#         )

#         tokens = set(special_tokens)
#         if "" in tokens:
#             raise ValueError("Special tokens cannot be empty")

#         self.special_tokens = frozenset(tokens)
#         ordered = sorted(tokens, key=lambda token: (-len(token), token))

#         self.special_token_pattern = (
#             re.compile(
#                 "("
#                 + "|".join(re.escape(token) for token in ordered)
#                 + ")"
#             )
#             if ordered
#             else None
#         )

#     def _split_special(self, text: str) -> Iterable[str]:
#         if self.special_token_pattern is None:
#             yield text
#         else:
#             yield from filter(None, self.special_token_pattern.split(text))

#     def pretokenize(self, sent: str) -> list[bytes]:
#         return list(self.pretokenize_iter((sent,)))

#     def pretokenize_iter(
#         self,
#         sents: Iterable[str],
#     ) -> Iterator[bytes]:
#         for sent in sents:
#             for part in self._split_special(sent):
#                 if part in self.special_tokens:
#                     yield part.encode("utf-8")
#                 else:
#                     for match in self.word_pattern.finditer(part):
#                         yield match.group(0).encode("utf-8")

#     def build_word_frequency(
#         self,
#         sents: Iterable[str],
#     ) -> Counter[bytes]:
#         frequencies: Counter[bytes] = Counter()

#         for sent in sents:
#             for part in self._split_special(sent):
#                 if part not in self.special_tokens:
#                     frequencies.update(
#                         match.group(0).encode("utf-8")
#                         for match in self.word_pattern.finditer(part)
#                     )

#         return frequencies
    

# class BPETokenizer:
#     def __init__(
#         self,
#         vocab_size: int,
#         special_tokens: Optional[List[str] | None] = None,
#     ) -> None:
#         self.vocab_size = vocab_size
#         self.special_tokens = special_tokens or []
#         self.special_tokens_bytes = [
#             token.encode("utf-8") for token in self.special_tokens
#         ]
#         self.merges: List[Tuple[bytes, bytes]] = []
#         self.stoi: Dict[bytes, int] = {}
#         self.itos: Dict[int, bytes] = {}
#         self.merges_rank: Dict[Tuple[bytes, bytes], int] = {}

#         self.pre_tokenizer = PreTokenizer()
        
#     def initialize(self):
#         for i, token_bytes in enumerate(self.special_tokens_bytes):
#             self.stoi[token_bytes] = i
#             self.itos[i] = token_bytes

#         offset = len(self.special_tokens_bytes)
#         for i in range(256):
#             self.stoi[bytes([i])] = i + offset
#             self.itos[offset + i] = bytes([i])

#         self.vocab = self.itos.copy()
#         # pair2new: (p1, p2) -> new_token_id
#         self.pair2new = {
#             (p1, p2): self.stoi[p1 + p2]
#             for (p1, p2) in self.merges
#         }

#     def read_file(self, file_path: str):
#         with open(file_path, "rb") as f:
#             text = f.read()
#         return text

#     def train(self, file_path: str):
#         text = self.read_file(file_path)

#         if self.special_tokens:
#             special_pattern = f"({'|'.join(re.escape(s) for s in self.special_tokens)})"
#             text_parts = re.split(special_pattern, text)
#         else:
#             text_parts = [text]

#         vocab = {string: int for int, string in self.itos.items()}
#         token_group = []
#         for part in text_parts:
#             if part in self.special_tokens or not part:
#                 continue
#             words2byte = self.pre_tokenizer.pretokenize(part)
#             for word in words2byte:
#                 token_group.append([vocab[bytes([b])] for b in word])

#         index = 0
#         pair_count = {}
#         token = {}
#         pre = {}
#         nxt = {}
#         pos = {}

#         for i, token_lst in enumerate(token_group):
#             if not token_lst or len(token_lst) <= 1:
#                 continue
#             token_lst_len = len(token_lst)
#             for j, token_id in enumerate(token_lst):
#                 index += 1
#                 token[index] = token_id


# def set_merges(self, merges: list[tuple[bytes, bytes]]):
#     self.merges = merges
#     # rank by order
#     self.merges_rank = {pair: i for i, pair in enumerate(merges)}

#     # ensure every merged token exists in vocab
#     for p1, p2 in merges:
#         new_tok = p1 + p2
#         if new_tok in self.special_tokens_bytes:
#             raise ValueError("A merge would create a special token bytes sequence.")
#         if new_tok not in self.stoi:
#             new_id = len(self.stoi)
#             self.stoi[new_tok] = new_id
#             self.itos[new_id] = new_tok

#     # fast mapping: (p1,p2) -> id(new_tok)
#     self.pair2new = {(p1, p2): self.stoi[p1 + p2] for (p1, p2) in merges}

Pair: TypeAlias = tuple[bytes, bytes]


class PreTokenizer:
    """GPT-2-style Unicode pre-tokenization with atomic special tokens."""

    _CONTRACTIONS = ("'s", "'d", "'m", "'t", "'ll", "'ve", "'re")
    _WHITE_SPACE_CODEPOINTS = frozenset(
        [
            *range(0x0009, 0x000E),
            0x0020,
            0x0085,
            0x00A0,
            0x1680,
            *range(0x2000, 0x200B),
            0x2028,
            0x2029,
            0x202F,
            0x205F,
            0x3000,
        ]
    )

    def __init__(self, special_tokens: Sequence[str] | None = None) -> None:
        if isinstance(special_tokens, str):
            raise TypeError("Pass special tokens as a sequence, not one string")
        tokens = tuple(special_tokens or ())

        if not all(isinstance(token, str) for token in tokens):
            raise TypeError("Special tokens must be strings")
        if any(token == "" for token in tokens):
            raise ValueError("Special tokens cannot be empty")
        if len(set(tokens)) != len(tokens):
            raise ValueError("Special tokens must be unique")

        # Encoding here also rejects strings containing unpaired surrogates.
        for token in tokens:
            token.encode("utf-8")

        self.special_tokens = tokens

        # Literal matching uses longest-first precedence at the same position.
        self._special_tokens_by_priority = tuple(
            sorted(tokens, key=lambda token: (-len(token), token))
        )

    @staticmethod
    def _is_letter(character: str) -> bool:
        return unicodedata.category(character).startswith("L")

    @staticmethod
    def _is_number(character: str) -> bool:
        return unicodedata.category(character).startswith("N")

    @classmethod
    def _is_whitespace(cls, character: str) -> bool:
        # Unicode White_Space, unlike str.isspace(), excludes U+001C..U+001F.
        return ord(character) in cls._WHITE_SPACE_CODEPOINTS

    @classmethod
    def _is_other(cls, character: str) -> bool:
        return (
            not cls._is_whitespace(character)
            and not cls._is_letter(character)
            and not cls._is_number(character)
        )

    @classmethod
    def _iter_ordinary_token_strings(cls, text: str) -> Iterator[str]:
        r"""
        Implement the GPT-2 pattern without a third-party regex dependency.

        This is equivalent to::

            '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|
            ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
        """
        index = 0
        text_length = len(text)

        while index < text_length:
            if text[index] == "'":
                contraction = next(
                    (
                        candidate
                        for candidate in cls._CONTRACTIONS
                        if text.startswith(candidate, index)
                    ),
                    None,
                )
                if contraction is not None:
                    yield contraction
                    index += len(contraction)
                    continue

            content_index = (
                index + 1
                if text[index] == " " and index + 1 < text_length
                else index
            )

            for predicate in (cls._is_letter, cls._is_number, cls._is_other):
                if not predicate(text[content_index]):
                    continue

                end = content_index + 1
                while end < text_length and predicate(text[end]):
                    end += 1

                yield text[index:end]
                index = end
                break
            else:
                # This branch is whitespace. To mirror \s+(?!\S), a run of
                # at least two whitespace characters before non-whitespace
                # leaves its final character for the next match.
                end = index + 1
                while end < text_length and cls._is_whitespace(text[end]):
                    end += 1

                if end < text_length and end - index >= 2:
                    end -= 1

                yield text[index:end]
                index = end

    def iter_segments(self, text: str) -> Iterator[tuple[bool, str]]:
        """Yield ``(is_special, segment)`` without losing any text."""
        if not isinstance(text, str):
            raise TypeError("PreTokenizer expects Unicode strings")

        if not self._special_tokens_by_priority:
            if text:
                yield False, text
            return

        cursor = 0
        while cursor < len(text):
            best_start: int | None = None
            best_token: str | None = None

            for token in self._special_tokens_by_priority:
                start = text.find(token, cursor)
                if start < 0:
                    continue
                if best_start is None or start < best_start:
                    best_start = start
                    best_token = token

            if best_start is None or best_token is None:
                yield False, text[cursor:]
                return

            if cursor < best_start:
                yield False, text[cursor:best_start]

            yield True, best_token
            cursor = best_start + len(best_token)

    def iter_tokens(self, text: str) -> Iterator[tuple[bool, bytes]]:
        """Yield special tokens or ordinary pre-tokens as UTF-8 bytes."""
        for is_special, segment in self.iter_segments(text):
            if is_special:
                yield True, segment.encode("utf-8")
                continue

            for token in self.iter_ordinary_tokens(segment):
                yield False, token

    def iter_ordinary_tokens(self, text: str) -> Iterator[bytes]:
        """Pre-tokenize text without recognizing configured special tokens."""
        if not isinstance(text, str):
            raise TypeError("PreTokenizer expects Unicode strings")
        for token in self._iter_ordinary_token_strings(text):
            yield token.encode("utf-8")

    def pretokenize(self, text: str) -> list[bytes]:
        """Pre-tokenize ordinary text; configured specials remain atomic."""
        return [token for _, token in self.iter_tokens(text)]

    def pretokenize_iter(
        self,
        texts: str | Iterable[str],
    ) -> Iterator[bytes]:
        if isinstance(texts, str):
            texts = (texts,)
        for text in texts:
            for _, token in self.iter_tokens(text):
                yield token

    def frequencies(
        self,
        texts: str | Iterable[str],
    ) -> Counter[bytes]:
        """Count ordinary pre-tokens, excluding configured special tokens."""
        if isinstance(texts, str):
            texts = (texts,)

        frequencies: Counter[bytes] = Counter()
        for text in texts:
            for is_special, token in self.iter_tokens(text):
                if not is_special:
                    frequencies[token] += 1

        return frequencies

    def build_word_frequency(
        self,
        texts: Iterable[str],
    ) -> Counter[bytes]:
        """Compatibility wrapper around :meth:`frequencies`."""
        return self.frequencies(texts)

    def read(
        self,
        file_path: str | Path,
        *,
        encoding: str = "utf-8",
    ) -> Iterator[list[str]]:
        """
        Read a corpus and yield one batch of ordinary segments.

        Reading the complete text is intentional: arbitrary byte chunking can
        split a UTF-8 character, a special token, or an ordinary pre-token and
        thereby change the training frequencies.
        """
        text = Path(file_path).read_bytes().decode(encoding)
        ordinary_segments = [
            segment
            for is_special, segment in self.iter_segments(text)
            if not is_special and segment
        ]
        if ordinary_segments:
            yield ordinary_segments


def merge_pair(
    pieces: Sequence[bytes],
    pair: Pair,
) -> list[bytes]:
    """Merge all non-overlapping occurrences of ``pair``, left to right."""
    merged_token = pair[0] + pair[1]
    result: list[bytes] = []
    index = 0

    while index < len(pieces):
        if (
            index + 1 < len(pieces)
            and pieces[index] == pair[0]
            and pieces[index + 1] == pair[1]
        ):
            result.append(merged_token)
            index += 2
        else:
            result.append(pieces[index])
            index += 1

    return result


class BPETokenizer:
    """
    Byte-level BPE tokenizer.

    ID layout:
      * 0..255: raw bytes
      * 256..256+M-1: learned merge tokens, in rank order
      * 256+M..256+M+S-1: special tokens, in caller order
    """

    FORMAT = "byte-bpe-v2"

    def __init__(
        self,
        *,
        merges: Sequence[Pair] = (),
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        self.pre_tokenizer = PreTokenizer(special_tokens)
        self.special_tokens = self.pre_tokenizer.special_tokens

        self.itos: dict[int, bytes] = {
            byte: bytes([byte]) for byte in range(256)
        }
        self.regular_to_id: dict[bytes, int] = {
            token: token_id for token_id, token in self.itos.items()
        }

        self.merges: tuple[Pair, ...]
        normalized_merges: list[Pair] = []
        self.merges_rank: dict[Pair, int] = {}
        self.pair2new: dict[Pair, int] = {}

        for rank, raw_pair in enumerate(merges):
            if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
                raise ValueError("Every merge must contain exactly two tokens")

            left, right = raw_pair
            if not isinstance(left, bytes) or not isinstance(right, bytes):
                raise TypeError("Merge components must be bytes")
            if not left or not right:
                raise ValueError("Merge components cannot be empty")

            pair = (left, right)
            if pair in self.merges_rank:
                raise ValueError(f"Duplicate merge pair: {pair!r}")
            if left not in self.regular_to_id:
                raise ValueError(f"Unknown left merge component: {left!r}")
            if right not in self.regular_to_id:
                raise ValueError(f"Unknown right merge component: {right!r}")

            new_token = left + right
            if new_token in self.regular_to_id:
                raise ValueError(
                    f"Merge creates an existing ordinary token: {new_token!r}"
                )

            token_id = 256 + rank
            normalized_merges.append(pair)
            self.merges_rank[pair] = rank
            self.pair2new[pair] = token_id
            self.regular_to_id[new_token] = token_id
            self.itos[token_id] = new_token

        self.merges = tuple(normalized_merges)

        # Specials deliberately use a separate reverse map. A special token
        # may have the same bytes as an ordinary token, e.g. special token "a".
        self.special_to_id: dict[str, int] = {}
        self._special_bytes_to_id: dict[bytes, int] = {}
        special_offset = 256 + len(self.merges)

        for index, token in enumerate(self.special_tokens):
            token_id = special_offset + index
            token_bytes = token.encode("utf-8")
            self.itos[token_id] = token_bytes
            self.special_to_id[token] = token_id
            self._special_bytes_to_id[token_bytes] = token_id

        self.vocab = self.itos
        self._encode_cache: dict[bytes, tuple[int, ...]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def _encode_pretoken(self, token: bytes) -> tuple[int, ...]:
        cached = self._encode_cache.get(token)
        if cached is not None:
            return cached

        pieces = [bytes([byte]) for byte in token]

        # Replaying the learned rules in rank order is equivalent to repeatedly
        # choosing the currently available pair with the lowest learned rank.
        for pair in self.merges:
            pieces = merge_pair(pieces, pair)

        encoded = tuple(self.regular_to_id[piece] for piece in pieces)
        self._encode_cache[token] = encoded
        return encoded

    def encode(
        self,
        text: str,
        *,
        recognize_special_tokens: bool = True,
    ) -> list[int]:
        token_ids: list[int] = []

        if recognize_special_tokens:
            tokens = self.pre_tokenizer.iter_tokens(text)
        else:
            tokens = (
                (False, token)
                for token in self.pre_tokenizer.iter_ordinary_tokens(text)
            )

        for is_special, token in tokens:
            if is_special:
                token_ids.append(self._special_bytes_to_id[token])
            else:
                token_ids.extend(self._encode_pretoken(token))

        return token_ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode without interpreting any configured special-token spelling."""
        return self.encode(text, recognize_special_tokens=False)

    def decode_bytes(self, token_ids: Iterable[int]) -> bytes:
        result = bytearray()

        for token_id in token_ids:
            if type(token_id) is not int:
                raise TypeError("Token IDs must be integers")
            try:
                result.extend(self.itos[token_id])
            except KeyError:
                raise ValueError(f"Unknown token ID: {token_id}") from None

        return bytes(result)

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        errors: str = "strict",
    ) -> str:
        # Decode after concatenating all bytes; individual byte tokens may not
        # be valid UTF-8 on their own.
        return self.decode_bytes(token_ids).decode("utf-8", errors=errors)

    def save(self, file_path: str | Path) -> None:
        payload = {
            "format": self.FORMAT,
            "unicode_version": unicodedata.unidata_version,
            "special_tokens": list(self.special_tokens),
            "merges": [
                [left.hex(), right.hex()] for left, right in self.merges
            ],
        }
        Path(file_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, file_path: str | Path) -> BPETokenizer:
        try:
            payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("Could not read tokenizer model") from error

        if not isinstance(payload, dict) or payload.get("format") != cls.FORMAT:
            raise ValueError("Unsupported tokenizer model format")
        if payload.get("unicode_version") != unicodedata.unidata_version:
            raise ValueError(
                "Tokenizer model uses a different Unicode database version"
            )

        special_tokens = payload.get("special_tokens")
        raw_merges = payload.get("merges")
        if (
            not isinstance(special_tokens, list)
            or not all(isinstance(token, str) for token in special_tokens)
            or not isinstance(raw_merges, list)
        ):
            raise ValueError("Invalid tokenizer model")

        try:
            merges = [
                (bytes.fromhex(pair[0]), bytes.fromhex(pair[1]))
                for pair in raw_merges
                if isinstance(pair, list) and len(pair) == 2
            ]
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid tokenizer model") from error

        if len(merges) != len(raw_merges):
            raise ValueError("Invalid tokenizer model")

        try:
            return cls(merges=merges, special_tokens=special_tokens)
        except (TypeError, ValueError, UnicodeError) as error:
            raise ValueError("Invalid tokenizer model") from error


class Trainer:
    """Byte-level BPE trainer with exact incremental pair statistics."""

    def __init__(
        self,
        vocab_size: int,
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        if type(vocab_size) is not int:
            raise TypeError("vocab_size must be an integer")

        self.preprocessor = PreTokenizer(special_tokens)
        self.special_tokens = self.preprocessor.special_tokens

        minimum_size = 256 + len(self.special_tokens)
        if vocab_size < minimum_size:
            raise ValueError(f"vocab_size must be at least {minimum_size}")

        self.vocab_size = vocab_size

    def train(self, input_path: str | Path) -> BPETokenizer:
        """Train from a UTF-8 text file."""
        # Byte reading avoids universal-newline conversion of CRLF and CR.
        text = Path(input_path).read_bytes().decode("utf-8")
        return self.train_text(text)

    def train_text(self, text: str) -> BPETokenizer:
        return self.train_frequencies(self.preprocessor.frequencies(text))

    def train_texts(self, texts: Iterable[str]) -> BPETokenizer:
        return self.train_frequencies(self.preprocessor.frequencies(texts))

    def train_frequencies(
        self,
        frequencies: Mapping[bytes, int],
    ) -> BPETokenizer:
        """
        Train from ordinary pre-token frequencies.

        Adjacent occurrences are counted with overlap, while each selected
        merge is applied non-overlapping from left to right. Incremental pair
        updates are computed by comparing each affected word before and after
        the merge, which correctly handles weighted and overlapping pairs.
        """
        normalized: Counter[bytes] = Counter()

        for word, frequency in frequencies.items():
            if not isinstance(word, bytes):
                raise TypeError("Frequency keys must be bytes")
            if type(frequency) is not int:
                raise TypeError("Frequencies must be integers")
            if frequency < 0:
                raise ValueError("Frequencies cannot be negative")
            if word and frequency:
                normalized[word] += frequency

        splits = {
            word: [bytes([byte]) for byte in word]
            for word in normalized
        }
        known_tokens = {bytes([byte]) for byte in range(256)}

        pair_frequencies: Counter[Pair] = Counter()
        pair_to_words: defaultdict[Pair, set[bytes]] = defaultdict(set)

        for word, pieces in splits.items():
            local_counts = Counter(zip(pieces, pieces[1:]))
            for pair, count in local_counts.items():
                pair_frequencies[pair] += count * normalized[word]
                pair_to_words[pair].add(word)

        # Entries are lazy: changed frequencies add a fresh snapshot, while
        # stale snapshots are discarded when popped.
        frequency_heap: list[tuple[int, Pair]] = [
            (-frequency, pair)
            for pair, frequency in pair_frequencies.items()
        ]
        heapq.heapify(frequency_heap)

        def get_best_pair() -> Pair:
            while frequency_heap:
                negative_frequency, pair = heapq.heappop(frequency_heap)
                frequency = -negative_frequency
                if pair_frequencies.get(pair) != frequency:
                    continue

                # Gather every valid pair tied at this frequency, then apply
                # the documented lexicographically-greatest tie break.
                tied_pairs = {pair}
                while (
                    frequency_heap
                    and frequency_heap[0][0] == negative_frequency
                ):
                    _, tied_pair = heapq.heappop(frequency_heap)
                    if pair_frequencies.get(tied_pair) == frequency:
                        tied_pairs.add(tied_pair)

                best_pair = max(tied_pairs)
                for tied_pair in tied_pairs:
                    if tied_pair != best_pair:
                        heapq.heappush(
                            frequency_heap,
                            (-frequency, tied_pair),
                        )
                return best_pair

            raise RuntimeError("Pair heap is inconsistent with pair counts")

        def apply_merge(best_pair: Pair) -> None:
            affected_words = tuple(pair_to_words.get(best_pair, ()))
            if not affected_words:
                raise RuntimeError("Selected pair has no affected words")

            global_deltas: Counter[Pair] = Counter()

            for word in affected_words:
                old_pieces = splits[word]
                old_counts = Counter(zip(old_pieces, old_pieces[1:]))
                if old_counts[best_pair] == 0:
                    raise RuntimeError("pair_to_words contains a stale word")

                new_pieces = merge_pair(old_pieces, best_pair)
                new_counts = Counter(zip(new_pieces, new_pieces[1:]))
                splits[word] = new_pieces

                for pair in old_counts.keys() | new_counts.keys():
                    occurrence_delta = new_counts[pair] - old_counts[pair]
                    if occurrence_delta:
                        global_deltas[pair] += (
                            occurrence_delta * normalized[word]
                        )

                    if new_counts[pair]:
                        pair_to_words[pair].add(word)
                    else:
                        words = pair_to_words.get(pair)
                        if words is not None:
                            words.discard(word)
                            if not words:
                                pair_to_words.pop(pair, None)

            for pair, delta in global_deltas.items():
                new_frequency = pair_frequencies.get(pair, 0) + delta
                if new_frequency < 0:
                    raise RuntimeError("Pair frequency became negative")

                if new_frequency:
                    pair_frequencies[pair] = new_frequency
                    heapq.heappush(
                        frequency_heap,
                        (-new_frequency, pair),
                    )
                else:
                    pair_frequencies.pop(pair, None)
                    if pair_to_words.get(pair):
                        raise RuntimeError(
                            "Zero-frequency pair still has affected words"
                        )
                    pair_to_words.pop(pair, None)

            if best_pair in pair_frequencies:
                raise RuntimeError("Selected pair survived its global merge")

        merge_budget = self.vocab_size - 256 - len(self.special_tokens)
        merges: list[Pair] = []

        for _ in range(merge_budget):
            if not pair_frequencies:
                break

            best_pair = get_best_pair()
            new_token = best_pair[0] + best_pair[1]
            if new_token in known_tokens:
                raise RuntimeError(
                    "A merge produced an existing ordinary token; "
                    "use an ID-based merge representation for this corpus"
                )

            merges.append(best_pair)
            known_tokens.add(new_token)
            apply_merge(best_pair)

        return BPETokenizer(
            merges=merges,
            special_tokens=self.special_tokens,
        )



    
