import os
from typing import BinaryIO, Dict, List, Tuple, Iterable, Iterator
import regex as re
from collections import defaultdict, Counter


class PreTokenizer:
    def __init__(self, special_tokens: List[str]) -> None:
        self.special_tokens = special_tokens
        self.special_tokens_patterns = self._patterns()
        self.word_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def _patterns(self) -> None:
        ```
        The goal is to balance meaningful chunks 
        (words, contractions) with atomic symbols (punctuation, spaces).
        ```
        if not self.special_tokens:
            self.special_tokens_patterns = r"(?!)"
            return None
        patterns = []
        for token in self.special_tokens:
            patterns.append(re.escape(token))
        self.special_tokens_patterns = "|".join(patterns)
        return None

    def chunk(self, file: BinaryIO, num_chunks: int, split_tokens: bytes):
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // num_chunks
        
        chunks = [i * chunk_size for i in range(num_chunks + 1)]
        chunks[-1] = file_size

        min_chunk_size = 4096

        for i in range(1, len(chunks) - 1):
            index = chunks[i]
            file.seek(index)
            while True:
                min_chunk = file.read(min_chunk_size)

                if min_chunk == b"":
                    chunks[i] = file_size
                    break

                found = min_chunk.find(split_tokens)
                if found != -1:
                    chunks[index] = index + found
                    break
                index += min_chunk_size

        return sorted(set(chunks))

    def read(self, file_path: str) -> Iterable[List[str]]:
        with open(file_path, "rb") as f:
            chunks = chunk(f, 100, "<|endoftext|>".encode("utf-8"))

            for start, end in zip(chunks[:-1], chunks[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", error="ignore")
                yield re.split(self.special_tokens_patterns, chunk)

    def build_word_frequency(self, sents: Iterable[str]) -> Dict:
        bword_dict = defaultdict()
        word_dict = defaultdict()

        for sent in sents:
            if not sent:
                continue
            matches = [word.group(0) for word in self.word_pattern.finditer(sent)]

            word_dict.update(matches)

        # String into byte frequencies
        for word, freq in word_dict.items():
            bword_dict[word.encode("utf-8")] = freq

        return bword_dict

    def pretonenize(self, sent: str) -> List[bytes]:
        splits = re.split(f"({self.special_tokens_patterns})", sent)

        output = []

        for split in splits:
            if split in self.special_tokens:
                output.append(split.encode("utf-8"))
            elif split:
                tokens = [
                    match.group(0).encode("utf-8")
                    for match in self.word_pattern.finditer(part)
                ]

                output.extend(tokens)

        return output

    def pretokenize_generator(self, sents: Iterable[str]) -> Iterable[bytes]:
        for sent in sents:
            splits = re.split(f"({self.special_tokens_patterns})", sent)
            for split in splits:
                if split in self.special_tokens:
                    yield split.encode("utf-8")
                elif split:
                    for match in self.word_pattern.finditer(split):
                        yeild match.group(0).encode("utf-8")
            