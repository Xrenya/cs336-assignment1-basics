import os
from typing import BinaryIO, Dict, List, Tuple, Iterable, Optional
import regex as re
from collections import defaultdict, Counter


class PreTokenizer:
    def __init__(self, special_tokens: List[str]) -> None:
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.special_tokens_patterns = "|".join(re.escape(token) for token in self.special_tokens) if self.special_tokens else r"(?!)"
        self.word_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

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
            chunks = self.chunk(f, 100, "<|endoftext|>".encode("utf-8"))

            for start, end in zip(chunks[:-1], chunks[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
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
        splits = re.split(f'({self.special_tokens_patterns})', sent)
        output = []

        for split in splits:
            if split in self.special_tokens:
                output.append(split.encode("utf-8"))
            elif split:
                tokens = [
                    match.group(0).encode("utf-8")
                    for match in self.word_pattern.finditer(split)
                ]

                output.extend(tokens)
        
        return output

    def pretokenize_iter(self, sents: Iterable[str]) -> Iterable[bytes]:
        for sent in sents:
            splits = re.split(f"({self.special_tokens_patterns})", sent)
            for split in splits:
                if split in self.special_tokens:
                    yield split.encode("utf-8")
                elif split:
                    for match in self.word_pattern.finditer(split):
                        yield match.group(0).encode("utf-8")

class BPE:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str] | None] = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.token2id = {token: index for index, token in vocab.items()}
        self.pretokenizer = PreTokenizer(self.special_tokens)
        self.word2id = defaultdict()
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str] | None] = None
    ) -> 'BPE':
        vocab = {}
        with open(vocab_filepath, "rb") as f:
            vocab_size_bytes = f.read(4)
            vocab_size = int.from_bytes(vocab_size_bytes, byteorder="little")
            for _ in range(vocab_size):
                btoken = f.read(4)
                token_id = int.from_bytes(btoken, byteorder="little")
                
                btoken_len = f.read(4)
                token_len = int.from_bytes(btoken_len, byteorder="little")
                token = f.read(token_len)
                vocab[token_id] = token

        merges = []
        with open(merges_filepath, "rb") as f:
            merges_bytes = f.read(4)
            merges_count = int.from_bytes(merges_bytes, byteorder="little")
            for _ in range(merges_count):
                len_bytes_1 = f.read(4)
                len_byte_1 = int.from_bytes(len_bytes_1, byteorder="little")
                byte_1 = f.read(len_byte_1)

                len_bytes_2 = f.read(4)
                len_byte_2 = int.from_bytes(len_bytes_2, byteorder="little")
                byte_2 = f.read(len_byte_2)

                merges.append((byte_1, byte_2))

        return cls(vocab, merges, special_tokens)
    
    def calculate_id(self, word: bytes) -> List[int]:
        token_ids = []
        bytes_list = [bytes([b]) for b in word]

        while len(bytes_list) > 1:
            min_id = None
            min_merge_pos = None

            for i, pair in enumerate(zip(bytes_list[:-1], bytes_list[1:])):
                idx = self.token2id.get(pair[0] + pair[1])
                if idx is not None and (min_id is None or idx < min_id):
                    min_id = idx
                    min_merge_pos = i

            if min_id is None:
                break
                
            bytes_list[min_merge_pos:min_merge_pos + 2] = [
                bytes_list[min_merge_pos] + bytes_list[min_merge_pos + 1]
            ]

        for part in bytes_list:
            try:
                id = self.token2id[part]
                token_ids.append(id)
            except KeyError:
                print(f"Not found '{part}'")
                pass
        return token_ids


    def encode(self, sent: str) -> List[int]:
        words = self.pretokenizer.pretonenize(sent)
        ids = []
        for word in words:
            if word in self.token2id:
                ids.append(self.token2id[word])
            elif word in self.word2id:
                ids.extend(self.word2id[word])
            else:
                token_id = self.calculate_id(word)
                self.word2id[word] = token_id
                ids.extend(token_id)
            
        return ids

    def encode_iterable(self, Iterable: Iterable[str]) -> Iterable[int]:
        word_iter = self.pretokenizer.pretokenize_iter(Iterable)
        for word in word_iter:
            if word in self.token2id:
                yield self.token2id[word]
            elif word in self.word2id:
                yield from self.word2id[word]
            else:
                token_id = self.calculate_id(word)
                self.word2id[word] = token_id
                yield from token_id

    def decode(
        self,
        ids: Iterable[int],
        end_token_id: Optional[int | None] = None
    ) -> str:
        btext = b""
        for id in ids:
            if id in self.vocab:
                btext += self.vocab[id]
            else:
                print(f"Token '{id}' is not found")
                continue
                
            if end_token_id is not None and id == end_token_id:
                break

        return btext.decode("utf-8", errors="ignore")
