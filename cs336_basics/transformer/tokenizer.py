import os
import heapq
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

                found_index = min_chunk.find(split_tokens)
                if found_index != -1:
                    chunks[i] = index + found_index
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
        bword_dict = Counter()
        word_dict = Counter()

        for sent in sents:
            if not sent:
                continue
            matches = [word.group(0) for word in self.word_pattern.finditer(sent)]
            word_dict.update(matches)

        # String into byte frequencies
        for word, freq in word_dict.items():
            bword_dict[word.encode("utf-8")] = freq

        return bword_dict

    def pretokenize (self, sent: str) -> List[bytes]:
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
        self.word2id = defaultdict(list)
    
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
        words = self.pretokenizer.pretokenize (sent)
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

    def encode_iterable(self, sents: Iterable[str]) -> Iterable[int]:
        word_iter = self.pretokenizer.pretokenize_iter(sents)
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


class Trainer:
    def __init__(self, vocab_size: int, special_tokens: List[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.preprocessor = PreTokenizer(special_tokens)
        self.vocab = defaultdict(bytes)
        self.merges = []
        self.splits = defaultdict(list)
        self.pairs_f = defaultdict(int)
        self.pair2word = defaultdict(set)
        self.freq_heap = []

    def init(self, word_freq: Dict):
        for word, word_freq in word_freq.items():
            self.splits[word] = [bytes([s]) for s in word]

            word_pieces = self.splits[word]

            if len(word_pieces) == 1:
                continue
            for pair in zip(word_pieces[:-1], word_pieces[1:]):
                self.pairs_f[pair] = self.pairs_f.get(pair, 0) + word_freq

                self.pair2word[pair].add(word)

        for pair, freq in self.pairs_f.items():
            heapq.heappush(self.freq_heap, (-freq, pair))

    def get_pairs(self):
        while self.freq_heap:
            freq, pair = heapq.heappop(self.freq_heap)
            freq *= -1
            if pair in self.pairs_f and self.pairs_f[pair] == freq:
                return pair
        raise ValueError("Heap does not return frequent pair")
    
    def update_pair_freq(self, new_pair, old_pair, word, word_freq):
        self.pair2word.setdefault(new_pair, set()).add(word)
        self.pairs_f[new_pair] = self.pairs_f.get(new_pair, 0) + word_freq
        heapq.heappush(self.freq_heap, (-self.pairs_f[new_pair], new_pair))

        if old_pair in self.pairs_f:
            self.pairs_f[old_pair] -= word_freq
            if self.pairs_f[old_pair] <= 0:
                del self.pairs_f[old_pair]
            else:
                heapq.heappush(self.freq_heap, (-self.pairs_f[old_pair], old_pair))

    def update(
        self,
        best_pair: Tuple[bytes, bytes],
        new_token: bytes,
        word_freqs: Dict,
    ):
        for word in list(self.pair2word.get(best_pair, set())):
            word_freq = word_freqs[word]
            word_pieces = self.splits[word]
            index = 0
            while index < len(word_pieces) - 1:
                if (
                    word_pieces[index] == best_pair[0]
                    and word_pieces[index + 1] == best_pair[1]
                ):
                    word_pieces[index] = new_token
                    word_pieces.pop(index + 1)
                    
                    if self.pairs_f[best_pair] <= 0:
                        del self.pairs_f[best_pair]
                    else:
                        heapq.heappush(self.freq_heap, (-self.pairs_f[best_pair], best_pair))

                    if index > 0:
                        new_pair_left = (word_pieces[index - 1], new_token)
                        old_pair_left = (word_pieces[index - 1], best_pair[0])
                        self.update_pair_freq(new_pair_left, old_pair_left, word, word_freq)
                    if index < len(word_pieces) - 1:
                        new_pair_right = (new_token, word_pieces[index + 1])
                        old_pair_right = (best_pair[1], word_pieces[index + 1])
                        self.update_pair_freq(new_pair_right, old_pair_right, word, word_freq)
                else:
                    index += 1

    def add_special_tokens(self):
        # Create list of special tokens in bytes
        bspecial_tokens = [token.encode("utf-8") for token in self.special_tokens]
        
        # Remove special tokens from current vocabulary
        to_remove = []
        for idx, token_bytes in self.token_vocab.items():
            if token_bytes in bspecial_tokens:
                to_remove.append(idx)
        for idx in to_remove:
            del self.token_vocab[idx]
        
        # Add special tokens at reserved positions
        for i, token in enumerate(self.special_tokens):
            reserved_id = self.vocab_size - len(self.special_tokens) + i
            self.token_vocab[reserved_id] = token.encode("utf-8")

    def train(
        self,
        input_path: str
    ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        word_freq = Counter()
        for chunk  in self.preprocessor.read(input_path):
            word_freq.update(self.preprocessor.build_word_frequency(chunk))

        self.token_vocab = {i: bytes([i]) for i in range(256)}
        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        self.merges = []

        self.init(word_freq)

        for merge_idx in range(num_merges):
            if not self.pairs_f:
                break

            best_pair = self.get_pairs()
            self.merges.append(best_pair)

            new_token = best_pair[0] + best_pair[1]
            self.token_vocab[256 + merge_idx] = new_token

            self.update(best_pair, new_token, word_freq)

        self.add_special_tokens()

        return self.token_vocab, self.merges

    
