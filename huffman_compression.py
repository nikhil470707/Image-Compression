"""Huffman compression utilities with optional Fibonacci heap.

This module implements Huffman coding with optional DPCM (differential
encoding) and RLE (run-length encoding) preprocessing. It includes a
HuffmanCompressor class and helper routines for encoding/decoding and
performance comparison.
"""

import pickle
import time
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any, Union
import heapq

from fibonacci_heap import FibonacciHeap, FibNode

class HuffmanNode:
    # Node in Huffman tree
    def __init__(self, symbol: Any = None, frequency: int = 0, 
                 left: Optional['HuffmanNode'] = None, 
                 right: Optional['HuffmanNode'] = None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = left
        self.right = right
        self.is_leaf = symbol is not None
    
    def __lt__(self, other):
        if self.frequency != other.frequency:
            return self.frequency < other.frequency
        if self.is_leaf and other.is_leaf:
            return str(self.symbol) < str(other.symbol)
        return self.is_leaf < other.is_leaf
    
    def __repr__(self):
        if self.is_leaf:
            return f"Leaf({self.symbol}:{self.frequency})"
        return f"Internal({self.frequency})"

class HuffmanCompressor:
    # Huffman compressor with optional preprocessing and heap choice.
    
    def __init__(self, use_fibonacci_heap: bool = True, 
                 use_rle: bool = False, 
                 use_diff_encoding: bool = True):
        self.use_fibonacci_heap = use_fibonacci_heap
        self.use_rle = use_rle
        self.use_diff_encoding = use_diff_encoding
        self.huffman_tree: Optional[HuffmanNode] = None
        self.codes: Dict[Any, str] = {}
        self.reverse_codes: Dict[str, Any] = {}
        
    def _differential_encode(self, data: List[int]) -> List[int]:
        # Apply 1D differential encoding (DPCM)
        if not data:
            return []
        
        encoded = [data[0]]
        for i in range(1, len(data)):
            diff = data[i] - data[i-1]
            encoded.append(diff)
        
        return encoded

    def _differential_decode(self, encoded_data: List[int]) -> List[int]:
        # Decode differential-encoded data
        if not encoded_data:
            return []
        
        decoded = [encoded_data[0]]
        for i in range(1, len(encoded_data)):
            original_value = decoded[-1] + encoded_data[i]
            decoded.append(original_value)
        
        return decoded
    
    def _run_length_encode(self, data: List[Any]) -> List[Tuple[Any, int]]:
        # Apply run-length encoding (RLE)
        if not data:
            return []
        
        encoded = []
        current_symbol = data[0]
        count = 1
        MAX_COUNT = 255

        for symbol in data[1:]:
            if symbol == current_symbol and count < MAX_COUNT:
                count += 1
            else:
                encoded.append((current_symbol, count))
                current_symbol = symbol
                count = 1
        
        encoded.append((current_symbol, count))
        return encoded
    
    def _run_length_decode(self, encoded_data: List[Tuple[Any, int]]) -> List[Any]:
        # Decode run-length encoded data
        decoded = []
        for symbol, count in encoded_data:
            decoded.extend([symbol] * count)
        return decoded
    
    def _build_frequency_table(self, data: Union[List[Any], List[Tuple[Any, int]]]) -> Dict[Any, int]:
        # Build frequency table from data
        return dict(Counter(data))
    
    def _bitstring_to_bytes(self, bitstring: str) -> Tuple[bytes, int]:
        # Convert bit string to bytes and return padding length
        padding_bits = (8 - (len(bitstring) % 8)) % 8
        padded_bitstring = bitstring + '0' * padding_bits
        
        byte_array = bytearray()
        for i in range(0, len(padded_bitstring), 8):
            byte = padded_bitstring[i:i+8]
            byte_array.append(int(byte, 2))
        
        return bytes(byte_array), padding_bits

    def _bytes_to_bitstring(self, byte_data: bytes, padding_bits: int) -> str:
        # Convert bytes back to bit string and strip padding
        bitstring = ''.join(f'{byte:08b}' for byte in byte_data)
        
        if padding_bits > 0:
            bitstring = bitstring[:-padding_bits]
        
        return bitstring
    
    def _serialize_tree(self, node: Optional[HuffmanNode]) -> Optional[Union[Any, Tuple]]:
        # Convert Huffman tree to serializable structure
        if node is None:
            return None
        
        if node.is_leaf:
            return node.symbol
        else:
            left = self._serialize_tree(node.left)
            right = self._serialize_tree(node.right)
            return ('I', left, right)
    
    def _deserialize_tree(self, serialized_tree: Optional[Union[Any, Tuple]]) -> Optional[HuffmanNode]:
        # Convert serialized structure back to a HuffmanNode
        if serialized_tree is None:
            return None
        
        if isinstance(serialized_tree, tuple) and len(serialized_tree) == 3 and serialized_tree[0] == 'I':
            _, left_serialized, right_serialized = serialized_tree
            left_node = self._deserialize_tree(left_serialized)
            right_node = self._deserialize_tree(right_serialized)
            return HuffmanNode(left=left_node, right=right_node, frequency=0)
        else:
            return HuffmanNode(symbol=serialized_tree, frequency=0)
    
    def _build_huffman_tree_fibonacci(self, frequency_table: Dict[Any, int]) -> Optional[HuffmanNode]:
        # Build Huffman tree using a Fibonacci heap
        if not frequency_table:
            return None
        
        if len(frequency_table) == 1:
            symbol = list(frequency_table.keys())[0]
            frequency = list(frequency_table.values())[0]
            return HuffmanNode(symbol, frequency)
        
        fib_heap = FibonacciHeap()
        
        for symbol, frequency in frequency_table.items():
            node = HuffmanNode(symbol, frequency)
            fib_heap.insert(frequency, node)
        
        while fib_heap.size() > 1:
            fib_node1 = fib_heap.extract_min()
            fib_node2 = fib_heap.extract_min()
            
            huffman_node1 = fib_node1.data
            huffman_node2 = fib_node2.data
            
            merged_frequency = huffman_node1.frequency + huffman_node2.frequency
            merged_node = HuffmanNode(
                frequency=merged_frequency,
                left=huffman_node1 if huffman_node1.frequency <= huffman_node2.frequency else huffman_node2,
                right=huffman_node2 if huffman_node1.frequency <= huffman_node2.frequency else huffman_node1
            )
            
            fib_heap.insert(merged_frequency, merged_node)
        
        root_fib_node = fib_heap.extract_min()
        return root_fib_node.data if root_fib_node else None

    def _build_huffman_tree_binary_heap(self, frequency_table: Dict[Any, int]) -> Optional[HuffmanNode]:
        # Build Huffman tree using a binary heap
        if not frequency_table:
            return None
        
        if len(frequency_table) == 1:
            symbol = list(frequency_table.keys())[0]
            frequency = list(frequency_table.values())[0]
            return HuffmanNode(symbol, frequency)
        
        heap = []
        for symbol, frequency in frequency_table.items():
            node = HuffmanNode(symbol, frequency)
            heapq.heappush(heap, (frequency, id(node), node))
        
        while len(heap) > 1:
            freq1, _, node1 = heapq.heappop(heap)
            freq2, _, node2 = heapq.heappop(heap)
            
            merged_frequency = freq1 + freq2
            merged_node = HuffmanNode(
                frequency=merged_frequency,
                left=node1 if node1.frequency <= node2.frequency else node2,
                right=node2 if node1.frequency <= node2.frequency else node1
            )
            
            heapq.heappush(heap, (merged_frequency, id(merged_node), merged_node))
        
        return heap[0][2] if heap else None
    
    def _generate_codes(self, root: HuffmanNode) -> Dict[Any, str]:
        # Generate Huffman codes from the tree
        if root is None:
            return {}
        
        codes = {}
        
        def traverse(node: HuffmanNode, code: str = ""):
            if node.is_leaf:
                codes[node.symbol] = code if code else "0"
                return
            
            if node.left:
                traverse(node.left, code + "0")
            if node.right:
                traverse(node.right, code + "1")
        
        traverse(root)
        return codes
    
    def _encode_data(self, data: Union[List[Any], List[Tuple[Any, int]]], codes: Dict[Any, str]) -> str:
        # Encode data using Huffman codes
        encoded_bits = []
        
        for symbol in data:
            if symbol in codes:
                encoded_bits.append(codes[symbol])
            else:
                raise ValueError(f"Symbol {symbol} not found in Huffman codes")
        
        return ''.join(encoded_bits)
    
    def _decode_data(self, encoded_bits: str, root: HuffmanNode) -> List[Any]:
        # Decode a bit string using the Huffman tree
        if root is None:
            return []
        
        if root.is_leaf:
            return [root.symbol] * len(encoded_bits) if encoded_bits else []
        
        decoded = []
        current_node = root
        
        for bit in encoded_bits:
            if bit == '0':
                current_node = current_node.left
            else:
                current_node = current_node.right
            
            if current_node is None:
                raise ValueError("Invalid encoded data")
            
            if current_node.is_leaf:
                decoded.append(current_node.symbol)
                current_node = root
        
        return decoded
    
    def compress(self, data: List[Any]) -> Tuple[bytes, Dict]:
        # Compress data using Huffman coding with optional DPCM and RLE
        if not data:
            return b"", {"original_length": 0, "compressed_length": 0, "tree": None}
        
        start_time = time.time()
        
        # 1. Differential Encoding (DPCM)
        processed_data = data
        if self.use_diff_encoding:
            processed_data = self._differential_encode(data)
        
        # 2. RLE Preprocessing
        if self.use_rle:
            processed_data = self._run_length_encode(processed_data)
        
        # 3. Build frequency table
        frequency_table = self._build_frequency_table(processed_data)
        
        # 4. Build Huffman tree
        if self.use_fibonacci_heap:
            self.huffman_tree = self._build_huffman_tree_fibonacci(frequency_table)
        else:
            self.huffman_tree = self._build_huffman_tree_binary_heap(frequency_table)
        
        # 5. Generate codes and encode
        self.codes = self._generate_codes(self.huffman_tree)
        self.reverse_codes = {v: k for k, v in self.codes.items()}
        encoded_bits = self._encode_data(processed_data, self.codes)
        
        # 6. Convert to bytes
        encoded_bytes, padding_bits = self._bitstring_to_bytes(encoded_bits)
        
        compression_time = time.time() - start_time
        
        metadata = {
            "serialized_tree": self._serialize_tree(self.huffman_tree),
            "original_length": len(data),
            "compressed_length": len(encoded_bits),
            "padding_bits": padding_bits,
            "use_rle": self.use_rle,
            "use_diff_encoding": self.use_diff_encoding,
            "compression_time": compression_time,
            "heap_type": "fibonacci" if self.use_fibonacci_heap else "binary"
        }
        
        return encoded_bytes, metadata
    
    def decompress(self, encoded_bytes: bytes, metadata: Dict) -> List[Any]:
        # Decompress encoded data using metadata
        start_time = time.time()
        
        tree = self._deserialize_tree(metadata["serialized_tree"])
        use_rle = metadata["use_rle"]
        use_diff_encoding = metadata.get("use_diff_encoding", False)
        padding_bits = metadata.get("padding_bits", 0)
        
        # Convert bytes to bit string
        encoded_bits = self._bytes_to_bitstring(encoded_bytes, padding_bits)
        
        # Decode using Huffman tree
        decoded_data = self._decode_data(encoded_bits, tree)
        
        # Reverse RLE
        if use_rle:
            decoded_data = self._run_length_decode(decoded_data)
        
        # Reverse DPCM
        if use_diff_encoding:
            decoded_data = self._differential_decode([int(x) for x in decoded_data])
        
        decompression_time = time.time() - start_time
        
        return decoded_data
    
    def get_compression_stats(self, original_data: List[Any], encoded_bytes: bytes, metadata: Dict) -> Dict:
        # Calculate compression statistics
        original_bits = len(original_data) * 8
        compressed_bits = metadata.get("compressed_length", len(encoded_bytes) * 8)
        
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else float('inf')
        space_saving = ((original_bits - compressed_bits) / original_bits) * 100 if original_bits > 0 else 0
        
        return {
            "original_bits": original_bits,
            "compressed_bits": compressed_bits,
            "compression_ratio": compression_ratio,
            "space_saving_percentage": space_saving,
            "compression_time": metadata.get("compression_time", 0),
            "heap_type": metadata.get("heap_type", "unknown"),
            "use_rle": metadata.get("use_rle", False),
            "use_diff_encoding": metadata.get("use_diff_encoding", False)
        }

def compare_heap_performance(data: List[Any], iterations: int = 1) -> Dict:
    """Compare Fibonacci heap vs Binary heap performance"""
    fib_times = []
    for _ in range(iterations):
        compressor_fib = HuffmanCompressor(use_fibonacci_heap=True, use_rle=False, use_diff_encoding=True)
        start_time = time.time()
        compressor_fib.compress(data)
        fib_times.append(time.time() - start_time)
    
    binary_times = []
    for _ in range(iterations):
        compressor_bin = HuffmanCompressor(use_fibonacci_heap=False, use_rle=False, use_diff_encoding=True)
        start_time = time.time()
        compressor_bin.compress(data)
        binary_times.append(time.time() - start_time)
    
    return {
        "fibonacci_heap_avg_time": sum(fib_times) / len(fib_times),
        "binary_heap_avg_time": sum(binary_times) / len(binary_times),
        "fibonacci_heap_times": fib_times,
        "binary_heap_times": binary_times,
        "data_size": len(data)
    }