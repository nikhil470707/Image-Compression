"""
Huffman Compression Module using Fibonacci Heap
Supports both grayscale and color image compression with RLE preprocessing
"""

import pickle
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
import heapq  # For comparison with binary heap

from fibonacci_heap import FibonacciHeap, FibNode

class HuffmanNode:
    """Node in Huffman tree"""
    def __init__(self, symbol: Any = None, frequency: int = 0, left: Optional['HuffmanNode'] = None, right: Optional['HuffmanNode'] = None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = left
        self.right = right
        self.is_leaf = symbol is not None
    
    def __lt__(self, other):
        if self.frequency != other.frequency:
            return self.frequency < other.frequency
        # Break ties consistently
        if self.is_leaf and other.is_leaf:
            return str(self.symbol) < str(other.symbol)
        return self.is_leaf < other.is_leaf
    
    def __repr__(self):
        if self.is_leaf:
            return f"Leaf({self.symbol}:{self.frequency})"
        return f"Internal({self.frequency})"

class HuffmanCompressor:
    """
    Huffman Compressor using Fibonacci Heap for optimal performance
    Supports RLE preprocessing for better compression ratios
    """
    
    def __init__(self, use_fibonacci_heap: bool = True, use_rle: bool = True):
        self.use_fibonacci_heap = use_fibonacci_heap
        self.use_rle = use_rle
        self.huffman_tree: Optional[HuffmanNode] = None
        self.codes: Dict[Any, str] = {}
        self.reverse_codes: Dict[str, Any] = {}
        
    def _run_length_encode(self, data: List[Any]) -> List[Tuple[Any, int]]:
        """Apply Run Length Encoding to data"""
        if not data:
            return []
        
        encoded = []
        current_symbol = data[0]
        count = 1
        
        for symbol in data[1:]:
            if symbol == current_symbol and count < 255:  # Limit count to prevent overflow
                count += 1
            else:
                encoded.append((current_symbol, count))
                current_symbol = symbol
                count = 1
        
        encoded.append((current_symbol, count))
        return encoded
    
    def _run_length_decode(self, encoded_data: List[Tuple[Any, int]]) -> List[Any]:
        """Decode Run Length Encoded data"""
        decoded = []
        for symbol, count in encoded_data:
            decoded.extend([symbol] * count)
        return decoded
    
    def _build_frequency_table(self, data: Union[List[Any], List[Tuple[Any, int]]]) -> Dict[Any, int]:
        """Build frequency table from data"""
        if self.use_rle and isinstance(data[0], tuple):
            # Data is already RLE encoded
            frequency_table = defaultdict(int)
            for symbol, count in data:
                frequency_table[(symbol, count)] += 1
            return dict(frequency_table)
        else:
            return dict(Counter(data))
    
    def _build_huffman_tree_fibonacci(self, frequency_table: Dict[Any, int]) -> Optional[HuffmanNode]:
        """Build Huffman tree using Fibonacci Heap"""
        if not frequency_table:
            return None
        
        if len(frequency_table) == 1:
            # Special case: only one unique symbol
            symbol = list(frequency_table.keys())[0]
            frequency = list(frequency_table.values())[0]
            return HuffmanNode(symbol, frequency)
        
        # Create Fibonacci heap and insert all symbols
        fib_heap = FibonacciHeap()
        
        for symbol, frequency in frequency_table.items():
            node = HuffmanNode(symbol, frequency)
            fib_heap.insert(frequency, node)
        
        # Build tree by repeatedly merging two smallest nodes
        while fib_heap.size() > 1:
            # Extract two minimum nodes
            fib_node1 = fib_heap.extract_min()
            fib_node2 = fib_heap.extract_min()
            
            huffman_node1 = fib_node1.data
            huffman_node2 = fib_node2.data
            
            # Create new internal node
            merged_frequency = huffman_node1.frequency + huffman_node2.frequency
            merged_node = HuffmanNode(
                frequency=merged_frequency,
                left=huffman_node1,
                right=huffman_node2
            )
            
            # Insert back into heap
            fib_heap.insert(merged_frequency, merged_node)
        
        # Return root of tree
        root_fib_node = fib_heap.extract_min()
        return root_fib_node.data if root_fib_node else None
    
    def _build_huffman_tree_binary_heap(self, frequency_table: Dict[Any, int]) -> Optional[HuffmanNode]:
        """Build Huffman tree using standard binary heap (for comparison)"""
        if not frequency_table:
            return None
        
        if len(frequency_table) == 1:
            symbol = list(frequency_table.keys())[0]
            frequency = list(frequency_table.values())[0]
            return HuffmanNode(symbol, frequency)
        
        # Create binary heap with all symbols
        heap = []
        for symbol, frequency in frequency_table.items():
            node = HuffmanNode(symbol, frequency)
            heapq.heappush(heap, (frequency, id(node), node))
        
        # Build tree
        while len(heap) > 1:
            freq1, _, node1 = heapq.heappop(heap)
            freq2, _, node2 = heapq.heappop(heap)
            
            merged_frequency = freq1 + freq2
            merged_node = HuffmanNode(
                frequency=merged_frequency,
                left=node1,
                right=node2
            )
            
            heapq.heappush(heap, (merged_frequency, id(merged_node), merged_node))
        
        return heap[0][2] if heap else None
    
    def _generate_codes(self, root: HuffmanNode) -> Dict[Any, str]:
        """Generate Huffman codes from tree"""
        if root is None:
            return {}
        
        codes = {}
        
        def traverse(node: HuffmanNode, code: str = ""):
            if node.is_leaf:
                codes[node.symbol] = code if code else "0"  # Handle single symbol case
                return
            
            if node.left:
                traverse(node.left, code + "0")
            if node.right:
                traverse(node.right, code + "1")
        
        traverse(root)
        return codes
    
    def _encode_data(self, data: Union[List[Any], List[Tuple[Any, int]]], codes: Dict[Any, str]) -> str:
        """Encode data using Huffman codes"""
        encoded_bits = []
        
        for symbol in data:
            if symbol in codes:
                encoded_bits.append(codes[symbol])
            else:
                raise ValueError(f"Symbol {symbol} not found in Huffman codes")
        
        return ''.join(encoded_bits)
    
    def _decode_data(self, encoded_bits: str, root: HuffmanNode) -> List[Any]:
        """Decode bit string using Huffman tree"""
        if root is None:
            return []
        
        if root.is_leaf:
            # Special case: only one unique symbol
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
    
    def compress(self, data: List[Any]) -> Tuple[str, Dict]:
        """
        Compress data using Huffman coding with optional RLE preprocessing
        
        Returns:
            Tuple of (encoded_bits, metadata)
        """
        if not data:
            return "", {"original_length": 0, "compressed_length": 0, "tree": None, "use_rle": False}
        
        start_time = time.time()
        
        # Step 1: Optional RLE preprocessing
        processed_data = data
        if self.use_rle:
            processed_data = self._run_length_encode(data)
        
        # Step 2: Build frequency table
        frequency_table = self._build_frequency_table(processed_data)
        
        # Step 3: Build Huffman tree
        if self.use_fibonacci_heap:
            self.huffman_tree = self._build_huffman_tree_fibonacci(frequency_table)
        else:
            self.huffman_tree = self._build_huffman_tree_binary_heap(frequency_table)
        
        # Step 4: Generate codes
        self.codes = self._generate_codes(self.huffman_tree)
        self.reverse_codes = {v: k for k, v in self.codes.items()}
        
        # Step 5: Encode data
        encoded_bits = self._encode_data(processed_data, self.codes)
        
        compression_time = time.time() - start_time
        
        # Metadata for decompression
        metadata = {
            "tree": self.huffman_tree,
            "original_length": len(data),
            "compressed_length": len(encoded_bits),
            "use_rle": self.use_rle,
            "compression_time": compression_time,
            "heap_type": "fibonacci" if self.use_fibonacci_heap else "binary"
        }
        
        return encoded_bits, metadata
    
    def decompress(self, encoded_bits: str, metadata: Dict) -> List[Any]:
        """
        Decompress encoded data using metadata
        
        Args:
            encoded_bits: Encoded bit string
            metadata: Metadata from compression
            
        Returns:
            Decompressed data
        """
        start_time = time.time()
        
        tree = metadata["tree"]
        use_rle = metadata["use_rle"]
        
        # Decode using Huffman tree
        decoded_data = self._decode_data(encoded_bits, tree)
        
        # Apply RLE decoding if it was used during compression
        if use_rle:
            decoded_data = self._run_length_decode(decoded_data)
        
        decompression_time = time.time() - start_time
        print(f"Decompression time: {decompression_time:.4f} seconds")
        
        return decoded_data
    
    def get_compression_stats(self, original_data: List[Any], encoded_bits: str, metadata: Dict) -> Dict:
        """Calculate compression statistics"""
        original_bits = len(original_data) * 8  # Assuming 8 bits per symbol
        compressed_bits = len(encoded_bits)
        
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else float('inf')
        space_saving = ((original_bits - compressed_bits) / original_bits) * 100 if original_bits > 0 else 0
        
        return {
            "original_bits": original_bits,
            "compressed_bits": compressed_bits,
            "compression_ratio": compression_ratio,
            "space_saving_percentage": space_saving,
            "compression_time": metadata.get("compression_time", 0),
            "heap_type": metadata.get("heap_type", "unknown"),
            "use_rle": metadata.get("use_rle", False)
        }

def compare_heap_performance(data: List[Any], iterations: int = 1) -> Dict:
    """Compare performance between Fibonacci heap and Binary heap"""
    
    # Test with Fibonacci heap
    fib_times = []
    for _ in range(iterations):
        compressor_fib = HuffmanCompressor(use_fibonacci_heap=True)
        start_time = time.time()
        encoded_bits, metadata = compressor_fib.compress(data)
        fib_times.append(time.time() - start_time)
    
    # Test with Binary heap
    binary_times = []
    for _ in range(iterations):
        compressor_bin = HuffmanCompressor(use_fibonacci_heap=False)
        start_time = time.time()
        encoded_bits, metadata = compressor_bin.compress(data)
        binary_times.append(time.time() - start_time)
    
    return {
        "fibonacci_heap_avg_time": sum(fib_times) / len(fib_times),
        "binary_heap_avg_time": sum(binary_times) / len(binary_times),
        "fibonacci_heap_times": fib_times,
        "binary_heap_times": binary_times,
        "data_size": len(data)
    }
