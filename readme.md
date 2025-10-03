# Advanced Color Image Compression System

A sophisticated image compression system implementing **Huffman Coding with Fibonacci Heap optimization** and **Run Length Encoding (RLE) preprocessing**. 

## Key Features

### Advanced Data Structures
- **Fibonacci Heap**: O(1) amortized insert and decrease-key operations for optimal Huffman tree construction
- **Huffman Trees**: Optimal prefix-free encoding for lossless compression
- **Hash Tables**: Efficient frequency counting and code mapping

### Compression Techniques
- **Channel-wise Compression**: Separate compression of RGB channels for better efficiency
- **Run Length Encoding (RLE)**: Preprocessing for images with repeated pixel patterns
- **Lossless Compression**: Perfect reconstruction guaranteed
- **Custom File Format**: `.huffimg` format with embedded metadata

### Performance Analysis
- **Heap Comparison**: Fibonacci Heap vs Binary Heap performance analysis
- **Compression Metrics**: Detailed statistics including compression ratios, space savings, and timing
- **Quality Metrics**: MSE and PSNR calculations for reconstruction quality


## Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd image-compression-system

#### Command Line Interface
```bash
# Compress an image
python image_compressor.py compress input.png output.huffimg

# Decompress an image
python image_compressor.py decompress output.huffimg restored.png

# Run comprehensive analysis
python image_compressor.py analyze input.png analysis_results/
```

#### Quick Testing
```bash
# Run all tests
python main_demo.py --demo

# Create sample images for testing
python main_demo.py --create-samples

# Quick functionality test
python main_demo.py --quick-test
```


### Theoretical Advantages of Fibonacci Heap

- **Insert**: O(1) amortized vs O(log n) for binary heap
- **Decrease-key**: O(1) amortized vs O(log n) for binary heap
- **Extract-min**: O(log n) amortized (same as binary heap)
- **Merge**: O(1) vs O(n) for binary heap

## Technical Implementation

### Fibonacci Heap Features
- **Cascading Cut**: Maintains heap property efficiently
- **Lazy Consolidation**: Defers work until extract-min operations
- **Marked Nodes**: Tracks nodes for cascading cuts
- **Circular Doubly-Linked Lists**: Efficient node management

### Huffman Compression Pipeline
1. **Frequency Analysis**: Count pixel/symbol occurrences
2. **RLE Preprocessing**: (Optional) Reduce repeated patterns
3. **Heap Construction**: Build priority queue with frequencies
4. **Tree Building**: Merge nodes using heap operations
5. **Code Generation**: Create optimal prefix codes
6. **Encoding**: Convert data to compressed bit string
7. **Serialization**: Save compressed data with metadata

### Channel-wise Processing
```
RGB Image (Height × Width × 3)
    ↓
Split into R, G, B channels
    ↓
Compress each channel independently
    ↓
Combine compressed channels
    ↓
Save as .huffimg file
```


### Direct Huffman Usage
```python
from huffman_compression import HuffmanCompressor

# Create compressor
compressor = HuffmanCompressor(use_fibonacci_heap=True)

# Compress data
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
encoded_bits, metadata = compressor.compress(data)

# Decompress
decoded_data = compressor.decompress(encoded_bits, metadata)
```

### Fibonacci Heap Usage
```python
from fibonacci_heap import FibonacciHeap

# Create heap
heap = FibonacciHeap()

# Insert elements
node1 = heap.insert(5, "five")
node2 = heap.insert(3, "three")
node3 = heap.insert(7, "seven")

# Extract minimum
min_node = heap.extract_min()  # Returns node with key=3

# Decrease key (useful for dynamic algorithms)
heap.decrease_key(node3, 1)  # Change key from 7 to 1
```

---

