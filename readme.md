# Image Compression System

A lossless image compression system implementing Huffman Coding with Fibonacci Heap optimization, DPCM (Differential Pulse Code Modulation), and Run Length Encoding.

## Project Overview

This project demonstrates advanced data structures and algorithms in a practical application:

- **Fibonacci Heap**: O(1) amortized insert and decrease-key operations
- **Huffman Coding**: Optimal prefix-free encoding for lossless compression
- **DPCM**: Differential encoding for natural images
- **RLE**: Run-length encoding for repeated patterns
- **Channel-wise Processing**: Independent RGB channel compression

**Performance**: Achieves 2-4:1 compression ratios on raw pixel data with perfect reconstruction.

---

## Installation

```bash
pip install numpy pillow

git clone <repository-url>
cd image-compression-system
```

## Usage

### Command Line Interface

```bash
# Compress an image
python image_compressor.py compress input.png output.huffimg

# Decompress an image
python image_compressor.py decompress output.huffimg restored.png


# Options:
#   --rle              Enable Run Length Encoding
#   --no-diff          Disable DPCM
#   --no-fibonacci     Use binary heap instead
```

### Run Demo

```bash
# Full demonstration with all tests
python main_demo.py --demo

# Create sample images
python main_demo.py --create-samples

# Quick functionality test
python main_demo.py --quick-test
```

---

## Performance Characteristics

### Theoretical Complexity

| Operation | Fibonacci Heap | Binary Heap |
|-----------|---------------|-------------|
| Insert | O(1) amortized | O(log n) |
| Extract-min | O(log n) amortized | O(log n) |
| Decrease-key | O(1) amortized | O(log n) |
| Merge | O(1) | O(n) |


## Technical Implementation

### Architecture

```
Raw Image Input (RGB/Grayscale)
    |
    v
Split into channels (R, G, B)
    |
    v
For each channel:
    1. DPCM (optional)
    2. RLE (optional)
    3. Huffman encoding
        - Build frequency table
        - Construct tree using Fibonacci heap
        - Generate optimal codes
        - Encode to bit stream
    4. Pack bits to bytes
    |
    v
Save as .huffimg with metadata (tree structure, parameters)
```

### Key Features

**1. Fibonacci Heap**
- Lazy consolidation: Defers work until extract-min
- Cascading cuts: Maintains heap property efficiently
- Marked nodes: Tracks nodes for optimal performance
- Circular doubly-linked lists: Efficient node management

**2. DPCM (Differential Encoding)**
```python
# Instead of storing: [100, 102, 101, 103, ...]
# Store differences:   [100, +2, -1, +2, ...]
# Result: Smaller value range leads to better compression
```

**3. Huffman Coding**
- Variable-length codes based on frequency
- More frequent symbols receive shorter codes
- Prefix-free: No code is prefix of another
- Optimal for symbol-by-symbol encoding

**4. Channel-wise Compression**
- Each RGB channel compressed independently
- Exploits inter-channel redundancy
- Allows parallel processing (future optimization)

---


## Educational Value

This project demonstrates:

**1. Advanced Data Structures**
- Fibonacci heaps with amortized analysis
- Binary trees (Huffman trees)
- Hash tables for frequency counting

**2. Algorithm Design**
- Greedy algorithms (Huffman coding)
- Dynamic programming concepts
- Amortized complexity analysis

**3. System Design**
- Custom file format design
- Serialization and deserialization
- Modular architecture

**4. Software Engineering**
- Clean code principles
- Comprehensive testing
- Performance benchmarking
- Documentation

---

## Important Notes

### Capabilities

This system:
- Compresses raw pixel data efficiently (2-4:1 ratios)
- Demonstrates advanced data structures and algorithms
- Achieves perfect lossless reconstruction
- Shows Fibonacci heap advantages in practice

### Limitations

This system does not:
- Beat PNG/JPEG file sizes (they use similar techniques plus decades of optimization)
- Provide production-ready compression (PNG/JPEG are better for that)
- Compress already-compressed data effectively

### Understanding the Results

Compression ratios are calculated against raw pixel data, not against PNG/JPEG files:

- **Raw pixel data**: width × height × channels × 1 byte (uncompressed)
- **PNG file**: Already compressed with DEFLATE + filters
- **JPEG file**: Already lossy-compressed with DCT + Huffman

Attempting to compress PNG/JPEG files will result in larger files because they are already near-optimally compressed.

### Best Use Cases

This system works well for:
- Compressing BMP files (uncompressed format)
- Compressing RAW camera data
- Compressing uncompressed TIFF files
- Learning and demonstrating data structures and algorithms

---


## Testing

```bash
# Run all tests
python main_demo.py --demo

# Test individual components
python -c "from fibonacci_heap import FibonacciHeap; heap = FibonacciHeap(); heap.insert(5, 'test'); print('Heap works')"

python -c "from huffman_compression import HuffmanCompressor; c = HuffmanCompressor(); print('Huffman works')"

python -c "from image_compressor import ImageCompressor; c = ImageCompressor(); print('Image compressor works')"
```

---
