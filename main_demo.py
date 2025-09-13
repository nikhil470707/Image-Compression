"""
Main Demo Script for Color Image Compression with Huffman + Fibonacci Heap
Run comprehensive tests and demonstrations of the compression system
"""

import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from fibonacci_heap import FibonacciHeap
from huffman_compression import HuffmanCompressor, compare_heap_performance
from image_compressor import ImageCompressor

def create_sample_images():
    """Create sample images for testing if they don't exist"""
    os.makedirs("sample_images", exist_ok=True)
    
    # Create a gradient grayscale image
    gradient_gray = np.zeros((200, 300), dtype=np.uint8)
    for i in range(200):
        gradient_gray[i, :] = int((i / 200) * 255)
    
    Image.fromarray(gradient_gray, mode='L').save("sample_images/gradient_grayscale.png")
    
    # Create a simple RGB pattern
    rgb_pattern = np.zeros((150, 200, 3), dtype=np.uint8)
    rgb_pattern[:50, :, 0] = 255  # Red band
    rgb_pattern[50:100, :, 1] = 255  # Green band  
    rgb_pattern[100:, :, 2] = 255  # Blue band
    
    Image.fromarray(rgb_pattern, mode='RGB').save("sample_images/rgb_pattern.png")
    
    # Create a noisy image for better compression testing
    np.random.seed(42)
    noisy_image = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
    Image.fromarray(noisy_image, mode='RGB').save("sample_images/noisy_rgb.png")
    
    # Create an image with repeated patterns (good for RLE)
    pattern_img = np.zeros((120, 180, 3), dtype=np.uint8)
    for i in range(0, 120, 20):
        for j in range(0, 180, 30):
            color = [i * 2, j, (i + j) % 255]
            pattern_img[i:i+20, j:j+30] = color
    
    Image.fromarray(pattern_img, mode='RGB').save("sample_images/pattern_rgb.png")
    
    print("Sample images created in 'sample_images' directory")

def test_fibonacci_heap():
    """Test Fibonacci Heap implementation"""
    print("=== TESTING FIBONACCI HEAP ===\n")
    
    heap = FibonacciHeap()
    
    # Test basic operations
    print("Testing basic operations...")
    test_data = [(10, 'ten'), (3, 'three'), (15, 'fifteen'), (1, 'one'), (8, 'eight')]
    
    for key, data in test_data:
        heap.insert(key, data)
        print(f"Inserted: {key} -> {data}")
    
    print(f"Heap size: {heap.size()}")
    print(f"Minimum: {heap.get_min().key if heap.get_min() else None}")
    
    # Test extract_min
    print("\nExtracting minimums:")
    while not heap.is_empty():
        min_node = heap.extract_min()
        print(f"Extracted: {min_node.key} -> {min_node.data}")
    
    print("Fibonacci Heap test completed successfully!\n")

def test_huffman_basic():
    """Test basic Huffman compression"""
    print("=== TESTING HUFFMAN COMPRESSION ===\n")
    
    # Test data
    test_string = "hello world! this is a test string with repeated characters."
    test_data = list(test_string)
    
    print(f"Original data: '{test_string}'")
    print(f"Data length: {len(test_data)} characters")
    
    # Test with Fibonacci heap
    print("\n--- Testing with Fibonacci Heap ---")
    compressor_fib = HuffmanCompressor(use_fibonacci_heap=True, use_rle=False)
    encoded_bits, metadata = compressor_fib.compress(test_data)
    decoded_data = compressor_fib.decompress(encoded_bits, metadata)
    
    print(f"Encoded bits length: {len(encoded_bits)}")
    print(f"Reconstruction successful: {''.join(decoded_data) == test_string}")
    
    stats_fib = compressor_fib.get_compression_stats(test_data, encoded_bits, metadata)
    print(f"Compression ratio: {stats_fib['compression_ratio']:.2f}")
    print(f"Space saving: {stats_fib['space_saving_percentage']:.1f}%")
    
    # Test with Binary heap
    print("\n--- Testing with Binary Heap ---")
    compressor_bin = HuffmanCompressor(use_fibonacci_heap=False, use_rle=False)
    encoded_bits_bin, metadata_bin = compressor_bin.compress(test_data)
    decoded_data_bin = compressor_bin.decompress(encoded_bits_bin, metadata_bin)
    
    print(f"Encoded bits length: {len(encoded_bits_bin)}")
    print(f"Reconstruction successful: {''.join(decoded_data_bin) == test_string}")
    
    stats_bin = compressor_bin.get_compression_stats(test_data, encoded_bits_bin, metadata_bin)
    print(f"Compression ratio: {stats_bin['compression_ratio']:.2f}")
    print(f"Space saving: {stats_bin['space_saving_percentage']:.1f}%")
    
    print("Huffman compression test completed successfully!\n")

def test_rle_compression():
    """Test Run Length Encoding preprocessing"""
    print("=== TESTING RLE PREPROCESSING ===\n")
    
    # Create data with many repeated elements
    repeated_data = [1] * 50 + [2] * 30 + [3] * 20 + [1] * 40 + [4] * 10
    
    print(f"Test data: {len(repeated_data)} elements with patterns")
    print(f"Unique elements: {len(set(repeated_data))}")
    
    # Test without RLE
    print("\n--- Without RLE ---")
    compressor_no_rle = HuffmanCompressor(use_fibonacci_heap=True, use_rle=False)
    encoded_no_rle, metadata_no_rle = compressor_no_rle.compress(repeated_data)
    stats_no_rle = compressor_no_rle.get_compression_stats(repeated_data, encoded_no_rle, metadata_no_rle)
    
    print(f"Compression ratio: {stats_no_rle['compression_ratio']:.2f}")
    print(f"Compression time: {metadata_no_rle['compression_time']:.4f}s")
    
    # Test with RLE
    print("\n--- With RLE ---")
    compressor_rle = HuffmanCompressor(use_fibonacci_heap=True, use_rle=True)
    encoded_rle, metadata_rle = compressor_rle.compress(repeated_data)
    stats_rle = compressor_rle.get_compression_stats(repeated_data, encoded_rle, metadata_rle)
    
    print(f"Compression ratio: {stats_rle['compression_ratio']:.2f}")
    print(f"Compression time: {metadata_rle['compression_time']:.4f}s")
    
    # Verify decompression
    decoded_rle = compressor_rle.decompress(encoded_rle, metadata_rle)
    print(f"RLE reconstruction successful: {decoded_rle == repeated_data}")
    
    print("RLE preprocessing test completed successfully!\n")

def test_image_compression():
    """Test image compression functionality"""
    print("=== TESTING IMAGE COMPRESSION ===\n")
    
    # Ensure sample images exist
    create_sample_images()
    
    test_images = [
        ("sample_images/gradient_grayscale.png", "Grayscale Gradient"),
        ("sample_images/rgb_pattern.png", "RGB Pattern"),
        ("sample_images/pattern_rgb.png", "Patterned RGB (good for RLE)")
    ]
    
    compressor = ImageCompressor(use_fibonacci_heap=True, use_rle=True)
    
    for image_path, description in test_images:
        if os.path.exists(image_path):
            print(f"\n--- Testing: {description} ---")
            print(f"Image: {image_path}")
            
            try:
                # Compress
                compressed_path = f"{image_path}.huffimg"
                stats = compressor.compress_image(image_path, compressed_path)
                
                # Decompress
                decompressed_path = f"{image_path}_decompressed.png"
                decomp_stats = compressor.decompress_image(compressed_path, decompressed_path)
                
                # Verify reconstruction
                original_img = compressor.load_image(image_path)
                reconstructed_img = compressor.load_image(decompressed_path)
                perfect_match = np.array_equal(original_img, reconstructed_img)
                
                print(f"Original size: {stats['original_file_size']:,} bytes")
                print(f"Compressed size: {stats['compressed_file_size']:,} bytes")
                print(f"Compression ratio: {stats['file_compression_ratio']:.2f}:1")
                print(f"Space saving: {stats['file_space_saving']:.1f}%")
                print(f"Compression time: {stats['total_compression_time']:.3f}s")
                print(f"Decompression time: {decomp_stats['decompression_time']:.3f}s")
                print(f"Perfect reconstruction: {perfect_match}")
                
                if "channel_stats" in stats:
                    print("Per-channel compression ratios:")
                    for channel, channel_stats in stats["channel_stats"].items():
                        ratio = channel_stats.get('compression_ratio', 0)
                        print(f"  {channel.capitalize()}: {ratio:.2f}:1")
                
            except Exception as e:
                print(f"Error testing {image_path}: {str(e)}")
        else:
            print(f"Image not found: {image_path}")
    
    print("\nImage compression test completed!\n")

def performance_comparison():
    """Compare performance between different heap implementations"""
    print("=== PERFORMANCE COMPARISON ===\n")
    
    # Test with different data sizes
    data_sizes = [100, 500, 1000, 2000]
    
    print("Comparing Fibonacci Heap vs Binary Heap performance:")
    print("-" * 60)
    print(f"{'Size':<8} {'Fib Heap (s)':<12} {'Binary Heap (s)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for size in data_sizes:
        # Generate test data with reasonable distribution
        np.random.seed(42)  # For reproducible results
        test_data = np.random.randint(0, 256, size).tolist()
        
        # Compare performance
        perf_results = compare_heap_performance(test_data, iterations=3)
        
        fib_time = perf_results["fibonacci_heap_avg_time"]
        bin_time = perf_results["binary_heap_avg_time"]
        speedup = bin_time / fib_time if fib_time > 0 else float('inf')
        
        print(f"{size:<8} {fib_time:<12.4f} {bin_time:<15.4f} {speedup:<10.2f}x")
    
    print("-" * 60)
    print("Note: Speedup shows how much faster/slower Fibonacci heap is")
    print("Values > 1.0 mean Fibonacci heap is faster\n")

def comprehensive_demo():
    """Run comprehensive demonstration of all features"""
    print("=" * 70)
    print("COMPREHENSIVE DEMO - Advanced Image Compression System")
    print("Using Huffman Coding with Fibonacci Heap Optimization")
    print("=" * 70)
    print()
    
    # Run all tests
    test_fibonacci_heap()
    test_huffman_basic()
    test_rle_compression()
    performance_comparison()
    test_image_compression()
    
    # Full analysis if sample images exist
    if os.path.exists("sample_images/rgb_pattern.png"):
        print("=== COMPREHENSIVE ANALYSIS ===\n")
        compressor = ImageCompressor()
        
        print("Running comprehensive analysis on RGB pattern image...")
        results = compressor.analyze_compression(
            "sample_images/rgb_pattern.png",
            "comprehensive_analysis"
        )
        
        print("\nBest performing configurations:")
        best_ratio = 0
        best_config = ""
        
        for config_name, result in results.items():
            ratio = result["compression_stats"].get("file_compression_ratio", 0)
            if ratio > best_ratio:
                best_ratio = ratio
                best_config = config_name
        
        print(f"Best compression ratio: {best_config} ({best_ratio:.2f}:1)")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("Check the generated files and analysis directories for detailed results.")
    print("=" * 70)

def interactive_menu():
    """Interactive menu for testing different features"""
    while True:
        print("\n" + "=" * 50)
        print("ADVANCED IMAGE COMPRESSION SYSTEM")
        print("=" * 50)
        print("1. Test Fibonacci Heap")
        print("2. Test Basic Huffman Compression")
        print("3. Test RLE Preprocessing")
        print("4. Test Image Compression")
        print("5. Performance Comparison")
        print("6. Run Comprehensive Demo")
        print("7. Create Sample Images")
        print("8. Exit")
        print("=" * 50)
        
        choice = input("Select option (1-8): ").strip()
        
        if choice == "1":
            test_fibonacci_heap()
        elif choice == "2":
            test_huffman_basic()
        elif choice == "3":
            test_rle_compression()
        elif choice == "4":
            test_image_compression()
        elif choice == "5":
            performance_comparison()
        elif choice == "6":
            comprehensive_demo()
        elif choice == "7":
            create_sample_images()
        elif choice == "8":
            print("Thank you for using the Advanced Image Compression System!")
            break
        else:
            print("Invalid choice. Please select 1-8.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            comprehensive_demo()
        elif sys.argv[1] == "--create-samples":
            create_sample_images()
        elif sys.argv[1] == "--quick-test":
            test_fibonacci_heap()
            test_huffman_basic()
            print("Quick test completed!")
        else:
            print("Usage: python main.py [--demo|--create-samples|--quick-test]")
    else:
        # Run interactive menu
        interactive_menu()
