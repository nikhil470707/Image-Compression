"""Image compressor utilities using Huffman coding.

Implements channel-wise compression for grayscale and RGB images with
optional DPCM and RLE preprocessing. Produces a simple serialized
container with encoded bytes and metadata.
"""

import os
import pickle
import time
from typing import Tuple, Dict
import numpy as np
from PIL import Image
import argparse

from huffman_compression import HuffmanCompressor

class ImageCompressor:
    # ImageCompressor: high-level routines for image compression.

    # The class offers methods to compress/decompress images and to run
    # basic analysis across different configurations.
    
    def __init__(self, use_fibonacci_heap: bool = True, 
                 use_rle: bool = False, 
                 use_diff_encoding: bool = True):
        self.use_fibonacci_heap = use_fibonacci_heap
        self.use_rle = use_rle
        self.use_diff_encoding = use_diff_encoding
        
    def load_image(self, filepath: str) -> np.ndarray:
        """Load image from file and return as numpy array"""
        try:
            img = Image.open(filepath)
            if img.mode not in ['RGB', 'L']:
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert('RGB')
            
            return np.array(img)
        except Exception as e:
            raise ValueError(f"Failed to load image {filepath}: {str(e)}")
    
    def save_image(self, image_array: np.ndarray, filepath: str):
        """Save numpy array as image file"""
        try:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            if len(image_array.shape) == 2:
                img = Image.fromarray(image_array, mode='L')
            else:
                img = Image.fromarray(image_array, mode='RGB')
            
            img.save(filepath)
        except Exception as e:
            raise ValueError(f"Failed to save image {filepath}: {str(e)}")
    
    def compress_image(self, image_path: str, output_path: str) -> Dict:
        """Compress image and save to .huffimg format"""
        print(f"Loading image: {image_path}")
        image_array = self.load_image(image_path)
        original_shape = image_array.shape
        
        print(f"Image shape: {original_shape}")
        print(f"Image mode: {'Grayscale' if len(original_shape) == 2 else 'RGB'}")
        
        start_time = time.time()
        
        if len(original_shape) == 2:
            compressed_data = self._compress_grayscale(image_array)
        else:
            compressed_data = self._compress_rgb(image_array)
        
        total_compression_time = time.time() - start_time
        
        compressed_file_data = {
            "image_shape": original_shape,
            "compressed_channels": compressed_data,
            "compression_method": "huffman_fibonacci" if self.use_fibonacci_heap else "huffman_binary",
            "use_rle": self.use_rle,
            "use_diff_encoding": self.use_diff_encoding,
            "total_compression_time": total_compression_time
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(compressed_file_data, f)
        
        # Calculate true raw pixel data size (what we actually compressed)
        raw_data_size = image_array.size * image_array.itemsize
        compressed_size = os.path.getsize(output_path)
        
        stats = {
            "original_file_size": raw_data_size,
            "compressed_file_size": compressed_size,
            "file_compression_ratio": raw_data_size / compressed_size if compressed_size > 0 else float('inf'),
            "file_space_saving": ((raw_data_size - compressed_size) / raw_data_size) * 100,
            "image_shape": original_shape,
            "total_compression_time": total_compression_time,
            "compression_method": compressed_file_data["compression_method"],
            "input_file_size": os.path.getsize(image_path)
        }
        
        if "channel_stats" in compressed_data:
            stats["channel_stats"] = compressed_data["channel_stats"]
        
        return stats
    
    def decompress_image(self, compressed_path: str, output_path: str) -> Dict:
        """Decompress image from .huffimg format"""
        print(f"Loading compressed file: {compressed_path}")
        
        try:
            with open(compressed_path, 'rb') as f:
                compressed_file_data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load compressed file: {str(e)}")
        
        start_time = time.time()
        
        image_shape = compressed_file_data["image_shape"]
        compressed_channels = compressed_file_data["compressed_channels"]
        
        use_fibonacci_heap = "fibonacci" in compressed_file_data.get("compression_method", "")
        use_rle = compressed_file_data.get("use_rle", False)
        use_diff_encoding = compressed_file_data.get("use_diff_encoding", False)
        
        if len(image_shape) == 2:
            reconstructed_image = self._decompress_grayscale(
                compressed_channels, image_shape, use_fibonacci_heap, use_rle, use_diff_encoding
            )
        else:
            reconstructed_image = self._decompress_rgb(
                compressed_channels, image_shape, use_fibonacci_heap, use_rle, use_diff_encoding
            )
        
        decompression_time = time.time() - start_time
        
        self.save_image(reconstructed_image, output_path)
        
        stats = {
            "decompression_time": decompression_time,
            "output_image_shape": reconstructed_image.shape,
            "compression_method": compressed_file_data.get("compression_method", "unknown"),
            "output_file_size": os.path.getsize(output_path)
        }
        
        return stats
    
    def _compress_grayscale(self, image_array: np.ndarray) -> Dict:
        """Compress grayscale image"""
        compressor = HuffmanCompressor(
            self.use_fibonacci_heap, self.use_rle, self.use_diff_encoding
        )
        
        pixel_data = image_array.flatten().tolist()
        encoded_bytes, metadata = compressor.compress(pixel_data)
        stats = compressor.get_compression_stats(pixel_data, encoded_bytes, metadata)
        
        return {
            "encoded_bytes": encoded_bytes,
            "metadata": metadata,
            "channel_stats": {"grayscale": stats}
        }
    
    def _compress_rgb(self, image_array: np.ndarray) -> Dict:
        """Compress RGB image using channel-wise compression"""
        height, width, channels = image_array.shape
        
        compressed_channels = {}
        channel_stats = {}
        channel_names = ['red', 'green', 'blue']
        
        for i in range(channels):
            print(f"Compressing {channel_names[i]} channel...")
            
            compressor = HuffmanCompressor(
                self.use_fibonacci_heap, self.use_rle, self.use_diff_encoding
            )
            
            channel_data = image_array[:, :, i].flatten().tolist()
            encoded_bytes, metadata = compressor.compress(channel_data)
            
            compressed_channels[channel_names[i]] = {
                "encoded_bytes": encoded_bytes,
                "metadata": metadata
            }
            
            stats = compressor.get_compression_stats(channel_data, encoded_bytes, metadata)
            channel_stats[channel_names[i]] = stats
        
        return {
            "channels": compressed_channels,
            "channel_stats": channel_stats
        }
    
    def _decompress_grayscale(self, compressed_data: Dict, image_shape: Tuple, 
                             use_fibonacci_heap: bool, use_rle: bool, 
                             use_diff_encoding: bool) -> np.ndarray:
        """Decompress grayscale image"""
        compressor = HuffmanCompressor(use_fibonacci_heap, use_rle, use_diff_encoding)
        
        encoded_bytes = compressed_data["encoded_bytes"]
        metadata = compressed_data["metadata"]
        
        pixel_data = compressor.decompress(encoded_bytes, metadata)
        reconstructed_image = np.array(pixel_data, dtype=np.uint8).reshape(image_shape)
        
        return reconstructed_image
    
    def _decompress_rgb(self, compressed_data: Dict, image_shape: Tuple, 
                       use_fibonacci_heap: bool, use_rle: bool, 
                       use_diff_encoding: bool) -> np.ndarray:
        """Decompress RGB image"""
        height, width, channels = image_shape
        reconstructed_channels = []
        
        channel_names = ['red', 'green', 'blue']
        
        for channel_name in channel_names:
            print(f"Decompressing {channel_name} channel...")
            
            compressor = HuffmanCompressor(use_fibonacci_heap, use_rle, use_diff_encoding)
            
            channel_compressed = compressed_data["channels"][channel_name]
            encoded_bytes = channel_compressed["encoded_bytes"]
            metadata = channel_compressed["metadata"]
            
            channel_data = compressor.decompress(encoded_bytes, metadata)
            channel_array = np.array(channel_data, dtype=np.uint8).reshape((height, width))
            reconstructed_channels.append(channel_array)
        
        reconstructed_image = np.stack(reconstructed_channels, axis=2)
        
        return reconstructed_image
    
    def analyze_compression(self, image_path: str, output_dir: str = "compression_analysis") -> Dict:
        """Comprehensive compression analysis comparing different methods"""
        os.makedirs(output_dir, exist_ok=True)
        print("=== COMPRESSION ANALYSIS ===")
        print(f"Image: {image_path}")
        
        results = {}
        
        configurations = [
            {"use_fibonacci_heap": True, "use_rle": False, "use_diff_encoding": True, "name": "Fib+DPCM"},
            {"use_fibonacci_heap": True, "use_rle": True, "use_diff_encoding": True, "name": "Fib+DPCM+RLE"},
            {"use_fibonacci_heap": True, "use_rle": False, "use_diff_encoding": False, "name": "Fib_Only"},
            {"use_fibonacci_heap": False, "use_rle": False, "use_diff_encoding": True, "name": "Binary+DPCM"}
        ]
        
        for config in configurations:
            print(f"\nTesting: {config['name']}")
            
            compressor = ImageCompressor(
                use_fibonacci_heap=config["use_fibonacci_heap"],
                use_rle=config["use_rle"],
                use_diff_encoding=config["use_diff_encoding"]
            )
            
            compressed_path = os.path.join(
                output_dir, f"compressed_{config['name'].lower()}.huffimg"
            )
            stats = compressor.compress_image(image_path, compressed_path)
            
            decompressed_path = os.path.join(
                output_dir, f"decompressed_{config['name'].lower()}.png"
            )
            decomp_stats = compressor.decompress_image(compressed_path, decompressed_path)
            
            original_img = compressor.load_image(image_path)
            reconstructed_img = compressor.load_image(decompressed_path)
            
            mse = np.mean((original_img.astype(float) - reconstructed_img.astype(float)) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            
            results[config["name"]] = {
                "compression_stats": stats,
                "decompression_stats": decomp_stats,
                "quality_metrics": {
                    "mse": mse,
                    "psnr": psnr,
                    "perfect_reconstruction": np.array_equal(original_img, reconstructed_img)
                }
            }
        
        self._generate_analysis_report(results, output_dir)
        
        return results
    
    def _generate_analysis_report(self, results: Dict, output_dir: str):
        """Generate detailed analysis report"""
        report_path = os.path.join(output_dir, "compression_analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("IMAGE COMPRESSION ANALYSIS REPORT\n")
            f.write("Huffman Coding with Fibonacci Heap + DPCM/RLE\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("PERFORMANCE SUMMARY (vs Raw Pixel Data):\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Method':<20} {'Raw Size (MB)':<15} {'Ratio':<10} {'Saved %':<12} {'Time (s)':<12} {'PSNR':<10}\n")
            f.write("-" * 80 + "\n")
            
            for method_name, result in results.items():
                comp_stats = result["compression_stats"]
                quality = result["quality_metrics"]
                
                raw_size_mb = comp_stats.get('original_file_size', 0) / (1024 * 1024)
                ratio = f"{comp_stats.get('file_compression_ratio', 0):.2f}"
                space_saved = f"{comp_stats.get('file_space_saving', 0):.1f}%"
                comp_time = f"{comp_stats.get('total_compression_time', 0):.3f}"
                psnr = "Perfect" if quality['psnr'] == float('inf') else f"{quality['psnr']:.1f} dB"
                
                f.write(f"{method_name:<20} {raw_size_mb:<15.2f} {ratio:<10} {space_saved:<12} {comp_time:<12} {psnr:<10}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            f.write("DETAILED RESULTS:\n\n")
            for method_name, result in results.items():
                f.write(f"{method_name}:\n")
                f.write("-" * 40 + "\n")
                
                comp_stats = result["compression_stats"]
                quality = result["quality_metrics"]
                
                f.write(f"  Raw Pixel Data: {comp_stats.get('original_file_size', 0):,} bytes\n")
                f.write(f"  Compressed File: {comp_stats.get('compressed_file_size', 0):,} bytes\n")
                f.write(f"  Compression Ratio: {comp_stats.get('file_compression_ratio', 0):.4f}:1\n")
                f.write(f"  Space Saving: {comp_stats.get('file_space_saving', 0):.2f}%\n")
                f.write(f"  Compression Time: {comp_stats.get('total_compression_time', 0):.4f}s\n")
                f.write(f"  MSE: {quality['mse']:.4f}\n")
                f.write(f"  PSNR: {quality['psnr']:.2f} dB\n")
                f.write(f"  Lossless: {quality['perfect_reconstruction']}\n")
                
                if "channel_stats" in comp_stats:
                    f.write("\n  Per-Channel Statistics:\n")
                    for channel, stats in comp_stats["channel_stats"].items():
                        f.write(f"    {channel.capitalize()}:\n")
                        f.write(f"      Ratio: {stats.get('compression_ratio', 0):.4f}:1\n")
                        f.write(f"      Saved: {stats.get('space_saving_percentage', 0):.2f}%\n")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("NOTE: Compression ratios are calculated against raw pixel data,\n")
            f.write("not against PNG/JPEG files which already use optimized compression.\n")
            f.write("=" * 80 + "\n")
        
        print(f"\nAnalysis report saved to: {report_path}")

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Image Compressor using Huffman Coding with Fibonacci Heap"
    )
    
    parser.add_argument(
        "command", 
        choices=["compress", "decompress", "analyze"], 
        help="Operation: compress, decompress, or analyze"
    )
    parser.add_argument("input", help="Input file path")
    parser.add_argument("output", help="Output file path")
    parser.add_argument(
        "--no-fibonacci", 
        action="store_true", 
        help="Use binary heap instead of Fibonacci heap"
    )
    parser.add_argument(
        "--rle", 
        action="store_true", 
        help="Enable Run Length Encoding (for images with repeated patterns)"
    )
    parser.add_argument(
        "--no-diff", 
        action="store_true",
        help="Disable Differential Encoding (DPCM)"
    )
    parser.add_argument(
        "--analysis-dir", 
        default="compression_analysis",
        help="Directory for analysis outputs (analyze command only)"
    )
    
    args = parser.parse_args()
    
    compressor = ImageCompressor(
        use_fibonacci_heap=not args.no_fibonacci,
        use_rle=args.rle,
        use_diff_encoding=not args.no_diff
    )
    
    print(f"Configuration:")
    print(f"  Fibonacci Heap: {compressor.use_fibonacci_heap}")
    print(f"  DPCM: {compressor.use_diff_encoding}")
    print(f"  RLE: {compressor.use_rle}")
    
    try:
        if args.command == "compress":
            print("\nStarting compression...")
            stats = compressor.compress_image(args.input, args.output)
            
            print("\n=== COMPRESSION RESULTS ===")
            print(f"Raw Pixel Data Size: {stats['original_file_size']:,} bytes ({stats['original_file_size'] / (1024*1024):.2f} MB)")
            print(f"Input File Size: {stats['input_file_size']:,} bytes ({stats['input_file_size'] / (1024*1024):.2f} MB)")
            print(f"Compressed Size: {stats['compressed_file_size']:,} bytes ({stats['compressed_file_size'] / (1024*1024):.2f} MB)")
            print("-" * 40)
            print(f"Compression Ratio (vs raw): {stats['file_compression_ratio']:.2f}:1")
            print(f"Space Saving: {stats['file_space_saving']:.1f}%")
            print(f"Time: {stats['total_compression_time']:.3f}s")
            print(f"Method: {stats['compression_method']}")
            
        elif args.command == "decompress":
            print("\nStarting decompression...")
            stats = compressor.decompress_image(args.input, args.output)
            
            print("\n=== DECOMPRESSION RESULTS ===")
            print(f"Time: {stats['decompression_time']:.3f}s")
            print(f"Output Shape: {stats['output_image_shape']}")
            print(f"Output Size: {stats['output_file_size']:,} bytes")
            print(f"Method: {stats['compression_method']}")
            
        elif args.command == "analyze":
            print("\nStarting comprehensive analysis...")
            compressor.analyze_compression(args.input, args.analysis_dir)
            print("\nAnalysis complete! Check the analysis directory for detailed results.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())