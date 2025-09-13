"""
Color Image Compressor using Huffman Coding with Fibonacci Heap
Supports both grayscale and RGB color images with channel-wise compression
"""

import os
import pickle
import time
from typing import Tuple, Dict, List, Optional
import numpy as np
from PIL import Image
import argparse

from huffman_compression import HuffmanCompressor

class ImageCompressor:
    """
    Advanced Image Compressor supporting:
    - Grayscale and RGB color images
    - Channel-wise compression for better efficiency
    - Fibonacci heap optimization
    - RLE preprocessing
    """
    
    def __init__(self, use_fibonacci_heap: bool = True, use_rle: bool = True):
        self.use_fibonacci_heap = use_fibonacci_heap
        self.use_rle = use_rle
        
    def load_image(self, filepath: str) -> np.ndarray:
        """Load image from file and return as numpy array"""
        try:
            img = Image.open(filepath)
            # Convert to RGB if needed (handles RGBA, P, L modes)
            if img.mode not in ['RGB', 'L']:
                if img.mode == 'RGBA':
                    # Convert RGBA to RGB by compositing on white background
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
            # Ensure values are in valid range
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            if len(image_array.shape) == 2:
                # Grayscale
                img = Image.fromarray(image_array, mode='L')
            else:
                # RGB
                img = Image.fromarray(image_array, mode='RGB')
            
            img.save(filepath)
        except Exception as e:
            raise ValueError(f"Failed to save image {filepath}: {str(e)}")
    
    def compress_image(self, image_path: str, output_path: str) -> Dict:
        """
        Compress image and save to custom format
        
        Args:
            image_path: Path to input image
            output_path: Path for compressed output (.huffimg)
            
        Returns:
            Compression statistics
        """
        print(f"Loading image: {image_path}")
        image_array = self.load_image(image_path)
        original_shape = image_array.shape
        
        print(f"Image shape: {original_shape}")
        print(f"Image mode: {'Grayscale' if len(original_shape) == 2 else 'RGB'}")
        
        start_time = time.time()
        
        if len(original_shape) == 2:
            # Grayscale image
            compressed_data = self._compress_grayscale(image_array)
        else:
            # RGB image
            compressed_data = self._compress_rgb(image_array)
        
        total_compression_time = time.time() - start_time
        
        # Prepare final data structure
        compressed_file_data = {
            "image_shape": original_shape,
            "compressed_channels": compressed_data,
            "compression_method": "huffman_fibonacci" if self.use_fibonacci_heap else "huffman_binary",
            "use_rle": self.use_rle,
            "total_compression_time": total_compression_time
        }
        
        # Save compressed data
        with open(output_path, 'wb') as f:
            pickle.dump(compressed_file_data, f)
        
        # Calculate statistics
        original_size = os.path.getsize(image_path)
        compressed_size = os.path.getsize(output_path)
        
        stats = {
            "original_file_size": original_size,
            "compressed_file_size": compressed_size,
            "file_compression_ratio": original_size / compressed_size,
            "file_space_saving": ((original_size - compressed_size) / original_size) * 100,
            "image_shape": original_shape,
            "total_compression_time": total_compression_time,
            "compression_method": compressed_file_data["compression_method"]
        }
        
        # Add per-channel statistics
        if "channel_stats" in compressed_data:
            stats["channel_stats"] = compressed_data["channel_stats"]
        
        return stats
    
    def decompress_image(self, compressed_path: str, output_path: str) -> Dict:
        """
        Decompress image from custom format
        
        Args:
            compressed_path: Path to compressed file (.huffimg)
            output_path: Path for decompressed image
            
        Returns:
            Decompression statistics
        """
        print(f"Loading compressed file: {compressed_path}")
        
        try:
            with open(compressed_path, 'rb') as f:
                compressed_file_data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load compressed file: {str(e)}")
        
        start_time = time.time()
        
        image_shape = compressed_file_data["image_shape"]
        compressed_channels = compressed_file_data["compressed_channels"]
        
        if len(image_shape) == 2:
            # Grayscale image
            reconstructed_image = self._decompress_grayscale(compressed_channels, image_shape)
        else:
            # RGB image
            reconstructed_image = self._decompress_rgb(compressed_channels, image_shape)
        
        decompression_time = time.time() - start_time
        
        # Save reconstructed image
        self.save_image(reconstructed_image, output_path)
        
        stats = {
            "decompression_time": decompression_time,
            "output_image_shape": reconstructed_image.shape,
            "compression_method": compressed_file_data.get("compression_method", "unknown")
        }
        
        return stats
    
    def _compress_grayscale(self, image_array: np.ndarray) -> Dict:
        """Compress grayscale image"""
        compressor = HuffmanCompressor(self.use_fibonacci_heap, self.use_rle)
        
        # Flatten image to 1D list
        pixel_data = image_array.flatten().tolist()
        
        # Compress
        encoded_bits, metadata = compressor.compress(pixel_data)
        
        # Calculate statistics
        stats = compressor.get_compression_stats(pixel_data, encoded_bits, metadata)
        
        return {
            "encoded_bits": encoded_bits,
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
            
            compressor = HuffmanCompressor(self.use_fibonacci_heap, self.use_rle)
            
            # Extract channel data
            channel_data = image_array[:, :, i].flatten().tolist()
            
            # Compress channel
            encoded_bits, metadata = compressor.compress(channel_data)
            
            # Store compressed data
            compressed_channels[channel_names[i]] = {
                "encoded_bits": encoded_bits,
                "metadata": metadata
            }
            
            # Calculate statistics
            stats = compressor.get_compression_stats(channel_data, encoded_bits, metadata)
            channel_stats[channel_names[i]] = stats
        
        return {
            "channels": compressed_channels,
            "channel_stats": channel_stats
        }
    
    def _decompress_grayscale(self, compressed_data: Dict, image_shape: Tuple) -> np.ndarray:
        """Decompress grayscale image"""
        compressor = HuffmanCompressor(self.use_fibonacci_heap, self.use_rle)
        
        encoded_bits = compressed_data["encoded_bits"]
        metadata = compressed_data["metadata"]
        
        # Decompress
        pixel_data = compressor.decompress(encoded_bits, metadata)
        
        # Reshape to original image shape
        reconstructed_image = np.array(pixel_data, dtype=np.uint8).reshape(image_shape)
        
        return reconstructed_image
    
    def _decompress_rgb(self, compressed_data: Dict, image_shape: Tuple) -> np.ndarray:
        """Decompress RGB image"""
        height, width, channels = image_shape
        reconstructed_channels = []
        
        channel_names = ['red', 'green', 'blue']
        
        for channel_name in channel_names:
            print(f"Decompressing {channel_name} channel...")
            
            compressor = HuffmanCompressor(self.use_fibonacci_heap, self.use_rle)
            
            channel_compressed = compressed_data["channels"][channel_name]
            encoded_bits = channel_compressed["encoded_bits"]
            metadata = channel_compressed["metadata"]
            
            # Decompress channel
            channel_data = compressor.decompress(encoded_bits, metadata)
            
            # Reshape channel data
            channel_array = np.array(channel_data, dtype=np.uint8).reshape((height, width))
            reconstructed_channels.append(channel_array)
        
        # Stack channels to form RGB image
        reconstructed_image = np.stack(reconstructed_channels, axis=2)
        
        return reconstructed_image
    
    def analyze_compression(self, image_path: str, output_dir: str = "compression_analysis") -> Dict:
        """
        Perform comprehensive compression analysis comparing different methods
        
        Args:
            image_path: Path to input image
            output_dir: Directory for analysis outputs
            
        Returns:
            Analysis results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("=== Compression Analysis ===")
        print(f"Image: {image_path}")
        
        results = {}
        
        # Test different configurations
        configurations = [
            {"use_fibonacci_heap": True, "use_rle": True, "name": "Fibonacci + RLE"},
            {"use_fibonacci_heap": True, "use_rle": False, "name": "Fibonacci Only"},
            {"use_fibonacci_heap": False, "use_rle": True, "name": "Binary Heap + RLE"},
            {"use_fibonacci_heap": False, "use_rle": False, "name": "Binary Heap Only"}
        ]
        
        for config in configurations:
            print(f"\nTesting: {config['name']}")
            
            # Create compressor with specific configuration
            compressor = ImageCompressor(
                use_fibonacci_heap=config["use_fibonacci_heap"],
                use_rle=config["use_rle"]
            )
            
            # Compress
            compressed_path = os.path.join(output_dir, f"compressed_{config['name'].replace(' ', '_').lower()}.huffimg")
            stats = compressor.compress_image(image_path, compressed_path)
            
            # Decompress to verify
            decompressed_path = os.path.join(output_dir, f"decompressed_{config['name'].replace(' ', '_').lower()}.png")
            decomp_stats = compressor.decompress_image(compressed_path, decompressed_path)
            
            # Verify reconstruction quality
            original_img = compressor.load_image(image_path)
            reconstructed_img = compressor.load_image(decompressed_path)
            
            # Calculate PSNR and MSE
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
        
        # Generate comparison report
        self._generate_analysis_report(results, output_dir)
        
        return results
    
    def _generate_analysis_report(self, results: Dict, output_dir: str):
        """Generate analysis report"""
        report_path = os.path.join(output_dir, "compression_analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=== COMPRESSION ANALYSIS REPORT ===\n\n")
            
            # Summary table
            f.write("COMPRESSION PERFORMANCE SUMMARY:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Method':<20} {'Ratio':<10} {'Space Saved':<12} {'Comp Time':<12} {'PSNR':<10}\n")
            f.write("-" * 80 + "\n")
            
            for method_name, result in results.items():
                comp_stats = result["compression_stats"]
                quality = result["quality_metrics"]
                
                ratio = f"{comp_stats.get('file_compression_ratio', 0):.2f}"
                space_saved = f"{comp_stats.get('file_space_saving', 0):.1f}%"
                comp_time = f"{comp_stats.get('total_compression_time', 0):.3f}s"
                psnr = f"{quality['psnr']:.1f}" if quality['psnr'] != float('inf') else "Perfect"
                
                f.write(f"{method_name:<20} {ratio:<10} {space_saved:<12} {comp_time:<12} {psnr:<10}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Detailed results
            for method_name, result in results.items():
                f.write(f"DETAILED RESULTS - {method_name}:\n")
                f.write("-" * 40 + "\n")
                
                comp_stats = result["compression_stats"]
                quality = result["quality_metrics"]
                
                f.write(f"File Compression Ratio: {comp_stats.get('file_compression_ratio', 0):.4f}\n")
                f.write(f"Space Saving: {comp_stats.get('file_space_saving', 0):.2f}%\n")
                f.write(f"Compression Time: {comp_stats.get('total_compression_time', 0):.4f} seconds\n")
                f.write(f"MSE: {quality['mse']:.4f}\n")
                f.write(f"PSNR: {quality['psnr']:.2f} dB\n")
                f.write(f"Perfect Reconstruction: {quality['perfect_reconstruction']}\n")
                
                if "channel_stats" in comp_stats:
                    f.write("\nPer-Channel Statistics:\n")
                    for channel, stats in comp_stats["channel_stats"].items():
                        f.write(f"  {channel.capitalize()}:\n")
                        f.write(f"    Compression Ratio: {stats.get('compression_ratio', 0):.4f}\n")
                        f.write(f"    Space Saving: {stats.get('space_saving_percentage', 0):.2f}%\n")
                
                f.write("\n" + "-" * 40 + "\n\n")
        
        print(f"\nAnalysis report saved to: {report_path}")

def main():
    """Command line interface for image compression"""
    parser = argparse.ArgumentParser(description="Advanced Image Compressor using Huffman Coding with Fibonacci Heap")
    
    parser.add_argument("command", choices=["compress", "decompress", "analyze"], 
                        help="Operation to perform")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--no-fibonacci", action="store_true", 
                        help="Use binary heap instead of Fibonacci heap")
    parser.add_argument("--no-rle", action="store_true", 
                        help="Disable Run Length Encoding preprocessing")
    parser.add_argument("--analysis-dir", default="compression_analysis",
                        help="Directory for analysis outputs (analyze command only)")
    
    args = parser.parse_args()
    
    # Create compressor
    compressor = ImageCompressor(
        use_fibonacci_heap=not args.no_fibonacci,
        use_rle=not args.no_rle
    )
    
    try:
        if args.command == "compress":
            print("Starting compression...")
            stats = compressor.compress_image(args.input, args.output)
            
            print("\n=== COMPRESSION RESULTS ===")
            print(f"Original file size: {stats['original_file_size']:,} bytes")
            print(f"Compressed file size: {stats['compressed_file_size']:,} bytes")
            print(f"Compression ratio: {stats['file_compression_ratio']:.2f}:1")
            print(f"Space saving: {stats['file_space_saving']:.1f}%")
            print(f"Compression time: {stats['total_compression_time']:.3f} seconds")
            print(f"Method: {stats['compression_method']}")
            
        elif args.command == "decompress":
            print("Starting decompression...")
            stats = compressor.decompress_image(args.input, args.output)
            
            print("\n=== DECOMPRESSION RESULTS ===")
            print(f"Decompression time: {stats['decompression_time']:.3f} seconds")
            print(f"Output image shape: {stats['output_image_shape']}")
            print(f"Method: {stats['compression_method']}")
            
        elif args.command == "analyze":
            print("Starting comprehensive analysis...")
            results = compressor.analyze_compression(args.input, args.analysis_dir)
            print("\nAnalysis complete! Check the analysis directory for detailed results.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())