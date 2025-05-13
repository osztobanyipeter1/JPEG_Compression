from jpeg_compressor import JPEGCompressor
from utils import show_images
from PIL import Image
import numpy as np
import os
import pickle
import imageio

def get_file_size(path):
    """Return file size in bytes"""
    return os.path.getsize(path)


def save_compressed_data(compressed_data, output_path):
    # Convert to a serializable format
    serializable_data = {
        'compressed_blocks': {},
        'original_height': compressed_data['original_height'],
        'original_width': compressed_data['original_width'],
        'quality': compressed_data['quality']
    }
    
    for channel in ['y', 'cb', 'cr']:
        channel_data = compressed_data['compressed_blocks'][channel]
        serializable_data['compressed_blocks'][channel] = {
            'huffman': {
                'dc': channel_data['huffman']['dc'],
                'ac': channel_data['huffman']['ac']
            },
            'encoded_blocks': [
                (dc, [(run, value) for run, value in ac]) 
                for dc, ac in channel_data['encoded_blocks']
            ],
            'original_height': channel_data['original_height'],
            'original_width': channel_data['original_width']
        }
    
    # Save as binary data
    with open(output_path, 'wb') as f:
        pickle.dump(serializable_data, f)
    
    return os.path.getsize(output_path)

def main():
    # Initialize compressor with quality (1-100)
    quality = 1
    compressor = JPEGCompressor(quality=quality)
    
    # Path to your test image
    image_path = "input/blackandwhite4000x2800.jpg"
    compressed_path = "compressed/compressed_blackandwhitesbig(q=1).jpec"
    
    # Get original file size
    original_size = get_file_size(image_path)
    
    # Load original image
    original_img = Image.open(image_path)
    original_array = np.array(original_img)
    
    print(f"\nOriginal image size: {original_size:,} bytes")
    print(f"Image dimensions: {original_array.shape[1]}x{original_array.shape[0]} pixels")
    
    # Compress the image
    print("\nStarting compression...")
    compressed_data = compressor.compress(image_path, compressed_path)
    
    # Save compressed data and get compressed size
    compressed_size = get_file_size(compressed_path)
    
    print("\nCompression results:")
    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {original_size/compressed_size:.2f}:1")
    print(f"Space savings: {(1 - compressed_size/original_size)*100:.2f}%")
    
    # Decompress the image
    print("\nStarting decompression...")
    decompressed_array = compressor.decompress(compressed_data)
    

    # Save the reconstructed image
    output_path = "decompressed/decompressed_blackandwhitesbig(q=1).jpg"
    imageio.imwrite(output_path, decompressed_array.astype('uint8'))
    print(f"\nDecompressed image saved to {output_path}")
    print(f"\nDecompressed image size: {get_file_size(output_path):,} bytes")

    # Calculate PSNR
    psnr = compressor.calculate_psnr(original_array, decompressed_array)
    print(f"\nQuality metrics:")
    print(f"PSNR: {psnr:.2f} dB")

    
    
    # Show the images
    show_images(original_array, decompressed_array, 
               f"Original ({original_size:,} bytes)", 
               f"Decompressed (Quality: {compressed_size:,} bytes)")

if __name__ == "__main__":
    main()