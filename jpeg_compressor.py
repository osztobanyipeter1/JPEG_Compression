import numpy as np
import imageio
from PIL import Image
from scipy.fftpack import dct, idct
import heapq
from collections import defaultdict
from utils import *

class JPEGCompressor:
    def __init__(self, quality=50):
        self.quality = quality
        self.initialize_quantization_tables()
        
    def initialize_quantization_tables(self):
        # Standard JPEG luminance quantization table
        self.luminance_quant_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])
        
        # Standard JPEG chrominance quantization table
        self.chrominance_quant_table = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ])
        
        # Adjust quantization tables based on quality
        if self.quality < 50:
            scale = 5000 / self.quality #for more aggressice compression
        else:
            scale = 200 - 2 * self.quality #for less aggressive compression
            
        self.luminance_quant_table = np.clip(np.round(self.luminance_quant_table * scale / 100), 1, 255) #make sure values are between 1 and 255
        self.chrominance_quant_table = np.clip(np.round(self.chrominance_quant_table * scale / 100), 1, 255)
    
    # Convert RGB to YCbCr color space
    def rgb_to_ycbcr(self, image): 
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        
        return np.stack([y, cb, cr], axis=2)
    
    # Convert YCbCr back to RGB
    def ycbcr_to_rgb(self, image):    
        y, cb, cr = image[:,:,0], image[:,:,1], image[:,:,2]
        
        r = y + 1.402 * (cr - 128)
        g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
        b = y + 1.772 * (cb - 128)
        
        return np.clip(np.stack([r, g, b], axis=2), 0, 255)
    
    def chroma_subsampling(self, image):
        y = image[:, :, 0]
        cb = image[:, :, 1]
        cr = image[:, :, 2]

        height, width = cb.shape
        if height % 2 != 0:
            cb = np.pad(cb, ((0, 1), (0, 0)), mode='constant')
            cr = np.pad(cr, ((0, 1), (0, 0)), mode='constant')
        if width % 2 != 0:
            cb = np.pad(cb, ((0, 0), (0, 1)), mode='constant')
            cr = np.pad(cr, ((0, 0), (0, 1)), mode='constant')

        # Subsample chroma (4:2:0 format)
        cb = cb[::2, ::2] 
        cr = cr[::2, ::2]

        return y, cb, cr


    def chroma_upsampling(self, y, cb, cr, original_shape): # Upsample chroma channels to match original size
        height, width = original_shape
        cb_upsampled = np.zeros((height, width))
        cr_upsampled = np.zeros((height, width))

        # Fill in the subsampled values
        cb_upsampled[::2, ::2] = cb
        cr_upsampled[::2, ::2] = cr

        # Interpolate rows
        for i in range(1, height, 2):
            cb_upsampled[i, ::2] = (cb_upsampled[i - 1, ::2] + cb_upsampled[min(i + 1, height - 1), ::2]) / 2
            cr_upsampled[i, ::2] = (cr_upsampled[i - 1, ::2] + cr_upsampled[min(i + 1, height - 1), ::2]) / 2

        # Interpolate columns
        for j in range(1, width, 2):
            cb_upsampled[:, j] = (cb_upsampled[:, j - 1] + cb_upsampled[:, min(j + 1, width - 1)]) / 2
            cr_upsampled[:, j] = (cr_upsampled[:, j - 1] + cr_upsampled[:, min(j + 1, width - 1)]) / 2

        return np.stack([y, cb_upsampled, cr_upsampled], axis=2)

        
    def process_image(self, image_path): # Load and preprocess the image
        img = Image.open(image_path)
        if img.mode != 'RGB': # Convert to RGB if not already
            img = img.convert('RGB')
        
        img_array = np.array(img)
        height, width = img_array.shape[:2]

        # Pad image to be divisible by 16
        pad_height = (16 - height % 16) % 16
        pad_width = (16 - width % 16) % 16
        padded_img = np.pad(img_array, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        
        return padded_img, height, width


    
    def pad_channel(self, channel): # Pad channel to be divisible by 8
        height, width = channel.shape
        pad_height = (8 - height % 8) % 8
        pad_width = (8 - width % 8) % 8
        return np.pad(channel, ((0, pad_height), (0, pad_width)), mode='constant')

    def compress(self, image_path, output_path):
        # Step 1: Preprocessing
        padded_img, original_height, original_width = self.process_image(image_path)
        original_shape = (original_height, original_width)

        # Step 2: Color space conversion
        ycbcr_img = self.rgb_to_ycbcr(padded_img)

        # Step 3: Chroma subsampling
        y, cb, cr = self.chroma_subsampling(ycbcr_img)

        # Pad all channels to multiples of 8
        y = self.pad_channel(y)
        cb = self.pad_channel(cb)
        cr = self.pad_channel(cr)

        # Step 4: Process each channel in 8x8 blocks
        with open(output_path, 'wb') as f:
            writer = BitWriter(f)
            
            # Write header (width, height, quality)
            writer.write_bits(format(original_width, '016b'))  # 16 bits for width
            writer.write_bits(format(original_height, '016b')) # 16 bits for height
            writer.write_bits(format(self.quality, '08b'))     # 8 bits for quality

            # Process each channel
            for channel, quant_table in [(y, self.luminance_quant_table),
                                    (cb, self.chrominance_quant_table),
                                    (cr, self.chrominance_quant_table)]:
                height, width = channel.shape
                blocks = []
                
                # Process blocks
                prev_dc = 0
                for i in range(0, height, 8):
                    for j in range(0, width, 8):
                        block = channel[i:i+8, j:j+8]
                        block = block - 128
                        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                        quantized_block = np.round(dct_block / quant_table)
                        zigzag_block = zigzag_scan(quantized_block)
                        
                        # DC coefficient
                        dc = zigzag_block[0]
                        dc_diff = dc - prev_dc
                        prev_dc = dc
                        
                        # AC coefficients (RLE)
                        ac_coeffs = zigzag_block[1:]
                        rle_ac = self.run_length_encode(ac_coeffs)
                        
                        # Encode DC, Huffman
                        dc_bits = HuffmanEncoder.encode_dc(dc_diff)
                        writer.write_bits(dc_bits)
                        
                        # Encode AC, Huffman
                        for run, value in rle_ac:
                            ac_bits = HuffmanEncoder.encode_ac(run, value)
                            writer.write_bits(ac_bits)
                
            writer.flush()

        print(f"Compression complete. Output saved to {output_path}") 
        return {
            'original_height': original_height,
            'original_width': original_width,
            'padded_height': y.shape[0],
            'padded_width': y.shape[1],
            'quality': self.quality,
            'compressed_blocks': {
                'y': self.process_channel(y, self.luminance_quant_table),
                'cb': self.process_channel(cb, self.chrominance_quant_table),
                'cr': self.process_channel(cr, self.chrominance_quant_table)
            }
        }


    def encode_dc_ac(self, blocks):
        """Entropy encoding for DC and AC coefficients"""
        encoded_blocks = []
        prev_dc = 0  # DC prediction starts at 0
        
        for block in blocks:
            # DC coefficient differential encoding
            dc = block[0]
            diff_dc = dc - prev_dc
            prev_dc = dc
            
            # AC coefficients run-length encoding
            ac_coeffs = block[1:]
            rle_ac = self.run_length_encode(ac_coeffs)
            
            encoded_blocks.append((diff_dc, rle_ac))
        
        return encoded_blocks

    def run_length_encode(self, ac_coeffs): #RLE for AC coefficients
        """Run-length encoding for AC coefficients in zigzag order"""
        rle = []
        zero_run = 0
        
        for coeff in ac_coeffs:
            if coeff == 0:
                zero_run += 1
            else:
                # Encode (zero_run, value) pair
                rle.append((zero_run, coeff))
                zero_run = 0
        
        # End of block marker
        if zero_run > 0:
            rle.append((0, 0))  # EOB marker
        
        return rle

    def decode_dc_ac(self, encoded_blocks): 
        """Decode entropy encoded DC and AC coefficients"""
        blocks = []
        prev_dc = 0
        
        for diff_dc, rle_ac in encoded_blocks:
            # Reconstruct DC coefficient
            dc = prev_dc + diff_dc
            prev_dc = dc
            
            # Reconstruct AC coefficients
            ac_coeffs = self.run_length_decode(rle_ac)
            
            # Reconstruct full block
            block = [dc] + ac_coeffs
            blocks.append(block)
        
        return blocks

    def run_length_decode(self, rle_data):
        """Decode run-length encoded AC coefficients"""
        ac_coeffs = []
        for zero_run, value in rle_data:
            if zero_run == 0 and value == 0:  # EOB marker
                ac_coeffs.extend([0] * (64 - len(ac_coeffs) - 1))
                break
            ac_coeffs.extend([0] * zero_run)
            ac_coeffs.append(value)
        
        # Pad with zeros if needed (shouldn't be necessary with EOB)
        ac_coeffs.extend([0] * (63 - len(ac_coeffs)))
        return ac_coeffs

    def process_channel(self, channel, quant_table):
        height, width = channel.shape
        blocks = []
        
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = channel[i:i+8, j:j+8]
                block = block - 128
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                quantized_block = np.round(dct_block / quant_table)
                zigzag_block = zigzag_scan(quantized_block)
                
                blocks.append(zigzag_block)
        
        # Entropy encoding
        encoded_blocks = self.encode_dc_ac(blocks)
        
        return {
            'encoded_blocks': encoded_blocks,
            'original_height': height,
            'original_width': width
        }

    def reconstruct_channel(self, channel_data, quant_table):
        encoded_blocks = channel_data['encoded_blocks']
        height = channel_data['original_height']
        width = channel_data['original_width']
        
        # Entropy decoding
        blocks = self.decode_dc_ac(encoded_blocks)
        
        # Rest of the reconstruction
        reconstructed = np.zeros((height, width))
        block_index = 0
        
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                zigzag_block = blocks[block_index]
                block_index += 1
                
                quantized_block = inverse_zigzag(zigzag_block, 8, 8)
                dct_block = quantized_block * quant_table
                idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                reconstructed_block = idct_block + 128
                reconstructed_block = np.clip(reconstructed_block, 0, 255)
                reconstructed[i:i+8, j:j+8] = reconstructed_block
        
        return reconstructed
    
    def decompress(self, compressed_data):
        compressed_blocks = compressed_data['compressed_blocks']
        original_height = compressed_data['original_height']
        original_width = compressed_data['original_width']
        
        # Reconstruct each channel
        y = self.reconstruct_channel(compressed_blocks['y'], self.luminance_quant_table)
        cb = self.reconstruct_channel(compressed_blocks['cb'], self.chrominance_quant_table)
        cr = self.reconstruct_channel(compressed_blocks['cr'], self.chrominance_quant_table)
        
        # Chroma upsampling
        padded_height = compressed_data['padded_height']
        padded_width = compressed_data['padded_width']
        ycbcr_img = self.chroma_upsampling(y, cb, cr, (padded_height, padded_width))

        # Convert back to RGB
        rgb_img = self.ycbcr_to_rgb(ycbcr_img)
        
        # Crop to original dimensions
        rgb_img = rgb_img[:original_height, :original_width]

        return rgb_img.astype(np.uint8)

    def save_reconstructed_image(self, decompressed_image, output_path="reconstructed_image.png"):
        imageio.imwrite(output_path, decompressed_image)
        print(f"Reconstructed image saved to {output_path}")

    def calculate_psnr(self, original, compressed):
        # Calculate Peak Signal-to-Noise Ratio
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

class HuffmanEncoder:
    @staticmethod
    def encode_dc(dc_diff):
        size = HuffmanEncoder.get_magnitude(dc_diff)
        code = HuffmanEncoder.DC_LUM.get(size, '111111110')  # fallback if out of range
        additional_bits = HuffmanEncoder.int_to_bin(dc_diff, size)
        return code + additional_bits

    @staticmethod
    def encode_ac(run, value):
        size = HuffmanEncoder.get_magnitude(value)
        key = (run, size)
        code = HuffmanEncoder.AC_LUM.get(key, '11111111001')  # fallback for rare values
        additional_bits = HuffmanEncoder.int_to_bin(value, size)
        return code + additional_bits

    @staticmethod
    def get_magnitude(value):
        abs_val = abs(int(value))
        if abs_val == 0:
            return 0
        return abs_val.bit_length()

    @staticmethod
    def int_to_bin(value, size):
        value = int(value)
        if size == 0:
            return ''
        if value >= 0:
            return format(value, f'0{size}b')
        else:
            max_val = (1 << size) - 1
            return format(value + (1 << size), f'0{size}b')  # 2's complement

    # Simplified Huffman tables
    DC_LUM = {
        0: '00', 1: '010', 2: '011', 3: '100',
        4: '101', 5: '110', 6: '1110', 7: '11110',
        8: '111110', 9: '1111110', 10: '11111110', 11: '111111110'
    }

    AC_LUM = {
        (0, 0): '1010',  # EOB
        (0, 1): '00', (0, 2): '01', (0, 3): '100',
        (1, 1): '1011', (1, 2): '1100',
        (2, 1): '11010', (2, 2): '11011',
        (3, 1): '11100', (4, 1): '111010', (5, 1): '111011',
        (6, 1): '1111000', (15, 0): '11111111001'  # ZRL
    }


class BitWriter:
    def __init__(self, file):
        self.file = file
        self.buffer = 0
        self.buffer_length = 0

    def write_bits(self, bits):
        for bit in bits:
            self.buffer = (self.buffer << 1) | int(bit)
            self.buffer_length += 1
            if self.buffer_length == 8:
                self.flush_byte()

    def flush_byte(self):
        self.file.write(bytes([self.buffer]))
        self.buffer = 0
        self.buffer_length = 0

    def flush(self):
        if self.buffer_length > 0:
            self.buffer = self.buffer << (8 - self.buffer_length)
            self.flush_byte()
