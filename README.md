# JPEG Compression
## Data Compression Methods
### Peter Osztobanyi - Semester Project

## Encoding process:
1. Preprocessing
2. Color Space Conversion
3. Chroma Subsampling
4. Block Processing:
5. Apply Discrete Cosine Transform (DCT)
6. Quantization
7. Zigzag Scan
8. Run-Length Encoding (RLE)
9. Huffman Encoding
10. Bitstream writing

Then you have a compressed .jpec file

## Decoding process:
1. Bitstream Parsing (Huffman decode)
2. Block Reconstruction:
3. Inverse Zigzag Scan
4. Dequantization
5. Inverse DCT (IDCT)
6. Chroma Upsampling
7. Color Space Conversion
8. Postprocessing

Then you will have the decompressed .jpg file

## Project files:
1. Utils.py
2. jpeg_compressor.py
3. demo.py

### Start demo.py to encode and decode images. For this purpose there is an "input" folder to put the images. In demo.py please specify the source of the relevant image.
