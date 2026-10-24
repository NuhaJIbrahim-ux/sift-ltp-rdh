# SIFT–LTP Adaptive Reversible Medical Image Data Hiding
Code, input images, output images and results for the paper:
"An adaptive reversible medical image data hiding framework using hybrid SIFT–LTP feature fusion for secure patient information embedding" 
 
## Folders
`code/` – `includeusingsiftltpnew2.py` (embedding, extraction and recovery); 
`includeusingsiftltpmeasuresnew2.py` (MSE, PSNR, SSIM, NCC and other measures).
- `input_images/` – the ten original MRI images (img1.jpg–img10.jpg).
- `output_images/` – embedded (stego) PNG images and recovered PNG images.
- `results/` 
Table 2 (embedding metrics), 
Table 3 (recovered-image metrics), 
Table 4 (varying payload sizes) as .xlsx files.

## Requirements

Python, OpenCV (`opencv-python`).
Install with: `pip install opencv-python `

## How to run
In `code/includeusingsiftltpnew2.py`, set `image_path`, `output_path`, `recovered_path` and `text` in the `__main__` block, then run:
   `python includeusingsiftltpnew2.py`
   
## Experimental environment
All timing experiments were run on a computer with: Core i7 processor, 32 GB RAM, NVIDIA GeForce RTX 2060 GPU, Windows 10 (64-bit). The code runs on the CPU only.

## Data and licence
- Input images: "Medical Image DataSet: Brain Tumor Detection" by Parisa Karimi Darabi, Roboflow Universe / Kaggle, licensed under CC BY 4.0. The images were used without modification, except that embedded and recovered versions are derived from them.


