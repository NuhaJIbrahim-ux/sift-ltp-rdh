"""
includusingsiftltpemeasuresnew2.py

Measures for the SIFT + LTP reversible medical image data hiding system.

It compares:
1. Original image vs embedded/stego image
2. Original image vs recovered image

The second comparison is the reversibility test.
For exact recovery:
    MSE = 0
    PSNR = infinity
    SSIM = 1
    different pixels = 0
"""

from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def load_rgb(path):
    return np.array(
        Image.open(path).convert("RGB"),
        dtype=np.uint8
    )


def mse(original, test):
    a = original.astype(np.float64)
    b = test.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(original, test):
    value = mse(original, test)

    if value == 0:
        return float("inf")

    return float(
        10.0 * np.log10((255.0 ** 2) / value)
    )


def calculate_ssim(original, test):
    return float(
        ssim(
            original,
            test,
            multichannel=True,
            data_range=255
        )
    )


def ncc(original, test):
    a = original.astype(np.float64).flatten()
    b = test.astype(np.float64).flatten()

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    numerator = np.sum(
        (a - a_mean) * (b - b_mean)
    )

    denominator = np.sqrt(
        np.sum((a - a_mean) ** 2) *
        np.sum((b - b_mean) ** 2)
    )

    if denominator == 0:
        return 1.0 if np.array_equal(original, test) else 0.0

    return float(numerator / denominator)


def mae(original, test):
    return float(
        np.mean(
            np.abs(
                original.astype(np.float64)
                - test.astype(np.float64)
            )
        )
    )


def changed_pixels(original, test):
    diff = np.any(original != test, axis=2)
    return int(np.sum(diff))


def maximum_absolute_error(original, test):
    return int(
        np.max(
            np.abs(
                original.astype(np.int16)
                - test.astype(np.int16)
            )
        )
    )


def embedding_capacity(text, stego_size):
    """
    Payload capacity in bits per image pixel.
    This is the actual Hamming-encoded payload length.
    """
    text_bytes = len(text.encode("utf-8"))
    original_bits = text_bytes * 8
    padded_bits = original_bits + (
        (4 - original_bits % 4) % 4
    )
    hamming_bits = (padded_bits // 4) * 7

    pixels = stego_size[0] * stego_size[1]

    return {
        "text_bytes": text_bytes,
        "original_payload_bits": original_bits,
        "hamming_payload_bits": hamming_bits,
        "bits_per_pixel": hamming_bits / pixels,
        "bits_per_channel_pixel": hamming_bits / pixels
    }


def print_measures(name, original, test):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("MSE:", mse(original, test))
    print("PSNR:", psnr(original, test))
    print("SSIM:", calculate_ssim(original, test))
    print("MAE:", mae(original, test))
    print("Changed pixels:", changed_pixels(original, test))
    print(
        "Maximum absolute pixel error:",
        maximum_absolute_error(original, test)
    )


def measure_files(original_path, stego_path, recovered_path):
    original = load_rgb(original_path)
    stego = load_rgb(stego_path)
    recovered = load_rgb(recovered_path)

    if original.shape != stego.shape:
        raise ValueError("Original and stego images have different sizes.")

    if original.shape != recovered.shape:
        raise ValueError(
            "Original and recovered images have different sizes."
        )

    print("\nIMAGE DIMENSIONS:", original.shape)

    # Measures of imperceptibility.
    print_measures(
        "1. Original Image vs Embedded (Stego) Image",
        original,
        stego
    )

    # Measures of exact reversibility.
    print_measures(
        "2. Original Image vs Recovered Image",
        original,
        recovered
    )

    exact = np.array_equal(original, recovered)

    print("\n" + "=" * 60)
    print("REVERSIBILITY TEST")
    print("=" * 60)
    print("Exact pixel recovery:", "PASS" if exact else "FAIL")

    if exact:
        print("All original pixels were recovered exactly.")
        print("MSE = 0")
        print("PSNR = infinity")
        print("SSIM = 1")
    else:
        print(
            "Some pixels differ. Check the recovery information "
            "and make sure the stego file is PNG."
        )

    return {
        "stego_mse": mse(original, stego),
        "stego_psnr": psnr(original, stego),
        "stego_ssim": calculate_ssim(original, stego),
        "recovery_mse": mse(original, recovered),
        "recovery_psnr": psnr(original, recovered),
        "recovery_ssim": calculate_ssim(original, recovered),
        "exact_recovery": exact,
        "changed_stego_pixels": changed_pixels(original, stego),
        "changed_recovered_pixels": changed_pixels(original, recovered)
    }


if __name__ == "__main__":

    original_path = (
        "./input images Medical Image DataSet "
        "Brain Tumor Detection/img10.jpg"
    )

    stego_path = (
        "./output images Medical Image DataSet "
        "Brain Tumor Detection/output_image10updatecode.png"
    )

    recovered_path = (
        "./output images Medical Image DataSet "
        "Brain Tumor Detection/recovered_image10.png"
    )

    results = measure_files(
        original_path,
        stego_path,
        recovered_path
    )

    print("\nFinal measurement dictionary:")
    for key, value in results.items():
        print(f"{key}: {value}")
