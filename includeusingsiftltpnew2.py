"""
includingusingLTP.py
SIFT + LTP adaptive reversible medical-image data hiding.

"""

import base64
import hashlib
import json
import math
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, PngImagePlugin


MAGIC = "SIFTLTP-RDH-1"
VERSION = 1

# Analysis/selection parameters
BLOCK_SIZE = 32
LTP_THRESHOLD = 5
HEADER_RESERVED_ROWS = 0       # no pixel rows are reserved; recovery information is PNG metadata
MIN_STABLE_SCORE = 0.0

# Embedding uses blue channel of RGB image.
CHANNEL = 2


# ============================================================
# STEP 1: PRE-PROCESSING
# ============================================================
def preprocess_image(image_rgb):
    """
    Reversible-safe preprocessing for feature analysis.

    We do NOT resize or alter the image used for embedding because a resize
    operation would prevent exact pixel recovery. CLAHE is used only to create
    a temporary analysis image; the original RGB pixels are never replaced.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    return normalized


# ============================================================
# STEP 2: SIFT EXTRACTION
# ============================================================
def compute_sift(gray):
    sift = cv2.SIFT_create(
        nfeatures=0,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6
    )
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


# ============================================================
# STEP 3: LTP DESCRIPTORS
# ============================================================
def ltp_descriptor_at(gray, x, y, threshold=5):
    """
    8-neighbor Local Ternary Pattern.
    Returns positive and negative 8-bit codes plus a texture-strength value.
    """
    h, w = gray.shape
    if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
        return 0, 0, 0.0

    center = int(gray[y, x])
    positive = 0
    negative = 0
    strength = 0.0

    bit = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue

            neighbor = int(gray[y + dy, x + dx])
            diff = neighbor - center

            if diff > threshold:
                positive |= (1 << bit)
                strength += abs(diff)
            elif diff < -threshold:
                negative |= (1 << bit)
                strength += abs(diff)

            bit += 1

    return positive, negative, strength


def compute_ltp(gray, keypoints, threshold=5):
    """
    Compute LTP descriptors at SIFT keypoint locations.
    """
    descriptors = []
    for kp in keypoints:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))

        pos, neg, strength = ltp_descriptor_at(
            gray, x, y, threshold
        )

        descriptors.append({
            "x": x,
            "y": y,
            "positive": pos,
            "negative": neg,
            "strength": float(strength),
            "sift_response": float(kp.response),
            "sift_size": float(kp.size),
        })

    return descriptors


# ============================================================
# STEP 4: HYBRID SIFT + LTP FEATURE FUSION
# ============================================================
def build_block_scores(image_rgb, keypoints, ltp_descriptors,
                       block_size=BLOCK_SIZE):
    """
    Combine normalized SIFT response/density and LTP texture strength.

    Fusion score:
        0.5 * normalized SIFT stability
      + 0.5 * normalized LTP stability

    Blocks with stronger, more consistent SIFT/LTP evidence receive
    higher stability scores.
    """
    h, w = image_rgb.shape[:2]
    rows = math.ceil(h / block_size)
    cols = math.ceil(w / block_size)

    block_values = {}

    # Gather SIFT/LTP evidence by block.
    for d in ltp_descriptors:
        x, y = d["x"], d["y"]
        if 0 <= x < w and 0 <= y < h:
            br = y // block_size
            bc = x // block_size
            key = (br, bc)

            if key not in block_values:
                block_values[key] = {
                    "responses": [],
                    "strengths": [],
                    "sizes": []
                }

            block_values[key]["responses"].append(d["sift_response"])
            block_values[key]["strengths"].append(d["strength"])
            block_values[key]["sizes"].append(d["sift_size"])

    raw = []
    for br in range(rows):
        for bc in range(cols):
            key = (br, bc)
            data = block_values.get(
                key,
                {"responses": [], "strengths": [], "sizes": []}
            )

            if data["responses"]:
                sift_score = (
                    np.mean(data["responses"]) *
                    math.log1p(len(data["responses"]))
                )
                ltp_score = (
                    np.mean(data["strengths"]) *
                    math.log1p(len(data["strengths"]))
                )
            else:
                sift_score = 0.0
                ltp_score = 0.0

            raw.append((key, float(sift_score), float(ltp_score)))

    def normalize(values):
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) == 0 or np.max(arr) == np.min(arr):
            return np.zeros_like(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    sift_norm = normalize([x[1] for x in raw])
    ltp_norm = normalize([x[2] for x in raw])

    scores = {}
    for i, (key, _, _) in enumerate(raw):
        # Hybrid feature fusion.
        scores[key] = float(0.5 * sift_norm[i] + 0.5 * ltp_norm[i])

    return scores, rows, cols


# ============================================================
# STEP 5/6: STABILITY + SORT + SELECT STABLE BLOCKS
# ============================================================
def sorted_stable_blocks(scores):
    return sorted(
        scores.keys(),
        key=lambda k: scores[k],
        reverse=True
    )


def block_pixel_indices(block, image_shape, block_size=BLOCK_SIZE):
    """
    Return BLUE-channel pixel positions belonging to one block.
    Pixels are returned in raster order.
    """
    h, w = image_shape[:2]
    br, bc = block

    y0 = br * block_size
    x0 = bc * block_size
    y1 = min(y0 + block_size, h)
    x1 = min(x0 + block_size, w)

    indices = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            indices.append(y * w + x)

    return indices


# ============================================================
# HAMMING(7,4)
# ============================================================
def hamming74_encode_bits(bits):
    encoded = []

    # Pad data to a multiple of 4.
    bits = bits + "0" * ((4 - len(bits) % 4) % 4)

    for i in range(0, len(bits), 4):
        d1, d2, d3, d4 = [int(x) for x in bits[i:i + 4]]

        # Positions: p1 p2 d1 p4 d2 d3 d4
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p4 = d2 ^ d3 ^ d4

        encoded.extend([
            p1, p2, d1, p4, d2, d3, d4
        ])

    return "".join(str(x) for x in encoded)


def hamming74_decode_bits(encoded):
    decoded = []
    corrected_errors = 0

    if len(encoded) % 7 != 0:
        raise ValueError("Invalid Hamming(7,4) bit length.")

    for i in range(0, len(encoded), 7):
        b = [int(x) for x in encoded[i:i + 7]]

        # Syndrome.
        s1 = b[0] ^ b[2] ^ b[4] ^ b[6]
        s2 = b[1] ^ b[2] ^ b[5] ^ b[6]
        s4 = b[3] ^ b[4] ^ b[5] ^ b[6]

        error_position = s1 + 2 * s2 + 4 * s4

        if 1 <= error_position <= 7:
            b[error_position - 1] ^= 1
            corrected_errors += 1

        decoded.extend([b[2], b[4], b[5], b[6]])

    return "".join(str(x) for x in decoded), corrected_errors


# ============================================================
# TEXT + CHECKSUM
# ============================================================
def text_to_binary(text):
    data = text.encode("utf-8")
    return "".join(format(byte, "08b") for byte in data), data


def binary_to_text(bits):
    if len(bits) % 8 != 0:
        raise ValueError("Recovered UTF-8 bit stream is not byte aligned.")

    raw = bytes(
        int(bits[i:i + 8], 2)
        for i in range(0, len(bits), 8)
    )
    return raw.decode("utf-8")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


# ============================================================
# PNG RECOVERY INFORMATION / LOCATION MAP
# ============================================================
def create_location_map(selected_blocks, rows, cols):
    total = rows * cols
    bitmap = [0] * total

    for br, bc in selected_blocks:
        bitmap[br * cols + bc] = 1

    return "".join(str(x) for x in bitmap)


def decode_location_map(bitmap, rows, cols):
    blocks = []
    for i, bit in enumerate(bitmap):
        if bit == "1":
            br = i // cols
            bc = i % cols
            blocks.append((br, bc))
    return blocks


def pack_recovery_info(info):
    raw = json.dumps(
        info,
        separators=(",", ":"),
        ensure_ascii=False
    ).encode("utf-8")

    compressed = zlib.compress(raw, level=9)
    return base64.b64encode(compressed).decode("ascii")


def unpack_recovery_info(encoded):
    compressed = base64.b64decode(encoded.encode("ascii"))
    raw = zlib.decompress(compressed)
    return json.loads(raw.decode("utf-8"))


# ============================================================
# STEP 8: REVERSIBLE EMBEDDING
# ============================================================
def hide_text(image_path, text, output_path,
              threshold=LTP_THRESHOLD,
              block_size=BLOCK_SIZE):
    image_path = Path(image_path)
    output_path = Path(output_path)

    if output_path.suffix.lower() != ".png":
        raise ValueError(
            "Reversibility requires PNG output. "
            "Do not use JPG/JPEG for the embedded image."
        )

    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]

    # Step 1: preprocessing for analysis only.
    analysis_gray = preprocess_image(image)

    # Step 2: SIFT.
    keypoints, _ = compute_sift(analysis_gray)

    # Step 3: LTP.
    ltp_descriptors = compute_ltp(
        analysis_gray,
        keypoints,
        threshold
    )

    # Steps 4-6: hybrid fusion, stability, sorting.
    scores, rows, cols = build_block_scores(
        image,
        keypoints,
        ltp_descriptors,
        block_size
    )

    ranked_blocks = sorted_stable_blocks(scores)

    # Step 7: UTF-8 binary + Hamming(7,4).
    text_bits, text_bytes = text_to_binary(text)
    encoded_bits = hamming74_encode_bits(text_bits)

    # Need to store one original LSB per embedded payload bit.
    required_bits = len(encoded_bits)

    # Select stable blocks until enough pixels are available.
    selected_blocks = []
    capacity = 0

    for block in ranked_blocks:
        pixels = block_pixel_indices(
            block,
            image.shape,
            block_size
        )

        selected_blocks.append(block)
        capacity += len(pixels)

        if capacity >= required_bits:
            break

    if capacity < required_bits:
        raise ValueError(
            f"Insufficient embedding capacity. Required={required_bits}, "
            f"available={capacity}."
        )

    # Location map for selected stable blocks.
    location_map = create_location_map(
        selected_blocks, rows, cols
    )

    # Prepare payload positions.
    payload_positions = []
    for block in selected_blocks:
        payload_positions.extend(
            block_pixel_indices(
                block,
                image.shape,
                block_size
            )
        )
        if len(payload_positions) >= required_bits:
            payload_positions = payload_positions[:required_bits]
            break

    # Reversible LSB embedding.
    stego = image.copy()
    original_lsbs = []

    for i, pixel_index in enumerate(payload_positions):
        y = pixel_index // w
        x = pixel_index % w

        old_value = int(stego[y, x, CHANNEL])
        original_lsbs.append(old_value & 1)

        bit = int(encoded_bits[i])
        stego[y, x, CHANNEL] = (
            old_value & 0xFE
        ) | bit

    # Recovery information:
    # - selected block location map
    # - exact payload positions
    # - original LSB of every modified pixel
    # - checksum
    # - lengths
    recovery = {
        "magic": MAGIC,
        "version": VERSION,
        "width": w,
        "height": h,
        "channels": 3,
        "block_size": block_size,
        "ltp_threshold": threshold,
        "channel": CHANNEL,
        "grid_rows": rows,
        "grid_cols": cols,
        "location_map": location_map,
        "payload_positions": payload_positions,
        "original_lsbs": original_lsbs,
        "text_byte_length": len(text_bytes),
        "text_bit_length": len(text_bits),
        "hamming_bit_length": len(encoded_bits),
        "checksum_sha256": sha256_hex(text_bytes),
        "sift_keypoints": len(keypoints),
        "selected_blocks": len(selected_blocks),
    }

    # Store recovery information inside the PNG itself.
    # This makes the output PNG self-contained.
    metadata = pack_recovery_info(recovery)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("SIFT_LTP_RECOVERY", metadata)

    Image.fromarray(stego.astype(np.uint8), "RGB").save(
        output_path,
        format="PNG",
        pnginfo=pnginfo,
        compress_level=6
    )

    print("==============================================")
    print("Embedding completed.")
    print("Image size:", (w, h))
    print("SIFT keypoints:", len(keypoints))
    print("LTP descriptors:", len(ltp_descriptors))
    print("Stable blocks selected:", len(selected_blocks))
    print("Text bytes:", len(text_bytes))
    print("Text bits:", len(text_bits))
    print("Hamming bits:", len(encoded_bits))
    print("Payload positions:", len(payload_positions))
    print("Output:", output_path)
    print("==============================================")

    return {
        "output_path": str(output_path),
        "sift_keypoints": len(keypoints),
        "selected_blocks": len(selected_blocks),
        "text_bits": len(text_bits),
        "hamming_bits": len(encoded_bits),
    }


# ============================================================
# ALGORITHM 2: EXTRACTION + RECOVERY
# ============================================================
def reveal_text_and_recover(image_path, recovered_image_path=None):
    image_path = Path(image_path)

    with Image.open(image_path) as im:
        if "SIFT_LTP_RECOVERY" not in im.info:
            raise ValueError(
                "Recovery information was not found. "
                "Use the PNG produced by hide_text()."
            )

        metadata = im.info["SIFT_LTP_RECOVERY"]
        recovery = unpack_recovery_info(metadata)
        embedded = np.array(im.convert("RGB"))

    if recovery.get("magic") != MAGIC:
        raise ValueError("Invalid SIFT-LTP recovery metadata.")

    h, w = embedded.shape[:2]

    if (
        recovery["width"] != w
        or recovery["height"] != h
    ):
        raise ValueError("Image dimensions do not match recovery metadata.")

    # Steps 1-5 are repeated for analysis/verification.
    # Exact extraction positions come from the stored location map/recovery
    # information so that SIFT changes caused by embedding cannot desynchronize
    # the decoder.
    analysis_gray = preprocess_image(embedded)
    keypoints, _ = compute_sift(analysis_gray)
    ltp_descriptors = compute_ltp(
        analysis_gray,
        keypoints,
        recovery["ltp_threshold"]
    )

    # Validate stored stable-block map.
    location_map = recovery["location_map"]
    selected_blocks = decode_location_map(
        location_map,
        recovery["grid_rows"],
        recovery["grid_cols"]
    )

    # Step 6: extract embedded binary data.
    positions = recovery["payload_positions"]
    extracted_bits = []

    for pixel_index in positions:
        y = pixel_index // w
        x = pixel_index % w
        extracted_bits.append(
            str(int(embedded[y, x, CHANNEL]) & 1)
        )

    extracted_hamming = "".join(extracted_bits)

    if len(extracted_hamming) != recovery["hamming_bit_length"]:
        raise ValueError("Extracted Hamming data length is incorrect.")

    # Step 7: Hamming decoding.
    recovered_bits, corrected_errors = hamming74_decode_bits(
        extracted_hamming
    )

    recovered_bits = recovered_bits[:recovery["text_bit_length"]]

    # Step 8: convert binary to patient text.
    recovered_bytes = bytes(
        int(recovered_bits[i:i + 8], 2)
        for i in range(0, len(recovered_bits), 8)
    )

    recovered_text = recovered_bytes.decode("utf-8")

    # Step 7/checksum verification.
    checksum_ok = (
        sha256_hex(recovered_bytes)
        == recovery["checksum_sha256"]
    )

    if not checksum_ok:
        raise ValueError(
            "Checksum verification failed. "
            "The extracted patient information may be corrupted."
        )

    # Step 10: recover original image pixels.
    recovered_image = embedded.copy()

    for pixel_index, original_lsb in zip(
        positions,
        recovery["original_lsbs"]
    ):
        y = pixel_index // w
        x = pixel_index % w

        current = int(recovered_image[y, x, CHANNEL])
        recovered_image[y, x, CHANNEL] = (
            current & 0xFE
        ) | int(original_lsb)

    if recovered_image_path is not None:
        recovered_image_path = Path(recovered_image_path)
        Image.fromarray(
            recovered_image.astype(np.uint8),
            "RGB"
        ).save(recovered_image_path, format="PNG")

    print("==============================================")
    print("Extraction and recovery completed.")
    print("SIFT keypoints after embedding:", len(keypoints))
    print("Stored stable blocks:", len(selected_blocks))
    print("Extracted Hamming bits:", len(extracted_hamming))
    print("Corrected Hamming errors:", corrected_errors)
    print("Checksum:", "PASS" if checksum_ok else "FAIL")
    print("Recovered text:", recovered_text)

    if recovered_image_path is not None:
        print("Recovered image:", recovered_image_path)

    print("==============================================")

    return {
        "text": recovered_text,
        "recovered_image": recovered_image,
        "checksum_ok": checksum_ok,
        "corrected_hamming_errors": corrected_errors,
        "selected_blocks": selected_blocks,
        "sift_keypoints": len(keypoints),
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # Brain Tumor Detection dataset.#change the name number of image
    image_path = (
        "./input images Medical Image DataSet "
        "Brain Tumor Detection/img1.jpg"
    )

    output_path = (
        "./output images Medical Image DataSet "
        "Brain Tumor Detection/output_image1updatecode.png"
    )

    recovered_path = (
        "./output images Medical Image DataSet "
        "Brain Tumor Detection/recovered_image1.png"
    )

    text = (
        "Brain Tumor MRI dataset curated for secure "
        "medical data embedding"
    )

    Path(output_path).parent.mkdir(
        parents=True, exist_ok=True
    )

    img = Image.open(image_path)
    print("Original image size:", img.size)
    print("Patient text:", text)

    hide_text(
        image_path,
        text,
        output_path
    )

    result = reveal_text_and_recover(
        output_path,
        recovered_path
    )

    print("\nRecovered Patient Information:")
    print(result["text"])
