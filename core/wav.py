"""WAV container helpers.

I. The problem this solves

1. The Qwen-TTS HTTP endpoint streams audio, so the WAV it returns carries a
   *placeholder* header: the RIFF and ``data`` chunk sizes are set to
   ``0x7FFFFEFF``-style sentinels instead of the real payload length.
2. Verified live: a 149,804-byte reply declared ``data`` size 2,147,483,547 and
   the real payload was 149,760 bytes. Python's ``wave`` module then reports
   1,073,741,773 frames (about 12 hours) for a 3-second clip.
3. Lenient players ignore the field and read to EOF, but strict parsers and IM
   clients may compute a bogus duration or refuse the file outright. AstrBot's
   media pipeline may also decide a conversion is needed based on the header.

II. The fix

Rewrite the RIFF size and the ``data`` chunk size to their real values, but only
when the ``data`` chunk runs to the end of the buffer and the declared sizes do
not match. Anything else is left byte-for-byte untouched.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations

import struct

#: Offsets we patch: 4 is the RIFF size, and the data chunk size is patched at
#: ``data_size_offset`` once the chunk has been located.
_RIFF_HEADER_SIZE = 12
_CHUNK_HEADER_SIZE = 8

#: Sentinels observed from streaming encoders (0x7FFFFEFF, 0x7FFFFEDB, ...).
#: Anything at or above this threshold cannot be a real PCM payload size for a
#: single chat reply, so it is treated as a placeholder.
_PLACEHOLDER_THRESHOLD = 0x7FFF0000


def normalize_wav_sizes(data: bytes) -> bytes:
    """Return ``data`` with corrected RIFF/``data`` sizes when they are wrong.

    The input is returned unchanged (same object) whenever the buffer is not a
    WAV file, is too short to contain a header, has a real ``data`` size, or has
    trailing chunks after ``data``.
    """
    if len(data) < _RIFF_HEADER_SIZE + _CHUNK_HEADER_SIZE:
        return data
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data

    declared_riff_size = struct.unpack_from("<I", data, 4)[0]
    actual_riff_size = len(data) - 8
    if declared_riff_size == actual_riff_size:
        return data

    offset = _RIFF_HEADER_SIZE
    data_size_offset: int | None = None
    declared_data_size = 0

    while offset + _CHUNK_HEADER_SIZE <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        if chunk_id == b"data":
            data_size_offset = offset + 4
            declared_data_size = chunk_size
            break
        offset += _CHUNK_HEADER_SIZE + chunk_size + (chunk_size % 2)
        # A bogus chunk size would push us past the buffer; stop rather than
        # guess, and leave the file alone.
        if offset > len(data):
            return data

    if data_size_offset is None:
        return data

    actual_data_size = len(data) - (data_size_offset + 4)
    if declared_data_size == actual_data_size and declared_riff_size == actual_riff_size:
        return data

    # Only patch when the payload really runs to the end of the buffer and the
    # declared size is a placeholder or an over-estimate. Trailing chunks (e.g.
    # a LIST/INFO block) mean the declared size may be legitimate.
    if declared_data_size < actual_data_size and declared_data_size < _PLACEHOLDER_THRESHOLD:
        return data

    patched = bytearray(data)
    struct.pack_into("<I", patched, data_size_offset, actual_data_size)
    struct.pack_into("<I", patched, 4, actual_riff_size)
    return bytes(patched)


def wav_duration_seconds(data: bytes) -> float:
    """Best-effort duration of a mono/stereo PCM WAV, ignoring bogus headers.

    Returns ``0.0`` when the buffer is not a PCM WAV file.
    """
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return 0.0

    offset = _RIFF_HEADER_SIZE
    channels = sample_rate = bits = 0
    data_size = 0
    while offset + _CHUNK_HEADER_SIZE <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        if chunk_id == b"fmt " and offset + 8 + 16 <= len(data):
            # fmt layout: format(2) channels(2) sample_rate(4) byte_rate(4)
            #              block_align(2) bits_per_sample(2)
            channels = struct.unpack_from("<H", data, offset + 10)[0]
            sample_rate = struct.unpack_from("<I", data, offset + 12)[0]
            bits = struct.unpack_from("<H", data, offset + 22)[0]
        elif chunk_id == b"data":
            # Trust the real remaining bytes rather than the declared size.
            data_size = len(data) - (offset + 8)
            break
        offset += _CHUNK_HEADER_SIZE + chunk_size + (chunk_size % 2)

    if not channels or not sample_rate or not bits:
        return 0.0
    bytes_per_second = sample_rate * channels * (bits // 8)
    if bytes_per_second <= 0:
        return 0.0
    return data_size / bytes_per_second
