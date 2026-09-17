"""Smoke test for wheels built against the VP8/VP9-decode-only LGPL FFmpeg.

The bundled FFmpeg (JScheumann/pyav-ffmpeg-lgpl) is built with
``--disable-everything --enable-decoder=vp8,vp9`` and no external libraries,
so the regular test suite -- which needs many codecs, demuxers and protocols --
cannot run against it. This module checks what that build promises:

* the libraries are LGPL and contain no GPL-only or libx264/libx265 code,
* exactly the VP8 and VP9 decoders exist, nothing else, and
* raw VP8/VP9 frames (as delivered by the Jetson encoder, IVF-framed) decode
  to RGB through the same API path the consuming application uses:
  ``CodecContext.create(codec, "r")`` -> ``decode(Packet)`` -> ``to_ndarray``.

The samples are 5-frame 64x48 IVF files generated with
``ffmpeg -f lavfi -i testsrc2=size=64x48:rate=10 -frames:v 5 -c:v libvpx[-vp9] -f ivf``
and embedded here so the test needs no downloads and no demuxer.
"""

import base64
import struct

import pytest

import av
import av._core

WIDTH, HEIGHT, FRAMES = 64, 48, 5

IVF_SAMPLES_B64 = {
    "vp8": (
        "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"
        "+iwnuBAAgBvhFxZ7kA=="
    ),
    "vp9": (
        "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"
    ),
}


def iter_ivf_frames(data: bytes):
    """Yield the raw codec payloads of an IVF byte string.

    IVF is a 32-byte "DKIF" file header followed by frames that each carry a
    12-byte {size, timestamp} header. This mirrors the framing the consuming
    application strips before handing payloads to PyAV.
    """
    assert data[:4] == b"DKIF"
    header_size = struct.unpack_from("<H", data, 6)[0]
    offset = header_size
    while offset < len(data):
        frame_size = struct.unpack_from("<I", data, offset)[0]
        offset += 12
        yield data[offset : offset + frame_size]
        offset += frame_size


def test_ffmpeg_is_lgpl_and_gpl_free() -> None:
    for name in ("libavutil", "libavcodec", "libavformat", "libswscale"):
        meta = av._core.library_meta[name]
        assert meta["license"].startswith("LGPL"), (name, meta["license"])
        assert "--enable-gpl" not in meta["configuration"], name
        assert "--enable-nonfree" not in meta["configuration"], name


def test_only_vp8_and_vp9_are_available() -> None:
    assert av.codecs_available == {"vp8", "vp9"}
    for name in ("vp8", "vp9"):
        codec = av.codec.Codec(name, "r")
        assert codec.name == name
        assert codec.type == "video"
    for name in ("h264", "hevc", "aac", "libx264"):
        with pytest.raises(av.codec.codec.UnknownCodecError):
            av.codec.Codec(name, "r")


@pytest.mark.parametrize("codec_name", ["vp8", "vp9"])
def test_decode_raw_frames_to_rgb(codec_name: str) -> None:
    import numpy as np

    data = base64.b64decode("".join(IVF_SAMPLES_B64[codec_name]))
    decoder = av.CodecContext.create(codec_name, "r")

    decoded = []
    for payload in iter_ivf_frames(data):
        for frame in decoder.decode(av.Packet(payload)):
            decoded.append(frame.to_ndarray(format="rgb24"))
    for frame in decoder.decode(None):
        decoded.append(frame.to_ndarray(format="rgb24"))

    assert len(decoded) == FRAMES
    for img in decoded:
        assert isinstance(img, np.ndarray)
        assert img.shape == (HEIGHT, WIDTH, 3)
        assert img.dtype == np.uint8
    # testsrc2 is not a flat image, so a real decode produces varied pixels.
    assert decoded[0].std() > 10
