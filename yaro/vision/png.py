"""极简 PNG 编解码 + 标注叠加渲染（纯标准库）。

为什么自己写：标注审核的核心证据是"把框画回图上"给人和模型看。为这一步引入
Pillow / OpenCV 会让"装完就能跑"的契约失效，而这个子集（8bit、非隔行、灰度/
RGB/RGBA）几十行就够，还能被单元测试逐字节验证。
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Iterable, Sequence

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS_BY_COLOR_TYPE = {0: 1, 2: 3, 4: 2, 6: 4}


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


@dataclass(frozen=True)
class Image:
    """解码后的位图：pixels 是逐行紧排的 8bit 样本。"""

    width: int
    height: int
    channels: int
    pixels: bytes

    @property
    def stride(self) -> int:
        return self.width * self.channels

    def to_rgb(self) -> "Image":
        if self.channels == 3:
            return self
        out = bytearray()
        src = self.pixels
        if self.channels == 1:
            for value in src:
                out += bytes((value, value, value))
        elif self.channels == 2:
            for index in range(0, len(src), 2):
                value = src[index]
                out += bytes((value, value, value))
        elif self.channels == 4:
            for index in range(0, len(src), 4):
                out += src[index : index + 3]
        else:  # pragma: no cover - 构造上不会出现
            raise ValueError(f"不支持的颜色通道数：{self.channels}")
        return Image(self.width, self.height, 3, bytes(out))


def encode_png(width: int, height: int, rows: Sequence[bytes]) -> bytes:
    """编码 8bit RGB PNG（每行必须 width*3 字节，filter=0）。"""
    if len(rows) != height:
        raise ValueError(f"rows 数量应为 {height}，实际 {len(rows)}")
    raw = bytearray()
    for row in rows:
        if len(row) != width * 3:
            raise ValueError(f"每行应为 {width * 3} 字节，实际 {len(row)}")
        raw.append(0)
        raw.extend(row)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return PNG_SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + _chunk(b"IEND", b"")


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode_png(data: bytes) -> Image:
    """解码 8bit 非隔行 PNG（灰度/RGB/灰度+A/RGBA）。失败抛 ValueError。"""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("不是 PNG：签名不匹配")
    position = len(PNG_SIGNATURE)
    width = height = 0
    color_type = 2
    idat = bytearray()
    while position + 8 <= len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        kind = data[position + 4 : position + 8]
        payload = data[position + 8 : position + 8 + length]
        position += 12 + length
        if kind == b"IHDR":
            if len(payload) < 13:
                raise ValueError("IHDR 长度不足")
            width, height, depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", payload[:13])
            if depth != 8:
                raise ValueError(f"只支持 8bit PNG，实际 {depth}bit")
            if interlace:
                raise ValueError("不支持隔行 PNG")
            if color_type not in _CHANNELS_BY_COLOR_TYPE:
                raise ValueError(f"不支持的颜色类型：{color_type}")
            if compression or filter_method:
                raise ValueError("不支持的压缩/滤波方法")
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            break
    if width <= 0 or height <= 0:
        raise ValueError("缺少有效 IHDR")
    channels = _CHANNELS_BY_COLOR_TYPE[color_type]
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:  # pragma: no cover - 损坏文件
        raise ValueError(f"IDAT 解压失败：{exc}") from exc
    stride = width * channels
    expected = (stride + 1) * height
    if len(raw) < expected:
        raise ValueError(f"像素数据不足：期望 {expected} 字节，实际 {len(raw)}")
    out = bytearray(height * stride)
    previous = bytearray(stride)
    offset = 0
    for y in range(height):
        filter_type = raw[offset]
        offset += 1
        line = bytearray(raw[offset : offset + stride])
        offset += stride
        if filter_type == 1:
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                line[x] = (line[x] + left) & 0xFF
        elif filter_type == 2:
            for x in range(stride):
                line[x] = (line[x] + previous[x]) & 0xFF
        elif filter_type == 3:
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                line[x] = (line[x] + ((left + previous[x]) >> 1)) & 0xFF
        elif filter_type == 4:
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                up_left = previous[x - channels] if x >= channels else 0
                line[x] = (line[x] + _paeth(left, previous[x], up_left)) & 0xFF
        elif filter_type != 0:
            raise ValueError(f"未知的 PNG filter 类型：{filter_type}")
        out[y * stride : (y + 1) * stride] = line
        previous = line
    return Image(width, height, channels, bytes(out))


def resize_nearest(image: Image, width: int, height: int) -> Image:
    """最近邻缩放（渲染缩略图/联系人表用，不追求画质）。"""
    width = max(1, int(width))
    height = max(1, int(height))
    rgb = image.to_rgb()
    out = bytearray(width * height * 3)
    src_w, src_h = rgb.width, rgb.height
    for y in range(height):
        sy = min(src_h - 1, (y * src_h) // height)
        for x in range(width):
            sx = min(src_w - 1, (x * src_w) // width)
            source = (sy * src_w + sx) * 3
            target = (y * width + x) * 3
            out[target : target + 3] = rgb.pixels[source : source + 3]
    return Image(width, height, 3, bytes(out))


def _blend_pixel(buffer: bytearray, index: int, color: tuple[int, int, int]) -> None:
    buffer[index] = color[0]
    buffer[index + 1] = color[1]
    buffer[index + 2] = color[2]


def draw_rect(image: Image, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int], thickness: int = 2) -> Image:
    """在 RGB 图上画空心矩形（坐标先裁剪到图内）。"""
    rgb = image.to_rgb()
    buffer = bytearray(rgb.pixels)
    width, height = rgb.width, rgb.height
    left = max(0, min(width - 1, int(round(x0))))
    right = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y0))))
    bottom = max(0, min(height - 1, int(round(y1))))
    if right <= left or bottom <= top:
        return rgb
    for offset in range(max(1, int(thickness))):
        for x in range(left, right + 1):
            for y in (top + offset, bottom - offset):
                if 0 <= y < height:
                    _blend_pixel(buffer, (y * width + x) * 3, color)
        for y in range(top, bottom + 1):
            for x in (left + offset, right - offset):
                if 0 <= x < width:
                    _blend_pixel(buffer, (y * width + x) * 3, color)
    return Image(width, height, 3, bytes(buffer))


def render_boxes(
    data: bytes,
    boxes: Iterable[dict],
    *,
    classes: Sequence[str] | None = None,
    thickness: int = 2,
) -> bytes:
    """把归一化 YOLO 框画回 PNG 字节流，返回新的 PNG 字节。

    ``boxes`` 每项形如 ``{"cls": 0, "cx": .5, "cy": .5, "w": .2, "h": .3}``。
    越界框也会被画出来（裁剪到图内），因为"框越界"本身就是审计要暴露的问题。
    """
    image = decode_png(data).to_rgb()
    palette = [(255, 99, 71), (64, 196, 255), (255, 206, 84), (124, 252, 0), (221, 160, 221), (255, 140, 0), (0, 206, 209), (255, 105, 180)]
    for box in boxes:
        cls = int(box.get("cls", 0))
        color = palette[cls % len(palette)]
        cx = float(box.get("cx", 0.0)) * image.width
        cy = float(box.get("cy", 0.0)) * image.height
        half_w = float(box.get("w", 0.0)) * image.width / 2
        half_h = float(box.get("h", 0.0)) * image.height / 2
        image = draw_rect(image, cx - half_w, cy - half_h, cx + half_w, cy + half_h, color, thickness=thickness)
    rows = [image.pixels[y * image.stride : (y + 1) * image.stride] for y in range(image.height)]
    return encode_png(image.width, image.height, rows)


def compose_contact_sheet(images: Sequence[Image], *, cols: int = 3, cell: int = 220, gap: int = 8, background: tuple[int, int, int] = (16, 16, 18)) -> Image:
    """把多张图缩略成一张联系人表（review 大量样本时一张图看全局）。"""
    if not images:
        return Image(cell, cell, 3, bytes((16, 16, 18)) * (cell * cell))
    cols = max(1, min(int(cols), len(images)))
    rows = (len(images) + cols - 1) // cols
    width = cols * cell + (cols + 1) * gap
    height = rows * cell + (rows + 1) * gap
    canvas = bytearray(bytes(background) * (width * height))
    for index, image in enumerate(images):
        scale = min(cell / max(1, image.width), cell / max(1, image.height))
        thumb = resize_nearest(image, max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        col = index % cols
        row = index // cols
        origin_x = gap + col * (cell + gap) + (cell - thumb.width) // 2
        origin_y = gap + row * (cell + gap) + (cell - thumb.height) // 2
        for y in range(thumb.height):
            target_row = (origin_y + y) * width
            for x in range(thumb.width):
                source = (y * thumb.width + x) * 3
                target = (target_row + origin_x + x) * 3
                canvas[target : target + 3] = thumb.pixels[source : source + 3]
    rows_bytes = [bytes(canvas[y * width * 3 : (y + 1) * width * 3]) for y in range(height)]
    return Image(width, height, 3, b"".join(rows_bytes))


def image_from_bytes(data: bytes) -> Image:
    return decode_png(data)


def to_png_bytes(image: Image) -> bytes:
    rgb = image.to_rgb()
    rows = [rgb.pixels[y * rgb.stride : (y + 1) * rgb.stride] for y in range(rgb.height)]
    return encode_png(rgb.width, rgb.height, rows)
