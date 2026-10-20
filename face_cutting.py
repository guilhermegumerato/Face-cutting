#!/usr/bin/env python3
"""
face_cutting.py

Extracts ONE face crop per tagged person from photos (XMP face tags),
automatically picking the best appearance of each person.

How it works:
  1. Reads the face rectangles written in the images' XMP metadata
     (mwg-rs, and the Microsoft/Picasa format as a fallback).
  2. For each face, computes resolution in pixels, sharpness (Laplacian
     variance) and an exposure/contrast factor.
  3. Groups by name (case-insensitively, ignoring equivalent Unicode
     forms) and keeps only the best-scoring candidate per person.
  4. Writes one .jpg file per person into the output folder.

Usage:
  python face_cutting.py
  python face_cutting.py --input . --output People --margin 0.05
  python face_cutting.py --report People_report.csv

Note: the script does not depend on the external exiftool program. It
reads the XMP directly with Pillow + defusedxml, which avoids problems
with the alignment between the name and area lists that exiftool
returns.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image, ImageOps

try:
    import numpy as np
except ImportError:  # pragma: no cover - friendly message
    sys.exit(
        "The 'numpy' library is not installed.\n"
        "Activate your virtual environment and run: pip install -r requirements.txt"
    )

try:
    import defusedxml  # noqa: F401  (required for Pillow to read XMP)
except ImportError:  # pragma: no cover - friendly message
    sys.exit(
        "The 'defusedxml' library is not installed.\n"
        "Activate your virtual environment and run: pip install -r requirements.txt"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

# Folder names that should never be scanned (besides the output folder).
EXCLUDED_DIR_NAMES = {".venv", "venv", "env", "__pycache__", ".git"}

# EXIF orientations that swap width and height.
ROTATED_ORIENTATIONS = {5, 6, 7, 8}

DEFAULT_MARGIN = 0.05
DEFAULT_QUALITY = 95
MAX_FILENAME_LENGTH = 120

# Single window for measuring sharpness: every crop is scaled to this
# size before the Laplacian is computed.
SHARPNESS_WINDOW = 384


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceRegion:
    """A tagged face, in normalized coordinates (0..1)."""

    name: str
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class Candidate:
    """An appearance of a person, already evaluated."""

    key: str
    display_name: str
    file: Path
    box: tuple[int, int, int, int]  # box in pixels of the work image
    orientation: int
    coords_are_display_oriented: bool
    image_size: tuple[int, int]
    face_w_px: float
    face_h_px: float
    sharpness: float
    quality: float
    score: float

    @property
    def face_px(self) -> float:
        return self.face_w_px * self.face_h_px


# ---------------------------------------------------------------------------
# XMP reading
# ---------------------------------------------------------------------------


def _as_list(value: object) -> list:
    """The XMP from Pillow may return a single item or a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _to_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def normalize_name(name: object) -> str:
    """Normalizes the name without changing the displayed spelling."""
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    text = unicodedata.normalize("NFC", str(name))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def name_key(name: str) -> str:
    """Grouping key: ignores case and Unicode composition differences."""
    return unicodedata.normalize("NFC", name).casefold().strip()


def _xmp_description(xmp: object) -> dict:
    """
    Returns the rdf:Description block of the XMP.

    Some editors write more than one rdf:Description in the file; in that
    case Pillow returns a list and the blocks are merged.
    """
    if not isinstance(xmp, dict):
        return {}
    description = xmp.get("xmpmeta", {}).get("RDF", {}).get("Description", {})
    if isinstance(description, list):
        merged: dict = {}
        for piece in description:
            if isinstance(piece, dict):
                merged.update(piece)
        return merged
    return description if isinstance(description, dict) else {}


def _regions_from_mwg_rs(description: dict) -> Iterator[FaceRegion]:
    """Standard format (Lightroom, digiKam): mwg-rs:Regions."""
    regions = description.get("Regions")
    if not isinstance(regions, dict):
        return

    applied = regions.get("AppliedToDimensions")
    dims_w = dims_h = None
    if isinstance(applied, dict):
        dims_w = _to_float(applied.get("w"))
        dims_h = _to_float(applied.get("h"))

    region_list = regions.get("RegionList")
    if not isinstance(region_list, dict):
        return
    bag = region_list.get("Bag")
    if not isinstance(bag, dict):
        return

    for item in _as_list(bag.get("li")):
        if not isinstance(item, dict):
            continue
        desc = item.get("Description", item)
        if not isinstance(desc, dict):
            continue

        name = normalize_name(desc.get("Name", ""))
        if not name:
            continue

        region_type = desc.get("Type")
        if region_type is not None and str(region_type).strip().lower() != "face":
            continue

        area = desc.get("Area")
        if not isinstance(area, dict):
            continue

        x = _to_float(area.get("x"))
        y = _to_float(area.get("y"))
        w = _to_float(area.get("w"))
        h = _to_float(area.get("h"))
        if None in (x, y, w, h):
            continue

        unit = str(area.get("unit", "normalized")).strip().lower()
        if unit != "normalized" and dims_w and dims_h:
            x, y, w, h = x / dims_w, y / dims_h, w / dims_w, h / dims_h

        if w <= 0 or h <= 0:
            continue

        yield FaceRegion(name=name, cx=x, cy=y, w=w, h=h)


def _regions_from_microsoft(description: dict) -> Iterator[FaceRegion]:
    """Microsoft/Picasa format: MP:RegionInfo (Rectangle = x, y, w, h)."""
    region_info = description.get("RegionInfo")
    if not isinstance(region_info, dict):
        return
    regions = region_info.get("Regions")
    if not isinstance(regions, dict):
        return
    bag = regions.get("Bag")
    if not isinstance(bag, dict):
        return

    for item in _as_list(bag.get("li")):
        if not isinstance(item, dict):
            continue

        name = normalize_name(item.get("PersonDisplayName", ""))
        rectangle = item.get("Rectangle")
        if not name or not isinstance(rectangle, str):
            continue

        parts = [_to_float(part) for part in rectangle.split(",")]
        if len(parts) != 4 or any(part is None for part in parts):
            continue
        left, top, w, h = parts  # type: ignore[misc]
        if w <= 0 or h <= 0:
            continue

        # The Microsoft rectangle is (top-left corner, width, height).
        yield FaceRegion(
            name=name,
            cx=left + w / 2.0,
            cy=top + h / 2.0,
            w=w,
            h=h,
        )


def extract_face_regions(xmp: object) -> list[FaceRegion]:
    """Extracts all faces from an XMP, removing duplicates across formats."""
    description = _xmp_description(xmp)

    regions: list[FaceRegion] = []
    seen: set[tuple] = set()

    # mwg-rs is the most complete format; MP comes in as a complement.
    for source in (_regions_from_mwg_rs(description), _regions_from_microsoft(description)):
        for region in source:
            signature = (
                name_key(region.name),
                round(region.cx, 5),
                round(region.cy, 5),
                round(region.w, 5),
                round(region.h, 5),
            )
            if signature in seen:
                continue
            seen.add(signature)
            regions.append(region)

    return regions


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------


def laplacian_variance(gray: np.ndarray) -> float:
    """Laplacian variance: the higher it is, the sharper the image."""
    if gray.ndim != 2 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    center = gray[1:-1, 1:-1]
    laplacian = (
        -4.0 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(laplacian.var())


def assess_crop(crop: Image.Image) -> tuple[float, float]:
    """
    Returns (sharpness, quality).

    Sharpness is measured at a standardized size so that photos with more
    megapixels alone are not favored. Quality combines sharpness, brightness
    and contrast, penalizing very dark/blown-out faces or faces with no
    contrast at all.
    """
    long_side = max(crop.size)
    if long_side < 3:
        return 0.0, 0.0

    # Normalizes all crops to the SAME window: crops with fewer pixels are
    # upscaled (BICUBIC), crops with more pixels are downscaled (LANCZOS).
    # Without this, small crops get the "ready-made" sharpness of the native
    # resolution and end up winning over bigger, better faces.
    scale = SHARPNESS_WINDOW / long_side
    target = (max(3, round(crop.width * scale)), max(3, round(crop.height * scale)))
    resampling = (
        Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BICUBIC
    )
    gray = crop.convert("L")
    if gray.size != target:
        gray = gray.resize(target, resampling)

    array = np.asarray(gray, dtype=np.float32)
    sharpness = laplacian_variance(array)

    if array.size == 0:
        return 0.0, 0.0

    mean = float(array.mean())
    std = float(array.std())

    brightness = max(0.25, 1.0 - abs(mean - 128.0) / 128.0)
    contrast = min(1.0, std / 50.0)

    quality = max(0.0, sharpness) * (0.6 + 0.4 * brightness) * (0.6 + 0.4 * contrast)
    return sharpness, quality


# ---------------------------------------------------------------------------
# Image and orientation
# ---------------------------------------------------------------------------


def exif_orientation(image: Image.Image) -> int:
    try:
        return int(image.getexif().get(0x0112, 1) or 1)
    except Exception:
        return 1


def apply_orientation(crop: Image.Image, orientation: int) -> Image.Image:
    """Applies the EXIF orientation to a crop, without relying on the original EXIF."""
    if orientation in (None, 1):
        return crop

    exif = crop.getexif()
    exif[0x0112] = orientation
    crop.info["exif"] = exif.tobytes()
    return ImageOps.exif_transpose(crop)


def prepare_work_image(
    image: Image.Image,
    orientation: int,
    xmp: object,
) -> tuple[Image.Image, bool]:
    """
    Decides whether the XMP coordinates are in the stored or the display orientation.

    Returns (work_image, coordinates_are_already_display_oriented).
    """
    if orientation not in ROTATED_ORIENTATIONS:
        return image, False

    description = _xmp_description(xmp)
    regions = description.get("Regions")
    applied = regions.get("AppliedToDimensions") if isinstance(regions, dict) else None
    if not isinstance(applied, dict):
        return image, False

    dims_w = _to_float(applied.get("w"))
    dims_h = _to_float(applied.get("h"))
    stored_w, stored_h = image.size

    # If the XMP dimensions are the already-rotated dimensions, the
    # coordinates are in display orientation.
    if (
        dims_w is not None
        and dims_h is not None
        and abs(dims_w - stored_h) <= 1
        and abs(dims_h - stored_w) <= 1
        and image.size != (int(dims_w), int(dims_h))
    ):
        return ImageOps.exif_transpose(image), True

    return image, False


def crop_region(
    work_image: Image.Image,
    region: FaceRegion,
    margin: float,
    coords_are_display_oriented: bool,
    orientation: int,
) -> tuple[Image.Image | None, tuple[int, int, int, int], float, float]:
    """Crops a face and also returns the box and the size in pixels."""
    width, height = work_image.size

    w = region.w * width
    h = region.h * height
    cx = region.cx * width
    cy = region.cy * height

    if w < 1 or h < 1:
        return None, (0, 0, 0, 0), w, h

    expand_x = w * margin
    expand_y = h * margin

    left = max(0, int(round(cx - w / 2 - expand_x)))
    top = max(0, int(round(cy - h / 2 - expand_y)))
    right = min(width, int(round(cx + w / 2 + expand_x)))
    bottom = min(height, int(round(cy + h / 2 + expand_y)))

    if right <= left or bottom <= top:
        return None, (0, 0, 0, 0), w, h

    crop = work_image.crop((left, top, right, bottom))

    # When the coordinates were in the stored orientation, the crop is
    # rotated afterwards; this way the face comes out in the correct position.
    if orientation not in (None, 1) and not coords_are_display_oriented:
        crop = apply_orientation(crop, orientation)

    return crop, (left, top, right, bottom), w, h


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def iter_images(input_dir: Path, output_dir: Path) -> Iterator[Path]:
    output_resolved = output_dir.resolve()

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            relative_parts = path.relative_to(input_dir).parts
        except ValueError:
            relative_parts = path.parts

        if any(part in EXCLUDED_DIR_NAMES or part.startswith(".") for part in relative_parts[:-1]):
            continue

        try:
            if path.resolve().is_relative_to(output_resolved):
                continue
        except (OSError, ValueError):
            pass

        yield path


# ---------------------------------------------------------------------------
# Output filename
# ---------------------------------------------------------------------------


def safe_filename(name: str) -> str:
    """Turns the person's name into a safe, portable filename."""
    text = unicodedata.normalize("NFC", name)
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    text = text[:MAX_FILENAME_LENGTH].rstrip(" .")
    return text


def unique_output_names(
    best: dict[str, Candidate],
) -> dict[str, str]:
    """Guarantees unique filenames even on case-insensitive file systems."""
    used: dict[str, str] = {}
    result: dict[str, str] = {}

    for key in sorted(best, key=lambda k: best[k].display_name.casefold()):
        base = safe_filename(best[key].display_name) or "Person"
        candidate = base
        counter = 2
        while candidate.casefold() in used and used[candidate.casefold()] != key:
            candidate = f"{base} ({counter})"
            counter += 1
        used[candidate.casefold()] = key
        result[key] = candidate

    return result


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Crops one face per tagged person from photos (XMP tags).",
    )
    parser.add_argument(
        "--input",
        default=str(script_dir),
        help="Folder with the photos (default: the script's folder).",
    )
    parser.add_argument(
        "--output",
        default=str(script_dir / "People"),
        help='Output folder (default: "People" next to the script).',
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"Extra margin around the face, as a fraction (default: {DEFAULT_MARGIN}).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_QUALITY,
        help=f"JPEG quality from 1 to 100 (default: {DEFAULT_QUALITY}).",
    )
    parser.add_argument(
        "--min-face",
        type=float,
        default=0.0,
        help="Ignores faces smaller than this value, in pixels (default: 0).",
    )
    parser.add_argument(
        "--report",
        nargs="?",
        const="__default__",
        default=None,
        help="Writes a CSV with the chosen photo and the metrics of each person.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help=(
            "Downscales the final crop to at most this size on the longest "
            "side, in pixels (0 = no limit; default: 0)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyzes and chooses, but does not write the images.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Shows files with no faces, ignored files, and errors.",
    )

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: input folder not found: {input_dir}", file=sys.stderr)
        return 1

    margin = max(0.0, float(args.margin))
    quality = max(1, min(100, int(args.quality)))

    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")

    files = list(iter_images(input_dir, output_dir))
    print(f"Analyzing {len(files)} images...")

    candidates: dict[str, list[Candidate]] = defaultdict(list)
    display_names: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[tuple[Path, str]] = []
    skipped_no_regions = 0
    skipped_no_face = 0

    for index, path in enumerate(files, start=1):
        try:
            with Image.open(path) as image:
                xmp = image.getxmp()
                regions = extract_face_regions(xmp)
                if not regions:
                    skipped_no_regions += 1
                    if args.verbose:
                        print(f"  [no faces] {path}")
                    continue

                orientation = exif_orientation(image)
                work_image, coords_are_display = prepare_work_image(
                    image, orientation, xmp
                )
                transposed = work_image is not image

                try:
                    for region in regions:
                        key = name_key(region.name)
                        if not key:
                            continue

                        crop, box, face_w_px, face_h_px = crop_region(
                            work_image,
                            region,
                            margin,
                            coords_are_display,
                            orientation,
                        )
                        if crop is None:
                            continue

                        face_px = face_w_px * face_h_px
                        if args.min_face and max(face_w_px, face_h_px) < args.min_face:
                            crop.close()
                            skipped_no_face += 1
                            continue

                        try:
                            sharpness, crop_quality = assess_crop(crop)
                        finally:
                            crop.close()

                        # The score mixes the face size (in pixels) with
                        # quality. The square roots keep a huge but soft
                        # photo from winning just for having many pixels.
                        score = math.sqrt(max(face_px, 0.0)) * math.sqrt(
                            max(crop_quality, 0.0)
                        )

                        display = normalize_name(region.name)
                        display_names[key][display] += 1
                        candidates[key].append(
                            Candidate(
                                key=key,
                                display_name=display,
                                file=path,
                                box=box,
                                orientation=orientation,
                                coords_are_display_oriented=coords_are_display,
                                image_size=work_image.size,
                                face_w_px=face_w_px,
                                face_h_px=face_h_px,
                                sharpness=sharpness,
                                quality=crop_quality,
                                score=score,
                            )
                        )
                finally:
                    # prepare_work_image may return a TRANSPOSED copy;
                    # it must be closed to avoid leaking memory.
                    if transposed:
                        work_image.close()

        except Exception as exc:  # noqa: BLE001 - we want to continue with the other photos
            errors.append((path, str(exc)))
            if args.verbose:
                print(f"  [error] {path}: {exc}")

        if index % 50 == 0 or index == len(files):
            print(f"  {index}/{len(files)} images processed")

    if not candidates:
        print("No tagged people were found.", file=sys.stderr)
        return 2

    # Picks the best candidate for each person.
    best: dict[str, Candidate] = {}
    for key, items in candidates.items():
        best[key] = max(
            items,
            key=lambda item: (
                item.score,
                item.face_px,
                item.sharpness,
                item.image_size[0] * item.image_size[1],
            ),
        )

    # Display name: the most frequent spelling among the appearances.
    for key, counter in display_names.items():
        best[key].display_name = sorted(
            counter.items(), key=lambda item: (-item[1], item[0].casefold())
        )[0][0]

    output_names = unique_output_names(best)

    print()
    print(f"People found: {len(best)}")
    print(f"Appearances evaluated: {sum(len(v) for v in candidates.values())}")
    print(f"Images with no tagged faces: {skipped_no_regions}")

    if args.verbose and skipped_no_face:
        print(f"Faces ignored by --min-face: {skipped_no_face}")

    if errors:
        print(f"Errors while reading {len(errors)} image(s):", file=sys.stderr)
        for path, message in errors[:20]:
            print(f"  {path}: {message}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)

    report_path: Path | None = None
    if args.report is not None:
        if args.report == "__default__":
            report_path = output_dir.parent / "People_report.csv"
        else:
            report_path = Path(args.report).expanduser().resolve()

    if args.dry_run:
        print("\n[dry-run] No files were written.")
        for key in sorted(best, key=lambda k: best[k].display_name.casefold()):
            item = best[key]
            print(
                f"  {item.display_name}: {output_names[key]}.jpg "
                f"<- {item.file.name} "
                f"({item.face_w_px:.0f}x{item.face_h_px:.0f}px, "
                f"{len(candidates[key])} appearances)"
            )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for key in sorted(best, key=lambda k: best[k].display_name.casefold()):
        item = best[key]
        output_file = output_dir / f"{output_names[key]}.jpg"

        max_side = max(0, int(args.max_side))
        try:
            icc_profile = None
            final_crop: Image.Image | None = None
            try:
                with Image.open(item.file) as original:
                    icc_profile = original.info.get("icc_profile")
                    work = original
                    if item.coords_are_display_oriented:
                        # exif_transpose returns a copy; close it afterwards.
                        work = ImageOps.exif_transpose(original)

                    try:
                        crop = work.crop(item.box)
                    finally:
                        if work is not original:
                            work.close()

                if item.orientation not in (None, 1) and not item.coords_are_display_oriented:
                    crop = apply_orientation(crop, item.orientation)

                if crop.mode != "RGB":
                    crop = crop.convert("RGB")

                if max_side and max(crop.size) > max_side:
                    crop.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

                save_kwargs: dict[str, object] = {
                    "format": "JPEG",
                    "quality": quality,
                    "optimize": True,
                }
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile

                crop.save(output_file, **save_kwargs)
                final_crop = crop
            finally:
                if final_crop is not None:
                    final_crop.close()

            written += 1
            print(
                f"  {item.display_name} -> {output_file.name} "
                f"({item.face_w_px:.0f}x{item.face_h_px:.0f}px)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [write error] {item.display_name}: {exc}", file=sys.stderr)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow(
                [
                    "person",
                    "source_file",
                    "face_width_px",
                    "face_height_px",
                    "sharpness",
                    "quality",
                    "score",
                    "appearances",
                    "output",
                ]
            )
            for key in sorted(best, key=lambda k: best[k].display_name.casefold()):
                item = best[key]
                writer.writerow(
                    [
                        item.display_name,
                        str(item.file),
                        f"{item.face_w_px:.1f}",
                        f"{item.face_h_px:.1f}",
                        f"{item.sharpness:.2f}",
                        f"{item.quality:.2f}",
                        f"{item.score:.2f}",
                        len(candidates[key]),
                        f"{output_names[key]}.jpg",
                    ]
                )
        print(f"Report: {report_path}")

    print(f"\nDone: {written} face(s) written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
