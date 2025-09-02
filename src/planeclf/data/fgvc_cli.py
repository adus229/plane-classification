import argparse
from pathlib import Path
from .fgvc import (
    download_fgvc_aircraft,
    extract_archive,
    build_imagefolder_from_fgvc,
    ensure_train_val_from_trainval,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="FGVC Aircraft 2013b downloader & preparer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Download the official FGVC 2013b tarball")
    p_dl.add_argument("--url", default=None, help="Override URL (optional)")
    p_dl.add_argument("--out", default="data_raw/fgvc-aircraft-2013b.tar.gz", help="Output archive path")

    p_prep = sub.add_parser("extract", help="Extract + build ImageFolder + split train/val")
    p_prep.add_argument("--archive", default="data_raw/fgvc-aircraft-2013b.tar.gz", help="Path to archive")
    p_prep.add_argument("--extract-to", default="data_raw", help="Extraction directory")
    p_prep.add_argument("--fgvc-root", default=None, help="Root after extraction (auto-detected if None)")
    p_prep.add_argument("--out", default="data_fgvc", help="Where to write the ImageFolder tree")
    p_prep.add_argument("--val", type=float, default=0.2, help="Validation ratio (from trainval)")
    p_prep.add_argument("--move", action="store_true", help="Move files instead of copy")

    args = parser.parse_args()
    if args.cmd == "download":
        out = download_fgvc_aircraft(url=args.url, out_path=Path(args.out))
        print(out)
    elif args.cmd == "extract":
        extract_archive(Path(args.archive), Path(args.extract_to))
        fgvc_root = Path(args.fgvc_root) if args.fgvc_root else Path(args.extract_to) / "fgvc-aircraft-2013b"
        build_imagefolder_from_fgvc(fgvc_root, Path(args.out), split="trainval", copy=(not args.move))
        ensure_train_val_from_trainval(Path(args.out), Path("data"), val_ratio=args.val, copy=(not args.move))

if __name__ == "__main__":
    main()
