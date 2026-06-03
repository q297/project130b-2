"""
Использование:
  python organize_speakers.py --ods pair_dataset.ods --output dataset/
"""

import argparse
import shutil
import pandas as pd
from pathlib import Path


def organize(ods_path: str, output_dir: str, base_dir: str = "."):
    df = pd.read_excel(ods_path, engine="odf")

    base = Path(base_dir)
    out = Path(output_dir)

    missing = []
    copied = 0

    for _, row in df.iterrows():
        speaker_id = int(row["speaker"])
        before_src = base / row["before"]
        after_src = base / row["after"]

        before_dst = out / f"speaker_{speaker_id:02d}" / "before"
        after_dst = out / f"speaker_{speaker_id:02d}" / "after"

        before_dst.mkdir(parents=True, exist_ok=True)
        after_dst.mkdir(parents=True, exist_ok=True)

        for src, dst_dir in [(before_src, before_dst), (after_src, after_dst)]:
            if not src.exists():
                missing.append(str(src))
                continue
            dst = dst_dir / src.name
            if dst.exists():
                dst = dst_dir / f"{src.stem}_{src.parent.name}{src.suffix}"
            shutil.copy2(src, dst)
            copied += 1

    print(f"Скопировано файлов: {copied}")
    if missing:
        print(f"Не найдено файлов: {len(missing)}")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... и ещё {len(missing) - 10}")
    else:
        print("Все файлы найдены.")
    print("\nСпикеры:")
    for speaker_dir in sorted(out.glob("speaker_*")):
        before_count = len(list((speaker_dir / "before").glob("*.wav")))
        after_count = len(list((speaker_dir / "after").glob("*.wav")))
        print(f"  {speaker_dir.name}: before={before_count}, after={after_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ods", required=True, help="Путь к pair_dataset.ods")
    parser.add_argument("--output", required=True, help="Папка для результата")
    parser.add_argument(
        "--base_dir",
        default=".",
        help="Базовая директория для путей из ODS (по умолчанию текущая)",
    )
    args = parser.parse_args()

    organize(args.ods, args.output, args.base_dir)
