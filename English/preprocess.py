"""English (Svarah) preprocessing.

Stages
  prepare   download Svarah from the HF hub, filter + subsample ~6 h, write
            raw_data/<speaker>/<utt>.{wav,lab} and split.json (5 h train / 1 h val,
            disjoint utterances AND disjoint transcripts).
  features  after Montreal Forced Aligner has produced TextGrids, extract
            mel / pitch / energy / duration and write train.txt / val.txt.

  python preprocess.py prepare  -c config.yaml
  # run MFA (see README), then:
  python preprocess.py features -c config.yaml
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import yaml
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def norm_text(t):
    t = re.sub(r"<[^>]*>|\[[^\]]*\]|\([^)]*\)", " ", t)  # drop annotation tags
    t = re.sub(r"\s+", " ", t).strip()
    return t


def prepare(cfg):
    import librosa
    from datasets import Audio, load_dataset

    d, pp = cfg["data"], cfg["preprocess"]["preprocessing"]
    sr = pp["audio"]["sampling_rate"]
    cleaners = pp["text"]["text_cleaners"]
    rng = random.Random(d["seed"])

    ds = load_dataset(d["hf_dataset"], split=d["hf_split"], token=os.environ.get("HF_TOKEN"))
    meta = ds.remove_columns("audio_filepath")  # columns are read without decoding audio

    # ---- filter on metadata only -------------------------------------------------
    cands = []
    for i, row in enumerate(tqdm(meta, desc="filtering")):
        text = norm_text(row["text"])
        dur = float(row["duration"])
        if not (d["min_duration"] <= dur <= d["max_duration"]):
            continue
        if len(text) < d["min_chars"] or not re.search(r"[A-Za-z]", text):
            continue
        spk = re.sub(r"\W+", "", str(row[d["speaker_by"]]).lower()) or "unk"
        cands.append(dict(idx=i, dur=dur, text=text, spk=spk))
    print(f"{len(cands)} usable utterances, {sum(c['dur'] for c in cands)/3600:.2f} h")

    # ---- group by transcript so one sentence never lands in both splits ---------
    groups = defaultdict(list)
    for c in cands:
        groups[c["text"].lower()].append(c)
    keys = sorted(groups)
    rng.shuffle(keys)

    val_s, train_s = d["val_hours"] * 3600, d["train_hours"] * 3600
    val, train, v, t = [], [], 0.0, 0.0
    for k in keys:  # fill validation first, then training, from the shuffled stream
        g = groups[k]
        gdur = sum(c["dur"] for c in g)
        if v < val_s:
            val += g; v += gdur
        elif t < train_s:
            train += g; t += gdur
        else:
            break
    print(f"train {t/3600:.2f} h ({len(train)} utts) | val {v/3600:.2f} h ({len(val)} utts)")
    if t < train_s * 0.95:
        print("WARNING: fewer than the requested training hours were available")

    # ---- write wav + lab ---------------------------------------------------------
    ds = ds.cast_column("audio_filepath", Audio(sampling_rate=sr))
    split = {"train": [], "val": [], "hours": {"train": t / 3600, "val": v / 3600}}
    for name, items in (("train", train), ("val", val)):
        for c in tqdm(items, desc=f"writing {name}"):
            a = ds[c["idx"]]["audio_filepath"]
            wav = np.asarray(a["array"], dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            wav = wav / (np.abs(wav).max() + 1e-9) * 0.95 * 32767
            base = f"{c['spk']}_{c['idx']:05d}"
            out = os.path.join(d["raw_path"], c["spk"])
            os.makedirs(out, exist_ok=True)
            wavfile.write(os.path.join(out, base + ".wav"), sr, wav.astype(np.int16))
            with open(os.path.join(out, base + ".lab"), "w", encoding="utf-8") as f:
                f.write(_clean_text(c["text"], cleaners))
            split[name].append(base)

    assert not set(split["train"]) & set(split["val"])
    os.makedirs(os.path.dirname(d["split_path"]), exist_ok=True)
    with open(d["split_path"], "w") as f:
        json.dump(split, f, indent=1)


def features(cfg):
    from preprocessor.preprocessor import Preprocessor

    Preprocessor(cfg["preprocess"], split_path=cfg["data"]["split_path"]).build_from_path()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "features"])
    ap.add_argument("-c", "--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    {"prepare": prepare, "features": features}[a.stage](cfg)
