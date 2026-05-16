"""Batch inference entry point for TED-TTS.

Loads samples from a JSON manifest (see datasets/MED-TTS/test_samples/eng.json
for the expected schema) and runs one of six inference modes per sample.

Supported modes (mirrors infer_single.py):
  0  No emotion control.
  1  Emotion reference audio + emo_alpha (global, shared across samples).
  2  8-dim emotion vector (one-hot per segment derived from the JSON 'emotion'
     field; override globally with --emo_vector).
  3  Per-segment emotion text descriptions taken from JSON 'emotion_description'.
  4  Mode 3 + global EOS duration control (target seconds from JSON 'time').
  5  Mode 3 + combined global + local duration control.

Expected JSON schema:
    [
      {
        "input":  {"text": "...", "emotion_seq": [...]},
        "output": {
          "language": "english",
          "speaker_id": "0011",
          "segments": [
            {
              "lines_seg":           "...",
              "emotion":             "Angry",
              "emotion_description": "The voice is tense and sharp.",
              "time":                "2.4",
              "prompt_wav":          "0011/Neutral/0011_000001.wav"
            }, ...
          ]
        }
      }, ...
    ]

Examples:
    python infer_batch.py --mode 3 \\
        --data_file ./datasets/MED-TTS/test_samples/eng.json \\
        --save_dir ./results/batch_eng_mode3

    python infer_batch.py --mode 5 \\
        --data_file ./datasets/MED-TTS/test_samples/eng.json \\
        --save_dir ./results/batch_eng_mode5 \\
        --max_samples 10
"""

import argparse
import json
import os
import traceback
from pathlib import Path

from indextts.infer_v2 import IndexTTS2


EMOTION_DIMENSIONS = [
    "happy", "angry", "sad", "afraid",
    "disgusted", "melancholic", "surprised", "calm",
]

# Maps JSON emotion labels onto one of EMOTION_DIMENSIONS.
EMOTION_NAME_MAP = {
    "angry":     "angry",
    "happy":     "happy",
    "neutral":   "calm",
    "sad":       "sad",
    "surprise":  "surprised",
    "surprised": "surprised",
    "fear":      "afraid",
    "fearful":   "afraid",
    "disgust":   "disgusted",
    "disgusted": "disgusted",
}


def emotion_to_vec(emotion: str) -> list:
    """Map an emotion label to a one-hot 8-dim vector."""
    key = emotion.lower().strip()
    target = EMOTION_NAME_MAP.get(key, "calm")
    vec = [0.0] * 8
    vec[EMOTION_DIMENSIONS.index(target)] = 1.0
    return vec


def seconds_to_mel_tokens(seconds, mel_to_sec_ratio=0.02):
    """Convert seconds (scalar or list) to mel token counts."""
    if isinstance(seconds, (list, tuple)):
        return [int(s / mel_to_sec_ratio) for s in seconds]
    return int(seconds / mel_to_sec_ratio)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TED-TTS batch inference with six emotion/duration modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--mode", type=int, choices=[0, 1, 2, 3, 4, 5], required=True,
                        help="Inference mode (see header docstring).")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the JSON sample manifest.")
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="Output directory (default writes under ./results/).")
    parser.add_argument("--dataset_dir", type=str,
                        default="./datasets/Emotion Speech Dataset",
                        help="ESD dataset root, used to resolve relative prompt_wav paths.")
    parser.add_argument("--spk_audio", type=str, default=None,
                        help="Global speaker prompt override; defaults to each sample's first prompt_wav.")

    parser.add_argument("--emo_audio", type=str, default=None,
                        help="Mode 1: global emotion reference wav.")
    parser.add_argument("--emo_alpha", type=float, default=1.0,
                        help="Mode 1: emotion blending weight.")

    parser.add_argument("--emo_vector", type=float, nargs=8, default=None,
                        metavar=("HAP", "ANG", "SAD", "FEA", "HAT", "LOW", "SUR", "NEU"),
                        help="Mode 2: global override; otherwise vectors are derived per segment.")

    parser.add_argument("--max_samples", type=int, default=None,
                        help="Stop after this many samples (default: process all).")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start at this sample index (default: 0).")
    parser.add_argument("--skip_existing", action="store_true", default=False,
                        help="Skip samples whose output wav already exists.")

    parser.add_argument("--model_dir", type=str, default="./checkpoints")
    parser.add_argument("--cfg_path", type=str, default="./checkpoints/config.yaml")
    parser.add_argument("--is_fp16", action="store_true", default=False)

    parser.add_argument("--max_mel_tokens", type=int, default=850)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_text_tokens_per_sentence", type=int, default=150)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=10.0)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--use_random", action="store_true", default=False)
    parser.add_argument("--method", type=str, default="hmm", choices=["hmm", "max_head"],
                        help="Modes 3/4/5: segment-switch detection method.")

    return parser.parse_args()


DEFAULT_SPK_FALLBACK = "./datasets/Ref/0011_000001.wav"


def resolve_spk_audio(item, dataset_dir, override):
    """Pick a speaker prompt path for one sample.

    Priority: --spk_audio override > JSON prompt_wav (joined with --dataset_dir)
    > bundled fallback at ./datasets/Ref/0011_000001.wav > None (skip).
    The bundled fallback lets the pipeline run without an ESD download.
    """
    if override:
        return override
    segments = item.get("output", {}).get("segments", [])
    if segments and segments[0].get("prompt_wav"):
        candidate = Path(dataset_dir) / segments[0]["prompt_wav"]
        if candidate.is_file():
            return str(candidate)
        print(f"  [warn] prompt_wav not found at '{candidate}'; "
              f"falling back to bundled '{DEFAULT_SPK_FALLBACK}'")
    if Path(DEFAULT_SPK_FALLBACK).is_file():
        return DEFAULT_SPK_FALLBACK
    return None


def process_sample(args, item, tts, sent_idx):
    """Run a single sample through the requested mode. Returns the output path or None."""
    segments = item.get("output", {}).get("segments", [])
    if not segments:
        print(f"  [skip] sample {sent_idx}: no segments")
        return None

    spk_audio = resolve_spk_audio(item, args.dataset_dir, args.spk_audio)
    if not spk_audio:
        print(f"  [skip] sample {sent_idx}: no speaker prompt (set --spk_audio or add prompt_wav)")
        return None

    language = item.get("output", {}).get("language", "english")
    output_wav = os.path.join(args.save_dir,
                              f"{language}_{sent_idx:04d}_mode{args.mode}.wav")

    if args.skip_existing and os.path.exists(output_wav):
        print(f"  [skip] sample {sent_idx}: output exists at {output_wav}")
        return output_wav

    lines_seg = [seg.get("lines_seg", "").strip() for seg in segments]
    emo_descriptions = [seg.get("emotion_description", "neutral") for seg in segments]
    emotions = [seg.get("emotion", "neutral") for seg in segments]
    times = [float(seg.get("time", 0)) for seg in segments]

    multi_text = "|".join(lines_seg)
    single_text = " ".join(lines_seg)

    common_kwargs = dict(
        spk_audio_prompt=spk_audio,
        output_path=output_wav,
        verbose=True,
        use_random=args.use_random,
        max_text_tokens_per_sentence=args.max_text_tokens_per_sentence,
        do_sample=True,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        length_penalty=args.length_penalty,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        max_mel_tokens=args.max_mel_tokens,
        method=args.method,
    )

    if args.mode == 0:
        tts.infer(text=single_text, **common_kwargs)

    elif args.mode == 1:
        tts.infer(
            text=single_text,
            emo_audio_prompt=args.emo_audio,
            emo_alpha=args.emo_alpha,
            **common_kwargs,
        )

    elif args.mode == 2:
        if args.emo_vector is not None:
            # Global CLI override: single-segment text with one shared vector.
            emo_vectors = [args.emo_vector]
            text = single_text
        else:
            emo_vectors = [emotion_to_vec(e) for e in emotions]
            text = multi_text
        tts.infer(
            text=text,
            emo_audio_prompt=None,
            emo_alpha=0,
            emo_vector=emo_vectors,
            **common_kwargs,
        )

    elif args.mode == 3:
        tts.infer(
            text=multi_text,
            emo_audio_prompt=None,
            emo_alpha=0,
            use_emo_text=True,
            emo_text="|".join(emo_descriptions),
            **common_kwargs,
        )

    else:
        target_tokens = seconds_to_mel_tokens(times)
        duration_mode = "global" if args.mode == 4 else "both"
        print(f"  [mode {args.mode}] duration_mode={duration_mode}, "
              f"target_seconds={times}, target_tokens={target_tokens}")
        tts.infer(
            text=multi_text,
            emo_audio_prompt=None,
            emo_alpha=0,
            use_emo_text=True,
            emo_text="|".join(emo_descriptions),
            target_duration_tokens=target_tokens,
            duration_mode=duration_mode,
            **common_kwargs,
        )

    return output_wav


def main():
    args = parse_args()

    if args.mode == 1 and args.emo_audio is None:
        raise ValueError("Mode 1 requires --emo_audio (shared across all samples).")

    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tts = IndexTTS2(model_dir=args.model_dir,
                    cfg_path=args.cfg_path,
                    is_fp16=args.is_fp16)

    end_idx = args.start_idx + args.max_samples if args.max_samples else len(data)
    end_idx = min(end_idx, len(data))
    samples = data[args.start_idx:end_idx]

    print()
    print("=" * 60)
    print(f"  TED-TTS batch inference (mode {args.mode})")
    print("=" * 60)
    print(f"  data_file: {args.data_file}")
    print(f"  range:     [{args.start_idx}, {end_idx})  ({len(samples)} samples)")
    print(f"  save_dir:  {args.save_dir}")
    print()

    processed = 0
    failed = 0
    for i, item in enumerate(samples):
        sent_idx = args.start_idx + i
        print(f"\n[{i + 1}/{len(samples)}] sample {sent_idx}")
        try:
            if process_sample(args, item, tts, sent_idx):
                processed += 1
        except Exception as exc:
            failed += 1
            print(f"  [error] sample {sent_idx}: {exc}")
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"  done: processed={processed}, failed={failed}, total={len(samples)}")
    print(f"  output dir: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
