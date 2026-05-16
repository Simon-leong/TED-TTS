"""Single-sample inference entry point for TED-TTS.

Supports six modes selected via --mode:
  0  No emotion control (speaker prompt drives both timbre and emotion).
  1  Emotion reference audio with emo_alpha weighting.
  2  8-dimensional emotion vector
     (order: Happy, Angry, Sad, Fear, Hate, Low, Surprise, Neutral).
  3  Per-segment emotion text descriptions ("|"-separated).
  4  Mode 3 plus global EOS duration control via --target_seconds.
  5  Mode 3 plus combined global + local duration control.

Examples:
    python infer_single.py --mode 0
    python infer_single.py --mode 3
    python infer_single.py --mode 5

    python infer_single.py --mode 1 \\
        --emo_audio "./datasets/Emotion Speech Dataset/0011/Angry/0011_000351.wav" \\
        --emo_alpha 0.8

    python infer_single.py --mode 2 --emo_vector 0 0 0 0 0 0 0.45 0

    python infer_single.py --mode 5 \\
        --spk_audio "<voice.wav>" \\
        --text "seg1|seg2|seg3" \\
        --emo_text "happy|sad|neutral" \\
        --target_seconds 2.0 1.5 1.8 \\
        --output my_output.wav
"""

import argparse

from indextts.infer_v2 import IndexTTS2


def seconds_to_mel_tokens(seconds, mel_to_sec_ratio=0.02):
    """Convert seconds (scalar or list) to mel token counts."""
    if isinstance(seconds, (list, tuple)):
        return [int(s / mel_to_sec_ratio) for s in seconds]
    return int(seconds / mel_to_sec_ratio)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TED-TTS single-sample inference with six emotion/duration modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--mode", type=int, choices=[0, 1, 2, 3, 4, 5], required=True,
                        help="Inference mode (see header docstring).")
    parser.add_argument("--spk_audio", type=str,
                        default="./datasets/Ref/0011_000001.wav",
                        help="Speaker reference wav (defaults to the bundled clip shipped with the repo).")
    parser.add_argument("--text", type=str,
                        default="Hello, nice to meet you!|But I'm very angry now.|We can get along peacefully.",
                        help="Text to synthesize; segments separated by '|'.")
    parser.add_argument("--output", type=str, default="./results/output.wav",
                        help="Output wav path (default writes under ./results/).")

    parser.add_argument("--emo_audio", type=str, default=None,
                        help="Mode 1: emotion reference wav.")
    parser.add_argument("--emo_alpha", type=float, default=1.0,
                        help="Mode 1: emotion blending weight.")

    parser.add_argument("--emo_vector", type=float, nargs=8, default=None,
                        metavar=("HAP", "ANG", "SAD", "FEA", "HAT", "LOW", "SUR", "NEU"),
                        help="Mode 2: 8-dim emotion vector.")

    parser.add_argument("--emo_text", type=str,
                        default=("happy: cheerful, energetic, bright confident tone"
                                 "|angry: very irritated, tense, louder and faster"
                                 "|neutral: peaceful, soft, steady rhythmal"),
                        help="Modes 3/4/5: per-segment emotion descriptions ('|'-separated).")

    parser.add_argument("--target_seconds", type=float, nargs="+",
                        default=[2.44, 2.12, 1.6],
                        help="Modes 4/5: target seconds per segment (must match text and emo_text segment count).")

    parser.add_argument("--model_dir", type=str, default="./checkpoints",
                        help="Directory containing model checkpoints.")
    parser.add_argument("--cfg_path", type=str, default="./checkpoints/config.yaml",
                        help="Path to config.yaml.")
    parser.add_argument("--is_fp16", action="store_true", default=False,
                        help="Enable FP16 inference.")

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


def main():
    args = parse_args()

    if args.mode == 1 and args.emo_audio is None:
        raise ValueError("Mode 1 requires --emo_audio.")
    if args.mode == 2 and args.emo_vector is None:
        raise ValueError("Mode 2 requires --emo_vector with eight floats.")

    # Modes 0/1/2 operate on a single emotion condition, so '|' in --text must be collapsed.
    if args.mode in (0, 1, 2) and "|" in args.text:
        merged = args.text.replace("|", " ")
        print(f"[mode {args.mode}] merged multi-segment text into one segment:")
        print(f"  before: {args.text}")
        print(f"  after:  {merged}")
        args.text = merged

    if args.mode in (4, 5):
        n_text = len(args.text.split("|"))
        n_emo = len(args.emo_text.split("|"))
        n_sec = len(args.target_seconds)
        if not (n_text == n_emo == n_sec):
            raise ValueError(
                f"Mode {args.mode} requires text, emo_text and target_seconds to share "
                f"the same segment count (got {n_text}, {n_emo}, {n_sec})."
            )

    tts = IndexTTS2(model_dir=args.model_dir, cfg_path=args.cfg_path, is_fp16=args.is_fp16)

    common_kwargs = dict(
        spk_audio_prompt=args.spk_audio,
        text=args.text,
        output_path=args.output,
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
        out = tts.infer(**common_kwargs)

    elif args.mode == 1:
        out = tts.infer(
            **common_kwargs,
            emo_audio_prompt=args.emo_audio,
            emo_alpha=args.emo_alpha,
        )

    elif args.mode == 2:
        # IndexTTS2 expects a list of vectors even for a single segment.
        out = tts.infer(
            **common_kwargs,
            emo_audio_prompt=None,
            emo_alpha=0,
            emo_vector=[args.emo_vector],
        )

    elif args.mode == 3:
        out = tts.infer(
            **common_kwargs,
            emo_audio_prompt=None,
            emo_alpha=0,
            use_emo_text=True,
            emo_text=args.emo_text,
        )

    else:
        target_tokens = seconds_to_mel_tokens(args.target_seconds)
        duration_mode = "global" if args.mode == 4 else "both"
        print(f"[mode {args.mode}] duration_mode={duration_mode}, "
              f"target_seconds={args.target_seconds}, target_tokens={target_tokens}")
        out = tts.infer(
            **common_kwargs,
            emo_audio_prompt=None,
            emo_alpha=0,
            use_emo_text=True,
            emo_text=args.emo_text,
            target_duration_tokens=target_tokens,
            duration_mode=duration_mode,
        )

    print(f"[mode {args.mode}] saved: {out}")


if __name__ == "__main__":
    main()
