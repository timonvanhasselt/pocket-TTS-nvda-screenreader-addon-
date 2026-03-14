"""
generate_bundled_voices.py

Run this script ONCE to generate the .npy voice embeddings for the
8 official Kyutai voices. Place the resulting .npy files in the addon's
voices/ directory before building the .nvda-addon package.

Expected directory layout (no arguments needed):
    %APPDATA%/nvda/pocket_tts/
        generate_bundled_voices.py   <- this script
        tokenizer.model
        onnx/                        <- ONNX models
        voices/                      <- .npy files will be saved here

pocket_tts_onnx.py is loaded from:
    %APPDATA%/nvda/addons/Pocket-TTS/synthDrivers/pocket_tts_onnx/

Requirements:
    pip install onnxruntime numpy soundfile sentencepiece requests scipy

Usage:
    python generate_bundled_voices.py
    python generate_bundled_voices.py --keep-wav
"""

import argparse
import os
import sys
import requests
import numpy as np

VOICES_REPO = "https://huggingface.co/kyutai/tts-voices/resolve/main"

# Name -> (HuggingFace path, license)
# Sources:
#   vctk/          CC BY 4.0  -- enhanced versions used by the Kyutai web demo
#   alba-mackenna/ CC BY 4.0
#   voice-zero/    CC0        -- curated selection from LibriVox
#   ears/          CC BY-NC 4.0  -- non-commercial use only
BUILTIN_VOICES = {
    "jane":           (f"{VOICES_REPO}/vctk/p339_023_enhanced.wav",                    "CC BY 4.0"),
    "alba":           (f"{VOICES_REPO}/alba-mackenna/casual.wav",                       "CC BY 4.0"),
    "bill_boerst":    (f"{VOICES_REPO}/voice-zero/bill_boerst.wav",                    "CC0"),
    "caro_davy":      (f"{VOICES_REPO}/voice-zero/caro_davy.wav",                      "CC0"),
    "peter_yearsley": (f"{VOICES_REPO}/voice-zero/peter_yearsley.wav",                 "CC0"),
    "stuart_bell":    (f"{VOICES_REPO}/voice-zero/stuart_bell.wav",                    "CC0"),
    "anna":           (f"{VOICES_REPO}/vctk/p228_023_enhanced.wav",                    "CC BY 4.0"),
    "azelma":         (f"{VOICES_REPO}/vctk/p303_023_enhanced.wav",                    "CC BY 4.0"),
    "charles":        (f"{VOICES_REPO}/vctk/p254_023_enhanced.wav",                    "CC BY 4.0"),
    "eponine":        (f"{VOICES_REPO}/vctk/p262_023_enhanced.wav",                    "CC BY 4.0"),
    "eve":            (f"{VOICES_REPO}/vctk/p361_023_enhanced.wav",                    "CC BY 4.0"),
    "fantine":        (f"{VOICES_REPO}/vctk/p244_023_enhanced.wav",                    "CC BY 4.0"),
    "george":         (f"{VOICES_REPO}/vctk/p315_023_enhanced.wav",                    "CC BY 4.0"),
    "mary":           (f"{VOICES_REPO}/vctk/p333_023_enhanced.wav",                    "CC BY 4.0"),
    "michael":        (f"{VOICES_REPO}/vctk/p360_023_enhanced.wav",                    "CC BY 4.0"),
    "paul":           (f"{VOICES_REPO}/vctk/p259_023_enhanced.wav",                    "CC BY 4.0"),
    "vera":           (f"{VOICES_REPO}/vctk/p229_023_enhanced.wav",                    "CC BY 4.0"),
    "jean":           (f"{VOICES_REPO}/ears/p010/freeform_speech_01_enhanced.wav",     "CC BY-NC 4.0"),
}


def download_wav(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  Already downloaded: {dest_path}")
        return True
    print(f"  Downloading {os.path.basename(dest_path)}...")
    headers = {"User-Agent": "NVDA-Addon-PocketTTS-voice-generator"}
    try:
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate bundled .npy voice embeddings")
    parser.add_argument("--keep-wav", action="store_true", help="Keep downloaded .wav files")
    args = parser.parse_args()

    # Fixed paths based on the known directory layout:
    #   %APPDATA%\nvda\pocket_tts\   <- this script lives here
    #       tokenizer.model
    #       onnx\
    #       voices\                  <- .npy files saved here
    #   %APPDATA%\nvda\addons\Pocket-TTS\synthDrivers\pocket_tts_onnx\
    #       pocket_tts_onnx.py       <- engine loaded from here
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_dir   = os.path.join(script_dir, "onnx")
    tokenizer  = os.path.join(script_dir, "tokenizer.model")
    output_dir = os.path.join(script_dir, "voices")
    nvda_dir   = os.path.dirname(script_dir)
    synth_dir  = os.path.join(nvda_dir, "addons", "Pocket-TTS", "synthDrivers", "pocket_tts_onnx")

    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "_wav_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Import engine from addon synthDrivers
    if os.path.isdir(synth_dir) and synth_dir not in sys.path:
        sys.path.insert(0, synth_dir)
    try:
        from pocket_tts_onnx import PocketTTSOnnx
    except ImportError:
        print(f"ERROR: Cannot import PocketTTSOnnx.")
        print(f"Expected location: {synth_dir}")
        print("Make sure the Pocket-TTS addon is installed in NVDA.")
        sys.exit(1)

    if not os.path.isdir(onnx_dir):
        print(f"ERROR: ONNX directory not found: {onnx_dir}")
        sys.exit(1)
    if not os.path.exists(tokenizer):
        print(f"ERROR: tokenizer.model not found: {tokenizer}")
        sys.exit(1)

    print(f"Loading ONNX engine from {onnx_dir}...")
    tts = PocketTTSOnnx(
        models_dir=onnx_dir,
        tokenizer_path=tokenizer,
        precision="int8",
        lsd_steps=1,
    )
    print("Engine loaded.\n")

    results = []
    for name, (url, license_) in BUILTIN_VOICES.items():
        npy_path = os.path.join(output_dir, f"{name}.npy")
        wav_path = os.path.join(tmp_dir, f"{name}.wav")

        print(f"[{name}]  ({license_})")

        if os.path.exists(npy_path) and os.path.getsize(npy_path) > 0:
            print(f"  {npy_path} already exists, skipping.\n")
            results.append((name, True, license_))
            continue

        if not download_wav(url, wav_path):
            results.append((name, False, license_))
            print()
            continue

        print(f"  Generating embedding...")
        try:
            embedding = tts.encode_voice(wav_path)
            np.save(npy_path, embedding)
            size_kb = os.path.getsize(npy_path) / 1024
            print(f"  Saved {npy_path}  ({size_kb:.0f} KB, shape {embedding.shape})")
            results.append((name, True, license_))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False, license_))

        if not args.keep_wav and os.path.exists(wav_path):
            os.remove(wav_path)

        print()

    # Clean up tmp dir
    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    # Summary
    print("=" * 50)
    print("Summary:")
    ok  = [(n, l) for n, s, l in results if s]
    err = [(n, l) for n, s, l in results if not s]
    for name, lic in ok:
        print(f"  OK   {name:<10} {lic}")
    for name, lic in err:
        print(f"  FAIL {name:<10} {lic}")
    print()
    if ok:
        print(f"{len(ok)} .npy files saved to: {output_dir}")
        print("Copy them into the addon's voices/ directory before packaging.")
    if err:
        print(f"\n{len(err)} voice(s) failed. Check errors above.")


if __name__ == "__main__":
    main()