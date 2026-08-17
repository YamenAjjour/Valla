import os
import subprocess

NEW_ROOT = "reddit"
TRAIN_NGRAM =  "/home/lotta_kiefer/alias_share/Valla/valla/methods/FeatureDifferenceGerman.py"
TRAIN_PPM = "/home/lotta_kiefer/alias_share/Valla/valla/methods/PPM_AV.py"
TRAIN_ADH = "/home/lotta_kiefer/alias_share/Valla/valla/methods/torched_AdHominem.py"
TRAIN_SBERT = "/home/lotta_kiefer/alias_share/Valla/valla/methods/SiameseBert.py"
OUT = "/home/lotta_kiefer/alias_share/valla/results"


def disable_wandb(env):
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"
    env["WANDB_CONSOLE"] = "off"
    env["WANDB_DISABLED"] = "true"

def run_training(script_path, train_path, test_path, out_path, mode):
    script_dir = os.path.dirname(os.path.abspath(script_path))
    out_path = out_path + os.sep
    env = os.environ.copy()
    disable_wandb(env)
    env["PYTHONPATH"] = "/home/lotta_kiefer/alias_share/Valla"

    if mode=="ngram":
        cmd = [
            "python",
            script_path,
            "--train_path", train_path,
            "--test_path", test_path,
            "--out_path", out_path
        ]
    if mode=="ppm":
        cmd = [
            "python",
            script_path,
            "--train_path", train_path,
            "--test_path", test_path,
            "--cache_path", out_path,
            "--dset_name", "ppm"
        ]
    if mode=="adh":
        cmd = [
            "python",
            script_path,
            "--train_path", train_path,
            "--test_path", test_path,
            "--save_model_dir", out_path,
            "--dont_use_fasttext"
        ]
    if mode=="sbert":
        cmd = [
            "python",
            script_path,
            "--train_path", train_path,
            "--test_path", test_path,
            "--output_path", out_path,
        ]

    print(f"Running: {' '.join(cmd)} (PYTHONPATH={script_dir})")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True, check=True, env=env
        )
    except subprocess.CalledProcessError as e:
        print("Error")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)

def main():
        train_twitter_path = "/home/lotta_kiefer/alias_share/data/twitter/train.jsonl"
        test_twitter_path = "/home/lotta_kiefer/alias_share/data/twitter/test.jsonl"
        val_twitter_path = "/home/lotta_kiefer/alias_share/data/twitter/validation.jsonl"
        ngram_dir = os.path.join(OUT, "twitter", "ngram")
        ppm_dir = os.path.join(OUT, "twitter", "ppm")
        adh_dir = os.path.join(OUT, "twitter", "adh")
        sbert_dir = os.path.join(OUT, "twitter_new", "sbert")
        os.makedirs(ngram_dir, exist_ok=True)
        os.makedirs(ppm_dir, exist_ok=True)
        os.makedirs(adh_dir, exist_ok=True)
        os.makedirs(sbert_dir, exist_ok=True)
        run_training(TRAIN_NGRAM, train_twitter_path, test_twitter_path, ngram_dir, "ngram")
        run_training(TRAIN_PPM, train_twitter_path, test_twitter_path, ppm_dir, "ppm")
        run_training(TRAIN_ADH, train_twitter_path, val_twitter_path, adh_dir, "adh")
        run_training(TRAIN_SBERT, train_twitter_path, val_twitter_path, sbert_dir, "sbert")


if __name__ == "__main__":
    main()