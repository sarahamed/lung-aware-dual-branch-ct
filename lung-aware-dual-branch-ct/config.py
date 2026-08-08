"""
Central configuration for the lung-aware dual-branch CT classification pipeline.

All paths are environment-variable driven so the code runs unmodified on any
machine (local, Colab, HPC). Override via environment variables rather than
editing this file:

    OUTPUT_ROOT   Where run folders (checkpoints/logs/outputs) are written.
                  Default: ./ct_lungcancer_experiments
    RUN_NAME      Name of a specific run folder under OUTPUT_ROOT. Default is
                  the run identifier used for the manuscript's reported
                  results; override to point at your own trained checkpoint.
    MOUNT_DRIVE   "1" (default) to auto-mount Google Drive when running in
                  Colab. Set to "0" to skip.
"""
import os


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


IN_COLAB = in_colab()

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "./ct_lungcancer_experiments")
RUN_NAME = os.environ.get("RUN_NAME", "ModelA_LungROI_CrossAttn_20260114_133341")

BASE_DIR = os.path.join(OUTPUT_ROOT, RUN_NAME)
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
REVISION_DIR = os.path.join(BASE_DIR, "revision_outputs")

# Reproducibility
SEED = 42

# Data / model shape
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 3

# Folder names on disk (IQ-OTHNCCD dataset) vs. display names used in reporting
CLASS_FOLDERS = ["Bengin cases", "Malignant cases", "Normal cases"]
CLASS_NAMES = ["Benign", "Malignant", "Normal"]

# Manuscript headline reference numbers, used by evaluation/reproduction checks
REPORTED_TEST_ACCURACY = 96.36  # percent
ACCURACY_TOLERANCE_PP = 0.5  # percentage points

# Status tag carried over from the manuscript's revision process (results
# generated before the leakage-free split correction are tagged with this).
STATUS_TAG = "[PROVISIONAL - pre-split-correction]"


def tag(msg: str) -> str:
    """Prefix a printed summary/headline with the provisional-split status tag."""
    return f"{STATUS_TAG} {msg}"


def tagged_name(filename: str) -> str:
    """Prefix an output filename's stem with the provisional tag (filesystem-safe)."""
    stem, ext = os.path.splitext(filename)
    return f"{stem}__PROVISIONAL-pre-split-correction{ext}"


def ensure_dirs():
    for d in (CKPT_DIR, LOG_DIR, OUT_DIR, REVISION_DIR):
        os.makedirs(d, exist_ok=True)


def mount_drive_if_colab():
    if IN_COLAB and os.environ.get("MOUNT_DRIVE", "1") == "1":
        from google.colab import drive
        drive.mount("/content/drive")
