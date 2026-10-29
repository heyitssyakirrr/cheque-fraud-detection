"""Application configuration.

One flat, validated settings object, loaded from `.env`. 
Fields are grouped below by which part of the pipeline they configure:

    1. Paths & environment          -- where things live on disk
    2. AI Foundry                   -- app/services/foundry_client.py
       Signature crop               -- app/services/signature_detection.py
    3. Date field location (OCR)    -- app/services/date_extraction.py
    4. Date digits                  -- app/services/date_extraction.py
    5. Preprocessing                -- app/services/preprocessing.py
    6. Legacy single-cheque mode    -- app/main.py (Upload/ folder + one CSV)
    7. Batch ZIP ingestion          -- run.py, app/ingestion.py, app/manifest.py
    8. Worker process sizing        -- run.py's ProcessPoolExecutor

Every field can be overridden by an environment variable of the same name
(case-insensitive), or by editing `.env`. 
"""

import math
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # Absolute, so the .env is found regardless of the launch directory.
    model_config = SettingsConfigDict(env_file=_ROOT / ".env", extra="ignore")

    # ------------------------------------------------------------------
    # 1. Paths & environment
    # ------------------------------------------------------------------
    root: Path = _ROOT

    max_upload_mb: int = 15
    """Upper bound for a single cheque image accepted by the web app."""

    # ------------------------------------------------------------------
    # 2. AI Foundry (signature and date reading)
    # ------------------------------------------------------------------
    llm_base_url: str = ""
    """AI Foundry base address, e.g. https://<foundry-host>. Required."""

    llm_chat_path: str = "/foundry/chat/completions"
    """Appended to `llm_base_url` for every request."""

    llm_model: str = "gpt-6-luna"

    helper_id: str = ""
    """Sent as the X-PBAI-Helper-Id header; blank = header not sent."""

    llm_max_tokens: int = 4000
    """Output limit per request. gpt-6-luna thinks before answering and that
    counts against this limit, so it must be well above the ~50 tokens the
    JSON answer itself needs (a low value returns an empty, cut-off reply)."""

    llm_timeout_seconds: float = 120.0

    llm_verify_ssl: bool = False
    """Whether to check Foundry's HTTPS certificate. Off to match the other
    internal projects, because the company CA is not in Python's trust
    list. Turning it on (once the CA is installed on the server) is safer:
    with it off, the crops go to whatever server answers at that address."""

    llm_max_attempts: int = 3
    """Attempts per request for network errors, HTTP 429 and 5xx. A request
    blocked by the platform guardrail is never retried."""

    llm_retry_backoff_seconds: float = 2.0
    """Wait before retry n is n times this many seconds."""

    llm_image_max_long_edge: int = 1024
    """Crops are downscaled to at most this many pixels on the long edge
    before sending -- plenty for a signature zone or date boxes."""

    # ------------------------------------------------------------------
    # Signature crop (what gets sent to AI Foundry)
    # ------------------------------------------------------------------
    signature_max_long_edge: int = 1600
    """Resize cap for the image the signature zone is cut from. Kept separate
    from `preprocess_max_long_edge`, which date_extraction.py's geometry is
    calibrated against."""

    signature_zone_left_fraction: float = 0.30
    signature_zone_top_fraction: float = 0.55
    signature_zone_bottom_fraction: float = 0.87
    """Fixed-fraction crop window (of the resized signature image) where a
    signature reliably sits on this cheque template -- generous on purpose.
    This crop is exactly what is sent to AI Foundry, so keep it to the
    signing area: the less of the cheque it shows (names, account number,
    MICR line), the less personal data leaves the server. Check
    signature/annotated.png to see the zone on real cheques."""

    # ------------------------------------------------------------------
    # 3. Date field location (OCR)
    # ------------------------------------------------------------------
    date_width_reference_edge: int = 2200
    """Reference width that all date-field measurements below are expressed
    against, so date geometry remains stable when preprocessing rescales a
    cheque to a different size."""

    date_label_aliases_tarikh: tuple[str, ...] = (
        "TARIKH", "TARlKH", "TARlK", "TARKH", "TARK",
    )
    date_label_aliases_date: tuple[str, ...] = (
        "DATE", "DATF", "D4TE", "DAE", "BDATE",
    )
    """TARIKH and "日DATE" are two separate, stacked printed lines (TARIKH on
    top, DATE beneath it) -- not one label. Kept as two alias groups (common
    OCR misreads of each) so the crop geometry can key off *which* line was
    actually found -- see `date_extraction._find_label_lines` /
    `_vertical_bounds_from_labels` -- instead of treating either match the
    same way."""

    date_label_aliases_ambiguous: tuple[str, ...] = ("WN",)
    """WN is a known but weak corruption of either line; only trusted in the
    normal upper-right label area (see the `weak_wn` check in
    `date_extraction._score_candidates`)."""

    date_label_line_gap_ratio: float = 0.2
    """Vertical crop bound = TARIKH's own top edge to DATE's own bottom edge
    (see `_vertical_bounds_from_labels` docstring for why). If only one line
    is found, the other's edge is estimated as this many typical
    single-line-heights away, plus this calibrated gap below."""

    date_field_vertical_safety_pad_ratio: float = 0.15
    """Small safety margin added above/below the measured label span, since
    handwritten digits can slightly overshoot the printed label's own
    height. Deliberately much smaller than the old fixed multipliers --
    those existed to compensate for using only one, possibly-wrong edge;
    with both edges measured directly this only needs to cover natural
    handwriting variance."""

    date_field_bottom_extra_pad_ratio: float = 0.35
    """Extra padding added only below DATE's own bottom edge, on top of the
    symmetric safety pad above. Handwritten descenders and closed loops (a
    looped '6' or '2') extend further down than a printed line's own bottom
    edge predicts -- asymmetric on purpose; the top edge needs no
    equivalent since TARIKH's top edge has no handwriting above it."""

    date_field_width_typical_heights: float = 17.0
    """Horizontal crop width, expressed in multiples of the document's
    typical single-line text height (a stable, format-invariant reference --
    see `_typical_line_height`), not the anchor label's own box height."""

    date_field_max_width_fraction: float = 0.55
    date_field_max_height_fraction: float = 0.32
    """Hard ceiling on the date crop, independent of anything OCR reports,
    as a last line of defense against any future anchor-measurement error.
    Expressed as a fraction of the (already format-normalised) page size."""

    ocr_zone_left_fraction: float = 0.40
    ocr_zone_top_fraction: float = 0.00
    ocr_zone_bottom_fraction: float = 0.45
    """Region OCR is restricted to when locating the TARIKH/DATE labels.
    Deliberately wider than `_template_date_zone` (0.60-0.99 x, 0.10-0.38 y)
    so the printed labels and enough neighbouring lines stay in frame for
    `_typical_line_height` to remain representative."""

    date_use_template_zone: bool = False
    """Debug aid only. When true, skips OCR and takes the date field from
    `_template_date_zone`'s fixed fractions -- wrong for real input, since
    cheques arrive from several banks at varying sizes and the date is not
    at a fixed offset. Finding the TARIKH/DATE labels is what makes the crop
    portable, so leave this off in production."""

    # ------------------------------------------------------------------
    # 4. Date digits
    # ------------------------------------------------------------------
    digit_count: int = 6
    """Expected number of handwritten date digits per cheque (DDMMYY). Used
    in the date prompt and to validate the model's reading."""

    # ------------------------------------------------------------------
    # 5. Preprocessing
    # ------------------------------------------------------------------
    preprocess_max_long_edge: int = 3000
    preprocess_contrast: float = 1.15
    preprocess_deskew: bool = True
    """These replace the sliders the removed web form used to supply."""

    preprocess_denoise: bool = False
    """Non-local-means denoising costs roughly a second per cheque and only
    feeds the OCR/date path. Off by default; turn on if date accuracy needs
    it."""

    # ------------------------------------------------------------------
    # 6. Legacy single-cheque mode (app/main.py)
    # ------------------------------------------------------------------
    upload_dir_name: str = "Upload"
    output_csv_name: str = "output.csv"
    """Folder scanned for individual cheque images, and the single results
    file the web app writes to. Unrelated to the batch ZIP pipeline below."""

    ground_truth_path: str = "data/ground_truth.csv"
    """CSV of known-correct answers for cheques you run repeatedly, so
    accuracy scoring doesn't require re-pasting ground truth every batch.
    Columns: filename, signature_count, date (DDMMYY or "no date")."""

    save_artifacts: bool = True
    """Whether to write per-cheque debug output (annotated images, crops,
    digit slices, CSVs, result.json) -- roughly 20 files per cheque. Leave
    off for production volume; turn on when investigating a specific
    batch."""

    # ------------------------------------------------------------------
    # 7. Batch ZIP ingestion (run.py, app/ingestion.py, app/manifest.py)
    # ------------------------------------------------------------------
    incoming_transport: str = "local"
    """Where the daily ZIP is fetched from. Only "local" is implemented;
    the validator below fails fast on anything else. Local is deliberately
    the default so the exact production code path can be tested on a
    laptop without SFTP."""

    incoming_root: str = "data/incoming"
    """Directory the runner looks in for the daily ZIP."""

    uat_image_folder: str = "/pythonITD/dwh_312/Cheque_Fraud_Detection/img"
    """UAT only: when set, run.py reads PNG/JPEG/TIFF files from this folder
    instead of the daily ZIP. Set UAT_IMAGE_FOLDER= (empty) in .env for
    production so the ZIP is used. A relative path is resolved from the
    project root."""

    batch_zip_name_pattern: str = "{date}_cheques.zip"
    """Expected filename for a given `--batch-id`, e.g. "20260916_cheques.zip"."""

    staging_dir_name: str = "data/staging"
    """Where the ZIP is copied to before processing (see `batch_zip_path`)."""

    max_attempts: int = 3
    """Total attempts per cheque before it's marked permanently failed --
    this counts attempts, not retries (so the default allows 2 retries
    after an initial failure)."""

    cleanup_zip_after_batch: bool = True
    """Delete the staged ZIP copy once every cheque in the batch is DONE or
    PERMANENT_FAILED. The original source ZIP is never touched."""

    batch_inflight_multiplier: int = 2
    """How many futures to keep queued per worker (`workers *
    batch_inflight_multiplier`), so the runner never holds all ~20,000 tasks
    in memory at once."""

    # ------------------------------------------------------------------
    # 8. Worker process sizing (run.py's ProcessPoolExecutor)
    # ------------------------------------------------------------------
    worker_count: int = 0
    """Fixed worker process count. 0 means auto-size from the logical-core
    budget below instead."""

    worker_utilization_target: float = 0.40
    """Fraction of *usable* logical cores (after reserving
    `reserved_logical_cores`) to spend on workers when `worker_count=0`.
    Example: 128 logical cores * 0.40 target - 8 reserved = 43 workers."""

    reserved_logical_cores: int = 8
    """Logical cores left for the OS / other processes on a shared server
    when auto-sizing workers."""

    max_worker_cap: int = 0
    """Optional hard ceiling applied after auto-sizing. 0 disables the
    cap."""

    threads_per_worker: int = 1
    """Threads each worker's native libraries (OpenMP/MKL/BLAS) may use. 0
    leaves them unrestricted -- meaning every worker starts its own
    16-thread pool, so e.g. 25 workers end up fighting over 25 cores with
    400 spinning threads between them. Keep
    `worker_count * threads_per_worker` at or under the core count."""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _validate_worker_budget(self) -> "Settings":
        if self.worker_count < 0:
            raise ValueError("worker_count must be >= 0")
        if not (0.05 <= self.worker_utilization_target <= 1.0):
            raise ValueError("worker_utilization_target must be between 0.05 and 1.0")
        if self.reserved_logical_cores < 0:
            raise ValueError("reserved_logical_cores must be >= 0")
        if self.max_worker_cap < 0:
            raise ValueError("max_worker_cap must be >= 0")
        return self

    @model_validator(mode="after")
    def _validate_batch_ingestion(self) -> "Settings":
        if self.incoming_transport != "local":
            raise ValueError(
                "Only incoming_transport='local' is implemented; "
                "add SFTP after its details are confirmed"
            )
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.batch_inflight_multiplier < 1:
            raise ValueError("batch_inflight_multiplier must be >= 1")
        return self

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    @property
    def upload_dir(self) -> Path:
        return self.root / self.upload_dir_name

    @property
    def output_csv(self) -> Path:
        return self.root / self.output_csv_name

    @property
    def incoming_dir(self) -> Path:
        value = Path(self.incoming_root)
        return value if value.is_absolute() else self.root / value

    @property
    def uat_image_dir(self) -> Path | None:
        if not self.uat_image_folder.strip():
            return None
        value = Path(self.uat_image_folder)
        return value if value.is_absolute() else self.root / value

    @property
    def staging_dir(self) -> Path:
        return self.root / self.staging_dir_name

    @property
    def batches_dir(self) -> Path:
        return self.root / "data" / "batches"

    def batch_dir(self, batch_id: str) -> Path:
        return self.batches_dir / batch_id

    def batch_zip_path(self, batch_id: str) -> Path:
        return self.staging_dir / f"{batch_id}_cheques.zip"

    @property
    def _ocr_models_base(self) -> Path:
        p = Path(self.ocr_models_dir)
        return p if p.is_absolute() else (self.root / p).resolve()

    ocr_models_dir: str = "../FileReader/models"
    """Path to the shared FileReader models/ folder, so the PaddleOCR
    weights live in one place instead of being duplicated into every
    project that needs them. Set in .env."""

    @property
    def ocr_det_dir(self) -> Path:
        return self._ocr_models_base / "en_PP-OCRv3_det_infer"

    @property
    def ocr_rec_dir(self) -> Path:
        return self._ocr_models_base / "en_PP-OCRv3_rec_infer"

    @property
    def ocr_cls_dir(self) -> Path:
        return self._ocr_models_base / "ch_ppocr_mobile_v2.0_cls_infer"

    # ------------------------------------------------------------------
    # Worker sizing logic
    # ------------------------------------------------------------------
    def recommend_workers(self, logical_cores: int, task_count: int) -> int:
        """Compute the worker count from either fixed or auto mode."""
        if task_count <= 0:
            return 1

        if self.worker_count > 0:
            requested = self.worker_count
        else:
            usable = max(1, logical_cores - self.reserved_logical_cores)
            requested = max(1, math.floor(usable * self.worker_utilization_target))

        if self.max_worker_cap > 0:
            requested = min(requested, self.max_worker_cap)

        return max(1, min(requested, task_count))


settings = Settings()