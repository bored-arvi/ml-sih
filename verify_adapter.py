from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

TRIAL_FILENAME = "trial color mismatch.png"
TRIAL_HEATMAP = "CERT-001_deltaE_heatmap.png"


def build_trial_response() -> Dict[str, Any]:
    return {
        "db_fields_text": "Yuvarai B Digital Marketing Workshop Wardiere Company 6 June 2025",
        "extracted_text": "CERTIFICATE OF PARTICIPATION This certificate is proudly presented to Yuvarai B For participating in the Digital Marketing Workshop held by Wardiere Company on 6 June 2025. l RANDOM SIGNATURE Just for testing purposes: Not valid for official purposes.",
        "heatmap_path": TRIAL_HEATMAP,
        "ner_fields": {
            "date_ner": "6 June 2025",
            "issuer_ner": "Yuvarai B For",
        },
        "non_db_text": "CERTIFICATE OF PARTICIPATION This certificate is proudly presented to For participating in the held by on 2025. l RANDOM SIGNATURE Just for testing purposes: Not valid for official purposes.",
        "noticeable_color_fraction": 0.11065673828125,
        "regex_fields": {
            "date": "6 June 2025",
            "event": "Digital Marketing Workshop",
            "issuer": "Wardiere Company",
            "name": "Yuvarai B",
        },
        "similarities": {
            "color_similarity": 0.9980857372283936,
            "combined_similarity": 0.9979523420333862,
            "db_field_score": 1.0000001192092896,
            "noticeable_color_fraction": 0.11065673828125,
            "text_similarity": 0.9978191256523132,
        },
    }


def run_verify_pipeline(image_path: str, certificate_id: str, output_dir: str) -> Dict[str, Any]:
    """
    Thin adapter around a local verify.py if present; otherwise raise a clear error.
    Expected functions in verify.py:
      - extract_text(path)
      - extract_fields_regex(text)
      - extract_fields_ner(text)
      - combine_db_fields(regex_fields, ner_fields)
      - extract_non_db_text(extracted_text, db_fields_text)
      - color_embedding(path)
      - compare_against_stored(certificate_id, new_non_db_text, new_color_emb, new_db_fields_text, new_image_path)
      - deltaE_heatmap(ref_image_path, new_image_path, output_path)
    """
    base_dir = Path(__file__).resolve().parent
    verify_path = base_dir / "verify.py"

    if not verify_path.exists():
        raise FileNotFoundError(
            "verify.py not found. Provide your verification pipeline at ML/verify.py or adjust the adapter."
        )

    import importlib.util

    spec = importlib.util.spec_from_file_location("verify", str(verify_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load verify.py module spec")
    verify = importlib.util.module_from_spec(spec)  # type: ignore[assignment]
    spec.loader.exec_module(verify)  # type: ignore[union-attr]

    extracted_text = verify.extract_text(image_path)
    regex_fields = verify.extract_fields_regex(extracted_text)
    ner_fields = verify.extract_fields_ner(extracted_text)
    db_fields_text = verify.combine_db_fields(regex_fields, ner_fields)
    non_db_text = verify.extract_non_db_text(extracted_text, db_fields_text)
    new_color_emb = verify.color_embedding(image_path)

    scores = verify.compare_against_stored(
        certificate_id=certificate_id,
        new_non_db_text=non_db_text,
        new_color_emb=new_color_emb,
        new_db_fields_text=db_fields_text,
        new_image_path=image_path,
    )

    heatmap_path = None
    noticeable_fraction = None
    ref_image_path = str((base_dir / f"{certificate_id}_image.png").resolve())
    output_heatmap = Path(output_dir) / f"{certificate_id}_deltaE_heatmap.png"
    if Path(ref_image_path).exists():
        noticeable_fraction, heatmap_path = verify.deltaE_heatmap(
            ref_image_path,
            image_path,
            output_path=str(output_heatmap),
        )

    return {
        "extracted_text": extracted_text,
        "regex_fields": regex_fields,
        "ner_fields": ner_fields,
        "db_fields_text": db_fields_text,
        "non_db_text": non_db_text,
        "similarities": scores,
        "noticeable_color_fraction": noticeable_fraction,
        "heatmap_path": str(heatmap_path) if heatmap_path else None,
    }


