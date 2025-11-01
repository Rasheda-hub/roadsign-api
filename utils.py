# utils.py
from typing import Any, Dict, List, Optional

def normalize_roboflow_response(rf_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize various Roboflow community model output formats into a list of detections:
    Each detection: { label, confidence, x, y, width, height, raw }
    Coordinates are returned as center x,y and width,height if available.
    """
    detections = []
    # common keys Roboflow uses: 'predictions', 'preds', 'objects'
    preds = rf_json.get("predictions") or rf_json.get("preds") or rf_json.get("objects") or []

    for p in preds:
        # label
        label = p.get("class") or p.get("label") or p.get("name")

        # confidence / score
        conf = p.get("confidence") or p.get("score") or p.get("confidence_score") or None
        try:
            conf = float(conf) if conf is not None else None
        except Exception:
            conf = None

        # try common bbox representations
        x = p.get("x") or p.get("center_x") or p.get("cx") or None
        y = p.get("y") or p.get("center_y") or p.get("cy") or None
        w = p.get("width") or p.get("w") or None
        h = p.get("height") or p.get("h") or None

        # sometimes bbox is [x1,y1,x2,y2]
        bbox = p.get("bbox") or p.get("box") or p.get("bounding_box")
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            try:
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
            except Exception:
                pass

        # fallback: some models return absolute pixel bbox as dictionary
        if isinstance(bbox, dict):
            x1 = bbox.get("x1") or bbox.get("left")
            y1 = bbox.get("y1") or bbox.get("top")
            x2 = bbox.get("x2") or bbox.get("right")
            y2 = bbox.get("y2") or bbox.get("bottom")
            if None not in (x1, y1, x2, y2):
                try:
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                except Exception:
                    pass

        detections.append({
            "label": label,
            "confidence": conf,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "raw": p
        })

    return detections
