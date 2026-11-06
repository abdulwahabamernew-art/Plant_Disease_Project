"""Gradio interface for the guarded plant-disease pipeline.

The interface keeps the plant gate visible, separates reviewable evidence from
predictions, and never presents model confidence as diagnostic accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from src.image_io import load_rgb
from src.pipeline import Pipeline
from src.ui_presenter import present
from src.visualization import annotate

UI = Path(__file__).resolve().parent
_pipeline: Any = None


def get_pipeline() -> Pipeline:
    """Load the inference pipeline lazily so the UI can render before model load."""
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def _diagnostic_root() -> Path:
    root = UI / "outputs" / "diagnostic"
    root.mkdir(parents=True, exist_ok=True)
    return root


def detector_only(source: str | Path, progress=None):
    """Run the safety gate and leaf detector without disease classification.

    This opt-in diagnostic is useful for checking detector behavior. If the gate
    rejects the input, the disease classifier is deliberately never called.
    """
    path = Path(source)
    image = load_rgb(path)
    pipeline = get_pipeline()
    gate = pipeline.classifier.plant(image)
    detection = pipeline.detector.predict(path)
    disease_done = False
    if gate["label"] == "not_plant":
        summary = (
            "### تم إيقاف التحليل بأمان\n\n"
            "لم تتعرف بوابة النموذج على نبات في الصورة؛ لذلك هذه النتيجة لا تتضمن تشخيص مرض. "
            "قد تخطئ البوابة، فجرّب صورة أقرب وأوضح للأوراق."
        )
    elif gate["label"] == "uncertain":
        summary = (
            "### البوابة غير متأكدة\n\n"
            "تم تشغيل كشف الأوراق فقط. لا تتضمن هذه النتيجة تشخيص مرض، وتحتاج "
            "إلى مراجعة بصرية قبل أي تصنيف."
        )
    else:
        summary = (
            "### تم تشغيل التشخيص البصري فقط\n\n"
            "هذه أداة فحص للكاشف وليست نتيجة مرضية نهائية."
        )

    root = _diagnostic_root()
    record = {
        "input": str(path),
        "plant_gate": gate,
        "detection": detection,
        "disease_classification_performed": disease_done,
        "note": "Detector-only diagnostic; no disease diagnosis is produced.",
    }
    report = root / f"{path.stem}_diagnostic.json"
    report.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return None, summary, (str(path), str(report))


def run_analysis(source: str | Path, progress=None):
    """Run the full pipeline and return UI-ready values."""
    if not source:
        return None, "", [], [], None, ""
    result = get_pipeline().predict(source, save=True)
    summary, rows, gallery, top3 = present(result)
    annotated = result.get("annotated_image")
    return annotated, summary, rows, gallery, result.get("json_path"), json.dumps(result, ensure_ascii=False, indent=2)


def preview(source):
    """Clear analysis outputs when the uploaded image is removed.

    The stable tuple shape is intentionally kept small and testable for the UI.
    """
    if source:
        return (source, "", [], [], None, "", None, {"interactive": True}, None, "", None)
    return (None, "", [], [], None, "", None, {"interactive": False}, None, "", None)


def clear_preview():
    """Return the nine values bound to the visible Gradio components."""
    return (None, "", [], [], None, "", None, {"interactive": False}, {"interactive": False})


def build_demo():
    """Build the Gradio Blocks interface; imports Gradio only when requested."""
    import gradio as gr

    css = """
    .hero { text-align: center; padding: 18px 10px 8px; }
    .hero h1 { color: #166534; margin-bottom: 4px; }
    .note { border-left: 4px solid #eab308; padding: 10px 14px; background: #fffbeb; }
    """
    with gr.Blocks(title="Plant Disease Project", css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🌿 Plant Disease Project\n"
            "### بوابة نبات آمنة + كشف أوراق + تصنيف 38 حالة مرضية",
            elem_classes="hero",
        )
        gr.Markdown(
            "ارفع صورة واضحة لورقة نبات. النظام يمرر الصورة أولاً عبر **plant gate**، "
            "ويعرض الأدلة والنتيجة التجريبية مع علامات المراجعة. الثقة ليست دقة تشخيصية.",
            elem_classes="note",
        )
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Image(type="filepath", label="📷 صورة الإدخال")
                analyze = gr.Button("🔍 تحليل الصورة", variant="primary")
                diagnostic = gr.Button("🧪 فحص البوابة والكاشف فقط")
            with gr.Column(scale=2):
                annotated = gr.Image(label="🖼️ الصورة المشروحة")
                status = gr.Markdown()
        with gr.Tab("📋 ملخص الأوراق"):
            table = gr.Dataframe(headers=["ID", "النبات", "الحالة", "الثقة", "المراجعة"], interactive=False)
            gallery = gr.Gallery(label="قصاصات الأوراق", columns=3, height="auto")
        with gr.Tab("📦 الأدلة والـ JSON"):
            json_path = gr.File(label="ملف metrics JSON")
            raw = gr.Code(label="نتيجة JSON", language="json", interactive=False)
        analyze.click(run_analysis, inputs=source, outputs=[annotated, status, table, gallery, json_path, raw])
        diagnostic.click(detector_only, inputs=source, outputs=[annotated, status, json_path])
        source.clear(
            clear_preview,
            outputs=[source, status, table, gallery, json_path, raw, annotated, analyze, diagnostic],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
