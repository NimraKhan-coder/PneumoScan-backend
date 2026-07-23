"""
Generates a professional PDF report for a single X-ray prediction,
matching the fields shown on the app's X-Ray Details screen.
"""

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

PRIMARY_COLOR = HexColor("#2196F3")
RISK_COLORS = {
    "high": HexColor("#D32F2F"),
    "moderate-high": HexColor("#F57C00"),
    "moderate": HexColor("#FBC02D"),
    "low-moderate": HexColor("#689F38"),
    "low": HexColor("#388E3C"),
}


def generate_xray_report_pdf(record: dict, xray_image_bytes: bytes | None) -> bytes:
    """
    record: the prediction row (id, prediction, confidence, risk_level,
            explanation, recommendation, created_at, patient_id, etc.)
    xray_image_bytes: raw bytes of the X-ray image, or None if unavailable
    Returns: raw PDF bytes.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=20, textColor=PRIMARY_COLOR,
    )
    label_style = ParagraphStyle(
        "LabelStyle", parent=styles["Normal"], fontSize=9, textColor=HexColor("#666666"),
        spaceBefore=10,
    )
    value_style = ParagraphStyle(
        "ValueStyle", parent=styles["Normal"], fontSize=13, spaceAfter=2,
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerStyle", parent=styles["Normal"], fontSize=8,
        textColor=HexColor("#888888"), alignment=TA_CENTER,
    )

    elements = []

    elements.append(Paragraph("PneumoScan AI — X-Ray Report", title_style))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(f"Report #{record['id'][:8]}", styles["Heading3"]))
    elements.append(Spacer(1, 0.5 * cm))

    # X-ray image, if available
    if xray_image_bytes:
        try:
            img_buffer = BytesIO(xray_image_bytes)
            img = RLImage(img_buffer, width=10 * cm, height=10 * cm, kind="proportional")
            img.hAlign = "CENTER"
            elements.append(img)
            elements.append(Spacer(1, 0.7 * cm))
        except Exception:
            pass  # if image fails to embed, still generate the rest of the report

    risk_level = record.get("risk_level", "").lower()
    risk_color = RISK_COLORS.get(risk_level, HexColor("#333333"))

    def field(label, value, value_color=None):
        elements.append(Paragraph(label.upper(), label_style))
        style = value_style
        if value_color:
            style = ParagraphStyle(
                "ColoredValue", parent=value_style, textColor=value_color,
            )
        elements.append(Paragraph(str(value), style))

    field("Patient ID", record.get("patient_id", "N/A"))
    field("Prediction", record.get("prediction", "N/A"))
    field("Confidence", f"{record.get('confidence', 0)}%")
    field("Risk Level", record.get("risk_level", "N/A"), value_color=risk_color)
    field("Explanation", record.get("explanation", "N/A"))
    field("Recommendation", record.get("recommendation", "N/A"))
    field("Created At", record.get("created_at", "N/A"))

    elements.append(Spacer(1, 1 * cm))
    elements.append(Paragraph(
        "This is an AI-assisted screening tool and not a substitute for professional "
        "medical diagnosis. Always consult a qualified doctor for confirmation and "
        "treatment decisions.",
        disclaimer_style,
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()