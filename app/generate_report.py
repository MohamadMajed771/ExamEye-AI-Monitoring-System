import sqlite3
import os
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch


DB_PATH = "database/events.db"
REPORTS_DIR = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)

report_path = os.path.join(
    REPORTS_DIR,
    "exam_report.pdf"
)

doc = SimpleDocTemplate(
    report_path,
    pagesize=letter
)

styles = getSampleStyleSheet()

elements = []

title = Paragraph(
    "ExamEye AI Monitoring Report",
    styles['Title']
)

elements.append(title)
elements.append(Spacer(1, 0.3 * inch))


conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT student_id,
           event_type,
           timestamp,
           confidence,
           screenshot_path
    FROM events
    ORDER BY timestamp DESC
""")

events = cursor.fetchall()

conn.close()


for event in events:

    student_id, event_type, timestamp, confidence, screenshot_path = event

    text = f"""
    <b>Student ID:</b> {student_id}<br/>
    <b>Event:</b> {event_type}<br/>
    <b>Timestamp:</b> {timestamp}<br/>
    <b>Confidence:</b> {confidence}
    """

    paragraph = Paragraph(text, styles['BodyText'])

    elements.append(paragraph)
    elements.append(Spacer(1, 0.2 * inch))

    if os.path.exists(screenshot_path):

        img = Image(
            screenshot_path,
            width=4 * inch,
            height=3 * inch
        )

        elements.append(img)
        elements.append(Spacer(1, 0.5 * inch))

    elements.append(PageBreak())


doc.build(elements)

print(f"Report generated: {report_path}")