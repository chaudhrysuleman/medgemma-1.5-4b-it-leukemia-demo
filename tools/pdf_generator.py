"""
PDF Report Generator Tool
Creates professional, downloadable PDF reports from analysis results
"""

import os
import re
from datetime import datetime
from fpdf import FPDF


class LeukemiaReportPDF(FPDF):
    """Professional PDF class for medical screening reports"""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
    
    def header(self):
        # Dark slate header bar
        self.set_fill_color(30, 41, 59)
        self.rect(0, 0, 210, 32, style='F')
        
        # Title
        self.set_font('Helvetica', 'B', 22)
        self.set_text_color(255, 255, 255)
        self.set_y(6)
        self.cell(0, 10, 'LeukemiaScope', align='C', new_x='LMARGIN', new_y='NEXT')
        
        self.set_font('Helvetica', '', 10)
        self.set_text_color(180, 200, 220)
        self.cell(0, 6, 'AI-Powered Blood Cell Analysis Report', align='C', new_x='LMARGIN', new_y='NEXT')
        
        self.ln(12)
    
    def footer(self):
        self.set_y(-20)
        # Divider line
        self.set_draw_color(100, 116, 139)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 5, f'Page {self.page_no()} | LeukemiaScope Report | Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C', new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 5, 'MedGemma Impact Challenge 2026 | By Chaudhry Muhammad Suleman & Muhammad Idnan', align='C')
    
    def add_section_header(self, title: str, icon: str = ""):
        """Add a professional section header"""
        self.ln(4)
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(30, 41, 59)
        display_title = f"{icon}  {title}" if icon else title
        self.cell(0, 8, self._safe(display_title), new_x='LMARGIN', new_y='NEXT')
        self.ln(2)
    
    def add_field_row(self, label: str, value: str, label2: str = None, value2: str = None):
        """Add a two-column field row"""
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(100, 116, 139)
        self.cell(40, 7, self._safe(label + ":"))
        self.set_font('Helvetica', '', 9)
        self.set_text_color(30, 41, 59)
        
        if label2:
            self.cell(55, 7, self._safe(str(value)))
            self.set_font('Helvetica', 'B', 9)
            self.set_text_color(100, 116, 139)
            self.cell(40, 7, self._safe(label2 + ":"))
            self.set_font('Helvetica', '', 9)
            self.set_text_color(30, 41, 59)
            self.cell(0, 7, self._safe(str(value2 or '')), new_x='LMARGIN', new_y='NEXT')
        else:
            self.cell(0, 7, self._safe(str(value)), new_x='LMARGIN', new_y='NEXT')
    
    def add_result_banner(self, classification: str, confidence: float, severity: str = None):
        """Add classification result banner with severity"""
        self.ln(3)
        
        # Colours based on classification
        if classification == "Normal":
            bg_r, bg_g, bg_b = 34, 197, 94
            label = "NORMAL — No Abnormality Detected"
        elif classification == "Leukemia":
            bg_r, bg_g, bg_b = 180, 83, 9
            label = "LEUKEMIA DETECTED — Abnormal Blast Cells Identified"
        else:
            bg_r, bg_g, bg_b = 234, 179, 8
            label = "UNCERTAIN — Review Required"
        
        # Main result box
        y = self.get_y()
        self.set_fill_color(bg_r, bg_g, bg_b)
        box_height = 28
        self.rect(10, y, 190, box_height, style='F')
        
        # Classification text
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 15)
        self.set_xy(10, y + 4)
        self.cell(190, 8, self._safe(label), align='C')
        
        # Confidence
        self.set_font('Helvetica', '', 11)
        self.set_xy(10, y + 14)
        conf_text = f"AI Confidence: {confidence:.1%}"
        if severity:
            conf_text += f"   |   Severity: {severity}"
        self.cell(190, 8, self._safe(conf_text), align='C')
        
        self.set_y(y + box_height + 5)
        self.set_text_color(30, 41, 59)
    
    def add_divider(self):
        """Add a subtle divider line"""
        self.ln(2)
        self.set_draw_color(226, 232, 240)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
    
    def add_text_block(self, text: str):
        """Add a block of text, handling markdown headers, bold, and lists"""
        lines = text.split('\n')
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                self.ln(2)
                continue
            
            # Handle headers (# or ##)
            if stripped.startswith('#'):
                header_text = stripped.lstrip('#').strip()
                self.ln(2)
                self.set_x(self.l_margin)
                self.set_font('Helvetica', 'B', 10)
                self.set_text_color(30, 41, 59)
                self.multi_cell(w=190, h=6, text=self._safe(header_text))
                self.set_font('Helvetica', '', 9)
                self.set_text_color(51, 65, 85)
                continue
            
            # Handle bullet points
            if stripped.startswith(('- ', '* ', '• ')):
                bullet_text = stripped.lstrip('-*• ').strip()
                clean = re.sub(r'\*\*(.*?)\*\*', r'\1', bullet_text)
                self.set_x(self.l_margin)
                self.set_font('Helvetica', '', 9)
                self.set_text_color(51, 65, 85)
                self.multi_cell(w=185, h=5, text=self._safe(f"  -  {clean[:300]}"))
                self.ln(1)
                continue
            
            # Handle numbered items
            if len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in '.)':
                item_text = stripped.lstrip('0123456789.) ').strip()
                clean = re.sub(r'\*\*(.*?)\*\*', r'\1', item_text)
                num = stripped[0]
                self.set_x(self.l_margin)
                self.set_font('Helvetica', '', 9)
                self.set_text_color(51, 65, 85)
                self.multi_cell(w=185, h=5, text=self._safe(f"  {num}.  {clean[:300]}"))
                self.ln(1)
                continue
                
            # Normal text line
            self.set_x(self.l_margin)
            self.set_font('Helvetica', '', 9)
            self.set_text_color(51, 65, 85)
            clean = re.sub(r'\*\*(.*?)\*\*', r'\1', stripped)
            self.multi_cell(w=190, h=5, text=self._safe(clean))
    
    def _add_bullet_item(self, text: str):
        """Add a single bullet item"""
        clean = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        self.set_x(self.l_margin)
        self.set_font('Helvetica', '', 9)
        self.set_text_color(51, 65, 85)
        self.multi_cell(w=185, h=5, text=self._safe(f"  -  {clean[:300]}"))
        self.ln(1)
    
    def _add_numbered_item(self, number: str, text: str):
        """Add a numbered list item"""
        clean = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        self.set_x(self.l_margin)
        self.set_font('Helvetica', '', 9)
        self.set_text_color(51, 65, 85)
        self.multi_cell(w=185, h=5, text=self._safe(f"  {number}.  {clean[:300]}"))
        self.ln(1)
    
    def add_bullet_list(self, items: list):
        """Add a formatted bullet list"""
        self.set_font('Helvetica', '', 9)
        self.set_text_color(51, 65, 85)
        for item in items:
            clean = re.sub(r'\*\*(.*?)\*\*', r'\1', item.lstrip('0123456789.-*) '))
            self._add_bullet_item(clean)
        self.ln(2)
    
    def add_info_table(self, rows: list):
        """Add a two-column info table with shading"""
        self.set_font('Helvetica', '', 9)
        for i, (label, value) in enumerate(rows):
            # Alternate row shading
            if i % 2 == 0:
                self.set_fill_color(248, 250, 252)
                fill = True
            else:
                fill = False
            
            self.set_text_color(100, 116, 139)
            self.set_font('Helvetica', '', 9)
            self.cell(70, 7, self._safe(label), fill=fill)
            self.set_text_color(30, 41, 59)
            self.set_font('Helvetica', 'B', 9)
            self.cell(0, 7, self._safe(str(value)), fill=fill, new_x='LMARGIN', new_y='NEXT')
        self.ln(2)
    
    @staticmethod
    def _safe(text: str) -> str:
        """Make text safe for Helvetica (latin-1 only)"""
        if not text:
            return ""
        # Replace common unicode characters
        replacements = {
            '\u2014': '--', '\u2013': '-', '\u2018': "'", '\u2019': "'",
            '\u201c': '"', '\u201d': '"', '\u2022': '-', '\u2026': '...',
            '\u2265': '>=', '\u2264': '<=', '\u03bc': 'u', '\u00b5': 'u',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.encode('latin-1', errors='replace').decode('latin-1')


def _calculate_age(dob: str) -> str:
    """Calculate age from DOB string."""
    if not dob:
        return "N/A"
    try:
        from datetime import datetime
        birth = datetime.strptime(dob.strip(), "%Y-%m-%d")
        today = datetime.now()
        age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        return f"{age} years"
    except (ValueError, TypeError):
        return "N/A"


def generate_pdf_report(
    patient_name: str,
    patient_dob: str,
    patient_id: str,
    classification: str,
    confidence: float,
    clinical_advice: str = None,
    next_steps: list = None,
    severity: str = None,
    patient_gender: str = None,
    output_path: str = None
) -> str:
    """
    Generate a professional PDF screening report.
    
    Returns:
        Path to generated PDF, or None if generation fails
    """
    try:
        pdf = LeukemiaReportPDF()
        pdf.add_page()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        age = _calculate_age(patient_dob)
        
        # ── Patient Information ──
        pdf.add_section_header("Patient Information")
        pdf.add_field_row("Full Name", patient_name or "Not provided", "Patient ID", patient_id or "N/A")
        pdf.add_field_row("Date of Birth", patient_dob or "Not provided", "Age", age)
        pdf.add_field_row("Gender", patient_gender or "Not specified", "Report Date", timestamp)
        
        pdf.add_divider()
        
        # ── Classification Result ──
        pdf.add_section_header("Classification Result")
        pdf.add_result_banner(classification, confidence, severity)
        
        pdf.add_divider()
        
        
        # ── Analysis Technology ──
        # (Removed for conciseness)
        
        # ── Clinical Recommendations ──
        
        # ── Clinical Recommendations ──
        if clinical_advice:
            pdf.add_section_header("Clinical Recommendations")
            
            # Parse and render advice (handle both string and complex types)
            advice_text = ""
            if isinstance(clinical_advice, list):
                parts = []
                for item in clinical_advice:
                    if isinstance(item, dict) and 'text' in item:
                        parts.append(item['text'])
                    elif isinstance(item, str):
                        parts.append(item)
                advice_text = "\n".join(parts)
            elif isinstance(clinical_advice, dict):
                advice_text = clinical_advice.get('text', str(clinical_advice))
            else:
                advice_text = str(clinical_advice)
            
            # Render full clinical advice — no truncation
            pdf.add_text_block(advice_text)
            
            pdf.add_divider()
        
        # ── Recommended Next Steps ──
        # Only show separate next steps if clinical advice didn't cover them (i.e. no clinical_advice provided)
        elif next_steps:
            pdf.add_section_header("Recommended Next Steps")
            pdf.add_bullet_list(next_steps)
            pdf.add_divider()
        
        # ── Disclaimer ──
        pdf.add_section_header("Important Disclaimer")
        
        # Warning box
        y = pdf.get_y()
        pdf.set_fill_color(254, 252, 232)
        pdf.set_draw_color(253, 224, 71)
        pdf.rect(10, y, 190, 38, style='DF')
        
        pdf.set_xy(14, y + 3)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(161, 98, 7)
        pdf.cell(0, 5, pdf._safe("WARNING: This report is for RESEARCH & EDUCATIONAL PURPOSES ONLY"), new_x='LMARGIN', new_y='NEXT')
        
        pdf.set_x(14)
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(133, 77, 14)
        disclaimer_text = (
            "This is NOT a medical diagnosis. This AI screening tool is a research prototype developed for "
            "the MedGemma Impact Challenge 2026. Results must be confirmed through standard laboratory "
            "procedures (CBC, bone marrow biopsy, flow cytometry) by a qualified haematologist or oncologist. "
            "Do not make clinical decisions based solely on this report."
        )
        pdf.multi_cell(182, 4, pdf._safe(disclaimer_text))
        
        pdf.set_y(y + 42)
        
        # ── Generate output ──
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = (patient_name or "anonymous").replace(" ", "_")[:20]
            output_path = f"/tmp/leukemiascope_report_{safe_name}_{ts}.pdf"
        
        pdf.output(output_path)
        return output_path
        
    except Exception as e:
        print(f"⚠️ PDF generation error: {e}")
        import traceback
        traceback.print_exc()
        return None
