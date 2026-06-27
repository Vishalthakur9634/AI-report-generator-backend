"""
create_template.py — Builds test_template.docx matching the JP Diagnostics letterhead exactly.
Each organ has its own paragraph with bold+underlined name as STATIC text.
Only the finding text is a {tag} placeholder.
"""

import docx
from docx.shared import Pt, Inches, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml.ns import qn


# ── Helpers ────────────────────────────────────────────────────────────────────

def no_space(para):
    """Remove all space-before and space-after from a paragraph."""
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(0)


def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    """Apply border styles to a table cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    borders = tcPr.find(qn('w:tcBorders'))
    if borders is None:
        borders = OxmlElement('w:tcBorders')
        tcPr.append(borders)
    for side, data in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        if data is None:
            continue
        el = borders.find(qn(f'w:{side}'))
        if el is None:
            el = OxmlElement(f'w:{side}')
            borders.append(el)
        for k, v in data.items():
            el.set(qn(f'w:{k}'), str(v))


def none_border():
    return {'val': 'none', 'sz': '0', 'color': 'auto'}


def thin_border():
    return {'val': 'single', 'sz': '6', 'color': '888888'}


def thick_border():
    return {'val': 'single', 'sz': '12', 'color': '000000'}


def remove_cell_margins(cell):
    """Zero out all padding inside a table cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for side in ['top', 'left', 'bottom', 'right']:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:w'), '30')
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_run(para, text, bold=False, italic=False, underline=False,
            size=11, color=None, font='Times New Roman'):
    """Add a formatted run to a paragraph."""
    run = para.add_run(text)
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.underline = underline
    if color:
        run.font.color.rgb = RGBColor(*color)
    return run


def organ_paragraph(doc, label, tag, italic_label=False):
    """
    Add one organ finding paragraph like:
      "The <bold+underline>Liver</bold+underline> {liver_finding}"
    """
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p.paragraph_format.line_spacing = 1.15
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # "The " prefix
    add_run(p, 'The ', size=11)

    # organ name — bold + underlined
    add_run(p, label, bold=True, underline=True, size=11)

    # space then placeholder
    add_run(p, ' ', size=11)
    add_run(p, '{' + tag + '}', size=11)
    return p


# ── Main builder ───────────────────────────────────────────────────────────────

def build_template():
    doc = docx.Document()
    FONT = 'Times New Roman'

    # Page margins
    for sec in doc.sections:
        sec.top_margin    = Inches(0.35)
        sec.bottom_margin = Inches(0.35)
        sec.left_margin   = Inches(0.55)
        sec.right_margin  = Inches(0.55)
        sec.page_width    = Inches(8.27)   # A4

    # ── 1. HEADER TABLE ──────────────────────────────────────────────────────
    # 3 columns: [logo | center branding | contact]
    ht = doc.add_table(rows=1, cols=3)
    ht.autofit = False
    ht.columns[0].width = Inches(1.3)
    ht.columns[1].width = Inches(4.5)
    ht.columns[2].width = Inches(1.97)

    for cell in ht.rows[0].cells:
        set_cell_borders(cell,
            top=none_border(), bottom=none_border(),
            left=none_border(), right=none_border())
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        remove_cell_margins(cell)

    # Left — logo placeholder (circular icon placeholder)
    p_logo = ht.cell(0, 0).paragraphs[0]
    p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    no_space(p_logo)
    add_run(p_logo, '[ JP\nLOGO ]', bold=True, size=9,
            color=(59, 30, 84), font=FONT)

    # Center — clinic name + tagline + motto
    p_brand = ht.cell(0, 1).paragraphs[0]
    p_brand.alignment = WD_ALIGN_PARAGRAPH.CENTER
    no_space(p_brand)

    # "JP DIAGNOSTICS" — big purple bold
    r1 = p_brand.add_run('JP DIAGNOSTICS')
    r1.font.name = FONT
    r1.font.size = Pt(26)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(0x3B, 0x1E, 0x54)

    r1.add_break()

    r2 = p_brand.add_run('RADIOLOGY | PATHOLOGY')
    r2.font.name = FONT
    r2.font.size = Pt(12)
    r2.font.bold = True

    r2.add_break()

    r3 = p_brand.add_run('WE DIAGNOSE RIGHT')
    r3.font.name = FONT
    r3.font.size = Pt(10)
    r3.font.bold = True
    r3.font.color.rgb = RGBColor(0xFF, 0x98, 0x00)   # orange

    # Right — contact info
    p_contact = ht.cell(0, 2).paragraphs[0]
    p_contact.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    no_space(p_contact)
    contact_text = (
        '\u260e +91-7579470000\n'
        '\u260e +91-7579430000\n'
        '\u2709 info@jpdiagnostics.in\n'
        '\u2302 JSR Tower, Goverdhan\n'
        '  Crossing, NH-19, Mathura'
    )
    add_run(p_contact, contact_text, size=8, font=FONT, color=(40, 40, 40))

    # ── 2. THIN SEPARATOR LINE ────────────────────────────────────────────────
    sep1 = doc.add_paragraph()
    no_space(sep1)
    sep1.paragraph_format.space_before = Pt(4)
    pPr = sep1._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bot = OxmlElement('w:bottom')
    bot.set(qn('w:val'), 'single')
    bot.set(qn('w:sz'), '6')
    bot.set(qn('w:space'), '1')
    bot.set(qn('w:color'), '888888')
    pBdr.append(bot)
    pPr.append(pBdr)

    # ── 3. PATIENT INFO TABLE ─────────────────────────────────────────────────
    pt = doc.add_table(rows=4, cols=4)
    pt.autofit = False

    col_widths = [Inches(0.9), Inches(2.7), Inches(1.1), Inches(2.57)]
    for row in pt.rows:
        for i, w in enumerate(col_widths):
            row.cells[i].width = w
        for cell in row.cells:
            remove_cell_margins(cell)
            set_cell_borders(cell,
                top=none_border(), bottom=none_border(),
                left=none_border(), right=none_border())

    # Row 0 — Patient ID / Reg Date
    def fill_row(row_idx, label1, val1_tag, label2, val2_tag):
        row = pt.rows[row_idx]
        for cell in row.cells:
            for p in cell.paragraphs:
                no_space(p)
                p.paragraph_format.space_after = Pt(2)

        p0 = row.cells[0].paragraphs[0]
        add_run(p0, label1, bold=True, size=9, font=FONT)

        p1 = row.cells[1].paragraphs[0]
        add_run(p1, '{' + val1_tag + '}', size=9, font=FONT)

        p2 = row.cells[2].paragraphs[0]
        add_run(p2, label2, bold=True, size=9, font=FONT)

        p3 = row.cells[3].paragraphs[0]
        add_run(p3, '{' + val2_tag + '}', size=9, font=FONT)

    fill_row(0, 'Patient ID', 'patient_id', 'Reg. Date', 'reg_date')
    fill_row(1, 'Name',       'patient_name', 'Report Date', 'report_date')

    # Row 2 — Age + Sex merged label
    row2 = pt.rows[2]
    for cell in row2.cells:
        no_space(cell.paragraphs[0])
        cell.paragraphs[0].paragraph_format.space_after = Pt(2)
    add_run(row2.cells[0].paragraphs[0], 'Age', bold=True, size=9, font=FONT)
    p_age = row2.cells[1].paragraphs[0]
    add_run(p_age, '{age}', size=9, font=FONT)
    add_run(p_age, '         Sex : ', bold=True, size=9, font=FONT)
    add_run(p_age, '{sex}', size=9, font=FONT)

    # Row 3 — Ref By
    row3 = pt.rows[3]
    for cell in row3.cells:
        no_space(cell.paragraphs[0])
        cell.paragraphs[0].paragraph_format.space_after = Pt(2)
    add_run(row3.cells[0].paragraphs[0], 'Ref. By', bold=True, size=9, font=FONT)
    add_run(row3.cells[1].paragraphs[0], '{ref_by}', size=9, font=FONT)

    # ── 4. SEPARATOR LINE (double) ────────────────────────────────────────────
    sep2 = doc.add_paragraph()
    no_space(sep2)
    sep2.paragraph_format.space_before = Pt(4)
    pPr2 = sep2._p.get_or_add_pPr()
    pBdr2 = OxmlElement('w:pBdr')
    bot2 = OxmlElement('w:bottom')
    bot2.set(qn('w:val'), 'single')
    bot2.set(qn('w:sz'), '8')
    bot2.set(qn('w:space'), '1')
    bot2.set(qn('w:color'), '000000')
    pBdr2.append(bot2)
    pPr2.append(pBdr2)

    # ── 5. SCAN TITLE ─────────────────────────────────────────────────────────
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(10)
    p_title.paragraph_format.space_after = Pt(10)
    r_title = p_title.add_run('USG WHOLE ABDOMEN MALE')
    r_title.font.name = FONT
    r_title.font.size = Pt(13)
    r_title.font.bold = True
    r_title.font.underline = True

    # ── 6. FINDINGS — Each organ is its own formatted paragraph ───────────────
    organ_paragraph(doc, 'Liver', 'liver_finding')
    organ_paragraph(doc, 'Gall Bladder', 'gallbladder_finding')
    organ_paragraph(doc, 'Pancreas', 'pancreas_finding')

    # Spleen — starts with organ name directly (no "The")
    p_spleen = doc.add_paragraph()
    p_spleen.paragraph_format.space_before = Pt(4)
    p_spleen.paragraph_format.space_after = Pt(4)
    p_spleen.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p_spleen.paragraph_format.line_spacing = 1.15
    p_spleen.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_run(p_spleen, 'Spleen', bold=True, underline=True, size=11, font=FONT)
    add_run(p_spleen, ' ', size=11, font=FONT)
    add_run(p_spleen, '{spleen_finding}', size=11, font=FONT)

    # Kidneys
    p_kidneys = doc.add_paragraph()
    p_kidneys.paragraph_format.space_before = Pt(4)
    p_kidneys.paragraph_format.space_after = Pt(4)
    p_kidneys.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p_kidneys.paragraph_format.line_spacing = 1.15
    p_kidneys.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_run(p_kidneys, 'Both ', size=11, font=FONT)
    add_run(p_kidneys, 'Kidneys', bold=True, underline=True, size=11, font=FONT)
    add_run(p_kidneys, ' ', size=11, font=FONT)
    add_run(p_kidneys, '{kidneys_finding}', size=11, font=FONT)

    organ_paragraph(doc, 'Urinary Bladder', 'urinary_bladder_finding')

    # Prostate (male only)
    p_prostate = doc.add_paragraph()
    p_prostate.paragraph_format.space_before = Pt(4)
    p_prostate.paragraph_format.space_after = Pt(4)
    p_prostate.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_run(p_prostate, 'Prostate', bold=True, underline=False, size=11, font=FONT)
    add_run(p_prostate, ': ', size=11, font=FONT)
    add_run(p_prostate, '{prostate_finding}', size=11, font=FONT)

    # Additional finding (free text for anything extra)
    p_extra = doc.add_paragraph()
    p_extra.paragraph_format.space_before = Pt(4)
    p_extra.paragraph_format.space_after = Pt(4)
    p_extra.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p_extra.paragraph_format.line_spacing = 1.15
    p_extra.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_extra = p_extra.add_run('{additional_finding}')
    r_extra.font.name = FONT
    r_extra.font.size = Pt(11)
    r_extra.font.italic = True

    # ── 7. IMPRESSION ─────────────────────────────────────────────────────────
    p_imp_title = doc.add_paragraph()
    p_imp_title.paragraph_format.space_before = Pt(8)
    p_imp_title.paragraph_format.space_after = Pt(4)
    r_imp = p_imp_title.add_run('IMPRESSION:')
    r_imp.font.name = FONT
    r_imp.font.size = Pt(11)
    r_imp.font.bold = True
    r_imp.font.underline = True

    p_imp_body = doc.add_paragraph()
    p_imp_body.paragraph_format.space_before = Pt(0)
    p_imp_body.paragraph_format.space_after = Pt(0)
    p_imp_body.paragraph_format.left_indent = Inches(0.3)
    p_imp_body.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p_imp_body.paragraph_format.line_spacing = 1.3
    r_imp_body = p_imp_body.add_run('{impression}')
    r_imp_body.font.name = FONT
    r_imp_body.font.size = Pt(11)
    r_imp_body.font.bold = True

    # ── 8. SPACE FOR SIGNATURES ───────────────────────────────────────────────
    for _ in range(3):
        sp = doc.add_paragraph()
        no_space(sp)

    # ── 9. DOCTORS SIGNATURE TABLE ────────────────────────────────────────────
    dt = doc.add_table(rows=1, cols=3)
    dt.autofit = False
    col_w = Inches(2.43)
    for i in range(3):
        dt.columns[i].width = col_w
    for cell in dt.rows[0].cells:
        set_cell_borders(cell,
            top=none_border(), bottom=none_border(),
            left=none_border(), right=none_border())
        remove_cell_margins(cell)

    def doc_cell(col, name, qual):
        p = dt.cell(0, col).paragraphs[0]
        no_space(p)
        r_name = p.add_run(name + '\n')
        r_name.font.name = FONT
        r_name.font.size = Pt(9)
        r_name.font.bold = True
        r_qual = p.add_run(qual)
        r_qual.font.name = FONT
        r_qual.font.size = Pt(7.5)

    doc_cell(0,
        'DR. NIKHIL VIKRAM',
        'D.N.B. Radiodiagnosis, Fellowship in advanced\nUSG Mumbai, P.G. DIP. MSK USG,\nUCAM, Spain FMF, Certified U.K. ID 209647,\nFIPM, Certified pain & Palliative Care Physician,\nFellowship in 2D Echo')
    doc_cell(1,
        'DR. NIDHI AGRAWAL',
        'M.D. (Radiology)\nDirector, Consultant Radiologist CT/MRI,\nCardiac and Breast Imaging Consultant.')
    doc_cell(2,
        'DR. DEEPAK AGRAWAL',
        'M.D. (Radiology), Gold Medalist\nDirector, Consultant Radiologist\nCT/MRI Head & Neck, Spine, MSK,\nChest-Abdomen, Whole Body Imaging\n& Fetal Medicine Specialist')

    # ── 10. FOOTER DISCLAIMER ─────────────────────────────────────────────────
    p_disc = doc.add_paragraph()
    p_disc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_disc.paragraph_format.space_before = Pt(8)
    p_disc.paragraph_format.space_after = Pt(2)
    r_disc = p_disc.add_run(
        'N.B. This is only a professional opinion and not the final diagnosis. '
        'Please correlate clinical findings with scan findings, if any disparity '
        'arises, please ask for rescan, please intimate us for any typing mistakes '
        'and sent the report for correction immediately within 7 days.\n'
        'THIS REPORT IS NOT VALID FOR MEDICO LEGAL PURPOSES'
    )
    r_disc.font.name = FONT
    r_disc.font.size = Pt(7)
    r_disc.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    # Services banner (purple text — shading would need XML, using color instead)
    p_srv = doc.add_paragraph()
    p_srv.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_srv.paragraph_format.space_before = Pt(4)
    p_srv.paragraph_format.space_after = Pt(2)
    r_srv = p_srv.add_run(
        'Silent MRI (3T Platform) | CT scan (True 16 multidetectors-96 slice per rotation Recon.) '
        '| 3D/4D Ultrasound | Digital X-Ray\n'
        '(DR-500 mA) | Mammography | Advanced Pathology Lab | '
        'ECG/EEG/NCV/EMG/TMT/CT Denta Scan | DEXA Scan | Sleep Study'
    )
    r_srv.font.name = FONT
    r_srv.font.size = Pt(8)
    r_srv.font.bold = True
    r_srv.font.color.rgb = RGBColor(0x50, 0x00, 0x78)

    # Red legal warning
    p_warn = doc.add_paragraph()
    p_warn.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_warn.paragraph_format.space_before = Pt(2)
    r_warn = p_warn.add_run(
        'Sex determination & sex selective abortion is prohibited & punishable offense. '
        '\u0917\u0930\u094d\u092d \u0915\u093e \u0932\u093f\u0902\u0917 '
        '\u0915\u094d\u0937\u0923 \u0915\u0930\u0928\u093e \u0915\u093e\u0928\u0942\u0928\u0928 '
        '\u0905\u092a\u0930\u093e\u0927 \u0939\u0948\u0964'
    )
    r_warn.font.name = FONT
    r_warn.font.size = Pt(8)
    r_warn.font.bold = True
    r_warn.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)

    doc.save('test_template.docx')
    print('SUCCESS: test_template.docx saved.')
    print('Tags: patient_id, patient_name, age, sex, ref_by, reg_date, report_date,')
    print('      liver_finding, gallbladder_finding, pancreas_finding, spleen_finding,')
    print('      kidneys_finding, urinary_bladder_finding, prostate_finding,')
    print('      additional_finding, impression')
 
 
if __name__ == '__main__':
    build_template()
