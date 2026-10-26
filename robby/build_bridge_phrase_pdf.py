"""Reproduce the four-page English plan: uv run --with reportlab python this.py."""
import argparse
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,KeepTogether

ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'static/bridge-phrases-plan.pdf');args=parser.parse_args()
    fonts=Path('/usr/share/fonts/truetype/dejavu')
    pdfmetrics.registerFont(TTFont('Body',str(fonts/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('Bold',str(fonts/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
    styles={'body':ParagraphStyle('body',fontName='Body',fontSize=9,leading=12.7,spaceAfter=7,textColor=colors.HexColor('#263446')),
        'h1':ParagraphStyle('h1',fontName='Bold',fontSize=20,leading=24,spaceAfter=13,textColor=colors.HexColor('#133b50')),
        'h2':ParagraphStyle('h2',fontName='Bold',fontSize=11,leading=15,spaceBefore=7,spaceAfter=7,textColor=colors.HexColor('#176d79')),
        'cell':ParagraphStyle('cell',fontName='Body',fontSize=8,leading=10.6,spaceAfter=0)}
    def p(text,style='body'):
        text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',escape(text))
        return Paragraph(text,styles[style])
    story=[]
    pages=(ROOT/'BRIDGE_PHRASES_PLAN.md').read_text().split('---page---')
    for index,page in enumerate(pages):
        if index:story.append(PageBreak())
        lines=page.strip().splitlines();i=0
        while i<len(lines):
            line=lines[i].strip();i+=1
            if not line:continue
            if line.startswith('|'):
                rows=[line]
                while i<len(lines) and lines[i].strip().startswith('|'):rows.append(lines[i].strip());i+=1
                data=[[p(c.strip(),'cell') for c in row.strip('|').split('|')] for row in rows]
                widths=[136,208,163] if len(data[0])==3 else [115,392]
                table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dceef0')),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f6f8'),colors.white]),
                    ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                    ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
                story.extend([table,Spacer(1,9)]);continue
            if line.startswith('# '):story.append(p(line[2:],'h1'))
            elif line.startswith('## '):story.append(p(line[3:],'h2'))
            else:story.append(p(('• '+line[2:]) if line.startswith('- ') else line))
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#bfd4d9'));canvas.line(44,38,551,38)
        canvas.setFont('Body',8);canvas.setFillColor(colors.HexColor('#536a78'))
        canvas.drawString(44,25,'ROBBY  /  BRIDGE PHRASES  /  PROPOSAL — NOT A BENCHMARK')
        canvas.drawRightString(551,25,str(doc.page));canvas.restoreState()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(args.out),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=40,bottomMargin=50,
        title='Robby: Contextual Bridge Phrases',author='LAION / Robby project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer);print(args.out)
if __name__=='__main__':main()
