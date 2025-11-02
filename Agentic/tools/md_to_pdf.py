from markdown import markdown
from bs4 import BeautifulSoup
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle, Preformatted, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
import re
from io import BytesIO  
import requests

def get_image_from_url(url, width=200):
    response = requests.get(url)
    if response.status_code == 200:
        img_data = BytesIO(response.content)
        img = Image(img_data, width=width, hAlign='CENTER')
        return img
    return None


pre_style = ParagraphStyle(
    name='CodeBlock',
    fontName='Courier',
    fontSize=9,
    leading=12,
    leftIndent=10,
    rightIndent=10,
    alignment=TA_LEFT,
    spaceBefore=6,
    spaceAfter=6,
)

def markdown_to_pdf(md_text: str, output_pdf_path: str):
    html = markdown(md_text, extensions=['fenced_code', 'tables'])
    soup = BeautifulSoup(html, 'html.parser')

    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []

    bullet_style = ParagraphStyle(name='Bullet', leftIndent=20)
    code_style = ParagraphStyle(name='Code', fontName='Courier', fontSize=9, leftIndent=20, rightIndent=20)
    pre_style = ParagraphStyle(name='Pre', fontName='Courier', fontSize=8, leading=10)

    def format_html(text):
        # Replace emojis, links, bold, italics, strikethroughs, inline code
        text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<b><i>\1</i></b>', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'~~(.*?)~~', r'<strike>\1</strike>', text)
        text = re.sub(r'`([^`]*)`', r'<font face="Courier">\1</font>', text)
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
        return text

    for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'li', 'pre', 'table','img']):
        if hasattr(element, 'name'):
            tag = element.name
            text = element.text

            if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                style = styles['Heading' + tag[1]]
                story.append(Paragraph(format_html(text), style))
                story.append(Spacer(1, 0.2 * inch))

            elif tag == 'p':
                if element.code and element.code.get_text():
                    # Inline or block code inside <code> tags
                    code_content = element.code.get_text()
                    story.append(Paragraph(code_content, code_style))
                else:
                    story.append(Paragraph(format_html(text), styles['Normal']))
                story.append(Spacer(1, 0.1 * inch))

            elif tag in ['ul', 'ol']:
                bullet_type = 'bullet' if tag == 'ul' else '1'
                items = []
                for li in element.find_all('li'):
                    li_text = format_html(li.decode_contents())
                    items.append(ListItem(Paragraph(li_text, bullet_style)))
                story.append(ListFlowable(items, bulletType=bullet_type))
                story.append(Spacer(1, 0.1 * inch))

            elif tag == 'pre':
                print("got code blockk")
                code_block = element.get_text()
                story.append(Preformatted(code_block, pre_style))
                story.append(Spacer(1, 0.1 * inch))

            elif tag == 'blockquote':
                quote_text = format_html(text)
                story.append(Paragraph(f"<i>{quote_text}</i>", styles['Normal']))
                story.append(Spacer(1, 0.1 * inch))

            elif tag == 'table':
                data = []
                # Handle thead
                headers = element.find('thead')
                if headers:
                    row = [th.get_text(strip=True) for th in headers.find_all('th')]
                    data.append(row)
                # Handle tbody or direct tr
                body = element.find('tbody') or element
                for tr in body.find_all('tr'):
                    row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if row:
                        data.append(row)

                if data:
                    table = Table(data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 0.2 * inch))

            if tag == 'img':
                img_url = element['src']
                img = get_image_from_url(img_url)
                if img:
                    story.append(img)
                    story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
    return output_pdf_path


