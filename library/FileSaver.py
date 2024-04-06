# from fpdf import FPDF
from docx import Document
from bs4 import BeautifulSoup
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

class FileSaver:
    def __init__(self, html):
        self.html = html
        self.text = self._parse_html(html)

    def _parse_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()

    def save_pdf(self, filename):
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica", 12)
        c.drawString(30, height - 30, self.text)
        c.save()

    def save_docx(self, filename):
        doc = Document()
        doc.add_paragraph(self.text)
        doc.save(filename)

    def save_html(self, filename):
        with open(filename, "w", encoding='utf-8') as file:
            file.write(self.html)