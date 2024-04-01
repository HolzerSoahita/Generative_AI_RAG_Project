from fpdf import FPDF
from docx import Document
from bs4 import BeautifulSoup

class FileSaver:
    def __init__(self, html):
        self.html = html
        self.text = self._parse_html(html)

    def _parse_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()

    def save_pdf(self, filename):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size = 15)
        pdf.cell(200, 10, txt = self.text, ln = True, align = 'C')
        pdf.output(filename)

    def save_docx(self, filename):
        doc = Document()
        doc.add_paragraph(self.text)
        doc.save(filename)

    def save_html(self, filename):
        with open(filename, "w") as file:
            file.write(self.html)