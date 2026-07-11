from pypdf import PdfReader
reader = PdfReader("accounting_star 20260320 JAE1.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()
with open("pdf_text.txt", "w") as f:
    f.write(text)
