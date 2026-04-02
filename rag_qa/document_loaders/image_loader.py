import os
from typing import Iterator

from .ocr import get_ocr
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class OCRIMGLoader(BaseLoader):
    def __init__(self, img_path: str) -> None:
        self.img_path = img_path

    def lazy_load(self) -> Iterator[Document]:
        line = self.img2text()
        yield Document(page_content=line, metadata={"source": self.img_path})

    def img2text(self) -> str:
        resp = ""
        ocr = get_ocr()
        result, _ = ocr(self.img_path)
        if result:
            ocr_result = [line[1] for line in result]
            resp += "\n".join(ocr_result)
        return resp


if __name__ == "__main__":
    samples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))
    img_path = os.path.join(samples_dir, "ocr_04.png")

    img_loader = OCRIMGLoader(img_path=img_path)
    doc = img_loader.load()
    print(doc)

