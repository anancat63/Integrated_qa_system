import os
from typing import Iterator

import cv2
import fitz
import numpy as np
from PIL import Image
from tqdm import tqdm

from .ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain.text_splitter import CharacterTextSplitter

PDF_OCR_THRESHOLD = (0.6, 0.6)


class OCRPDFLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        line = self.pdf2text()
        yield Document(page_content=line, metadata={"source": self.file_path})

    def pdf2text(self) -> str:
        ocr = get_ocr()
        doc = fitz.open(self.file_path)

        resp = ""
        b_unit = tqdm(total=doc.page_count, desc="OCRPDFLoader context page index: 0")

        for i, page in enumerate(doc):
            b_unit.set_description("OCRPDFLoader context page index: {}".format(i))
            b_unit.refresh()

            text = page.get_text("text")
            resp += text + "\n"

            img_list = page.get_image_info(xrefs=True)
            for img in img_list:
                if xref := img.get("xref"):
                    bbox = img["bbox"]

                    if (
                        (bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0]
                        or (bbox[3] - bbox[1]) / (page.rect.height) < PDF_OCR_THRESHOLD[1]
                    ):
                        continue

                    pix = fitz.Pixmap(doc, xref)

                    if int(page.rotation) != 0:
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                            pix.height, pix.width, -1
                        )
                        tmp_img = Image.fromarray(img_array)
                        ori_img = cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR)
                        rot_img = self.rotate_img(img=ori_img, angle=360 - page.rotation)
                        img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                    else:
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                            pix.height, pix.width, -1
                        )

                    result, _ = ocr(img_array)
                    if result:
                        ocr_result = [line[1] for line in result]
                        resp += "\n".join(ocr_result)

            b_unit.update(1)

        return resp

    def rotate_img(self, img, angle):
        h, w = img.shape[:2]
        rotate_center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
        new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
        return rotated_img


if __name__ == "__main__":
    samples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))
    pdf_path = os.path.join(samples_dir, "ocr_03.pdf")

    pdf_loader = OCRPDFLoader(file_path=pdf_path)
    doc = pdf_loader.load()
    print(type(doc))
    print(doc)

    text_spliter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    result = text_spliter.split_documents(doc)
    print(len(result))
    print(result[0])

