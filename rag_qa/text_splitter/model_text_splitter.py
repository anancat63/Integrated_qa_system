import os
import re
from typing import List, Optional, Union

from langchain.text_splitter import CharacterTextSplitter
from modelscope.pipelines import pipeline


class AliTextSplitter(CharacterTextSplitter):
    def __init__(
        self,
        pdf: bool = False,
        *,
        model_dir: Optional[str] = None,
        device: Union[str, int] = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.model_dir = model_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "nlp_bert_document-segmentation_chinese-base")
        )
        self.device = device
        self._pipeline = None

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub(r"\s", " ", text)
            text = re.sub(r"\n\n", "", text)

        if self._pipeline is None:
            self._pipeline = pipeline(
                task="document-segmentation",
                model=self.model_dir,
                device=self.device,
            )

        result = self._pipeline(documents=text)
        return [i for i in result["text"].split("\n\t") if i]

