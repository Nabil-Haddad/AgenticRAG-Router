from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Document:
    id : str
    content : str
    source: str
    page_num : Optional[int]
    metadata : dict = field(default_factory=dict)

    @property
    def word_count(self)->int:
        return len(self.content.split())

    def is_valid(self)->bool:
        return len(self.content) >= 10