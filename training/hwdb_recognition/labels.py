import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelMap:
    characters: list[str]

    @property
    def size(self) -> int:
        return len(self.characters)

    def index_to_char(self, index: int) -> str:
        return self.characters[index]

    def char_to_index(self, character: str) -> int:
        return self.characters.index(character)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.characters, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LabelMap":
        characters = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(characters, list) or not all(isinstance(item, str) for item in characters):
            raise ValueError(f"Invalid label map: {path}")
        return cls(characters)
