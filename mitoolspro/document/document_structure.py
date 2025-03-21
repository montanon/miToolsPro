from typing import List


class BBox:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0

    @property
    def center(self):
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    @property
    def xcenter(self):
        return (self.x0 + self.x1) / 2

    @property
    def ycenter(self):
        return (self.y0 + self.y1) / 2

    def __repr__(self):
        return f"BBox(x0={self.x0}, y0={self.y0}, x1={self.x1}, y1={self.y1})"

    def __eq__(self, other):
        if not isinstance(other, BBox):
            return False
        return (
            self.x0 == other.x0
            and self.y0 == other.y0
            and self.x1 == other.x1
            and self.y1 == other.y1
        )

    def to_json(self):
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_json(cls, json_data):
        return cls(
            x0=json_data["x0"],
            y0=json_data["y0"],
            x1=json_data["x1"],
            y1=json_data["y1"],
        )


class Char:
    def __init__(self, text, fontname, size, x0, y0, x1, y1):
        self.text = text
        self.fontname = fontname
        self.size = size
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def to_json(self):
        return {
            "text": self.text,
            "fontname": self.fontname,
            "size": self.size,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }

    def __repr__(self):
        return f"Char({self.text!r}, font={self.fontname}, size={self.size})"

    @property
    def bbox(self):
        return BBox(self.x0, self.y0, self.x1, self.y1)

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0

    @property
    def bold(self):
        return "bold" in self.fontname.lower()

    @property
    def italic(self):
        return "italic" in self.fontname.lower() or "oblique" in self.fontname.lower()

    @classmethod
    def from_json(cls, json_data):
        return cls(
            text=json_data["text"],
            fontname=json_data["fontname"],
            size=json_data["size"],
            x0=json_data["x0"],
            y0=json_data["y0"],
            x1=json_data["x1"],
            y1=json_data["y1"],
        )

    def __eq__(self, other):
        if not isinstance(other, Char):
            return False
        return (
            self.text == other.text
            and self.fontname == other.fontname
            and abs(self.size - other.size) < 0.0001
            and self.x0 == other.x0
            and self.y0 == other.y0
            and self.x1 == other.x1
            and self.y1 == other.y1
        )


class Run:
    def __init__(self, fontname, size, text=None, chars=None):
        self.fontname = fontname
        self.size = size
        self.chars = []
        if text is not None:
            for char in text:
                self.chars.append(
                    Char(
                        text=char, fontname=fontname, size=size, x0=0, y0=0, x1=0, y1=0
                    )
                )
        elif chars is not None:
            self.chars = chars

    @property
    def text(self):
        return "".join(c.text for c in self.chars)

    def append_char(self, char):
        self.chars.append(char)

    def __add__(self, other):
        if not isinstance(other, Run):
            raise TypeError(f"Cannot add Run with {type(other)}")
        combined_chars = self.chars + other.chars
        return Run(fontname=self.fontname, size=self.size, chars=combined_chars)

    def __eq__(self, other):
        if not isinstance(other, Run):
            return False
        return (
            self.fontname == other.fontname
            and abs(self.size - other.size) < 0.0001
            and self.text == other.text
            and len(self.chars) == len(other.chars)
            and all(
                c1.x0 == c2.x0 and c1.y0 == c2.y0 and c1.x1 == c2.x1 and c1.y1 == c2.y1
                for c1, c2 in zip(self.chars, other.chars)
            )
        )

    def to_json(self):
        return {
            "fontname": self.fontname,
            "size": self.size,
            "text": self.text,
            "chars": [c.to_json() for c in self.chars],
        }

    def __repr__(self):
        return f"Run(text={self.text!r}, font={self.fontname}, size={self.size}, bbox={self.bbox})"

    @classmethod
    def from_text(cls, text, fontname, size):
        return cls(fontname=fontname, size=size, text=text)

    @classmethod
    def from_chars(cls, chars, fontname=None, size=None):
        if not chars:
            raise ValueError("Cannot create Run from empty chars list")
        fontname = fontname or chars[0].fontname
        size = size or chars[0].size
        return cls(fontname=fontname, size=size, chars=chars)

    @classmethod
    def from_json(cls, json_data):
        chars = [Char.from_json(char_data) for char_data in json_data["chars"]]
        return cls(fontname=json_data["fontname"], size=json_data["size"], chars=chars)

    def is_bold(self):
        return "bold" in self.fontname.lower()

    def is_italic(self):
        return "italic" in self.fontname.lower() or "oblique" in self.fontname.lower()

    @property
    def bbox(self):
        chars_bboxs = [char.bbox for char in self.chars if char.text != "\n"]
        x0 = min(bbox.x0 for bbox in chars_bboxs)
        y0 = min(bbox.y0 for bbox in chars_bboxs)
        x1 = max(bbox.x1 for bbox in chars_bboxs)
        y1 = max(bbox.y1 for bbox in chars_bboxs)
        return BBox(x0, y0, x1, y1)


class BoxElement:
    pass


class Line(BoxElement):
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.runs = []

    @property
    def text(self):
        return "".join(run.text for run in self.runs)

    def add_run(self, run):
        self.runs.append(run)

    def get_all_chars(self):
        return [char for run in self.runs for char in run.chars]

    def to_json(self):
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "text": self.text,
            "runs": [r.to_json() for r in self.runs],
        }

    def __repr__(self):
        return f"Line(text={self.text!r})"

    @classmethod
    def from_json(cls, json_data):
        line = cls(json_data["x0"], json_data["y0"], json_data["x1"], json_data["y1"])
        for run_data in json_data["runs"]:
            line.add_run(Run.from_json(run_data))
        return line

    def __eq__(self, other):
        if not isinstance(other, Line):
            return False
        return (
            self.x0 == other.x0
            and self.y0 == other.y0
            and self.x1 == other.x1
            and self.y1 == other.y1
            and len(self.runs) == len(other.runs)
            and all(r1 == r2 for r1, r2 in zip(self.runs, other.runs))
        )


class Image(BoxElement):
    def __init__(
        self, bbox: BBox, stream: bytes = None, name: str = "", mimetype: str = None
    ):
        self.bbox = bbox
        self.stream = stream
        self.name = name
        self.mimetype = mimetype

    def to_json(self):
        return {
            "bbox": self.bbox.to_json(),
            "stream": self.stream,
            "name": self.name,
            "mimetype": self.mimetype,
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(
            bbox=BBox.from_json(json_data["bbox"]),
            stream=json_data.get("stream"),
            name=json_data.get("name", ""),
            mimetype=json_data.get("mimetype"),
        )

    def __repr__(self):
        return f"Image(name={self.name}, bbox={self.bbox})"


class Box:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.elements: List[BoxElement] = []

    @property
    def text(self):
        return "\n".join(el.text for el in self.elements if isinstance(el, Line))

    def add_line(self, line):
        if not isinstance(line, Line):
            raise ValueError("Line must be a Line")
        self.elements.append(line)

    def add_image(self, image):
        if not isinstance(image, Image):
            raise ValueError("Image must be an Image")
        self.elements.append(image)

    def get_all_lines(self):
        return [el for el in self.elements if isinstance(el, Line)]

    def get_all_images(self):
        return [el for el in self.elements if isinstance(el, Image)]

    def get_all_chars(self):
        return [
            char
            for el in self.elements
            if isinstance(el, Line)
            for char in el.get_all_chars()
        ]

    def to_json(self):
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "text": self.text,
            "elements": [
                {"type": "line", **el.to_json()}
                if isinstance(el, Line)
                else {"type": "image", **el.to_json()}
                for el in self.elements
            ],
        }

    def __repr__(self):
        return f"Box(lines={len(self.get_all_lines())}, images={len(self.get_all_images())})"

    @classmethod
    def from_json(cls, json_data):
        box = cls(json_data["x0"], json_data["y0"], json_data["x1"], json_data["y1"])
        for el_data in json_data["elements"]:
            if el_data["type"] == "line":
                box.add_line(Line.from_json(el_data))
            elif el_data["type"] == "image":
                box.add_image(Image.from_json(el_data))
        return box

    def __eq__(self, other):
        if not isinstance(other, Box):
            return False
        return (
            self.x0 == other.x0
            and self.y0 == other.y0
            and self.x1 == other.x1
            and self.y1 == other.y1
            and len(self.elements) == len(other.elements)
            and all(l1 == l2 for l1, l2 in zip(self.elements, other.elements))
        )


class Page:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.boxes = []

    @property
    def text(self):
        return "\n".join(box.text for box in self.boxes)

    def add_box(self, box):
        self.boxes.append(box)

    def get_all_lines(self):
        lines = []
        for box in self.boxes:
            lines.extend(box.get_all_lines())
        return lines

    def get_all_chars(self):
        chars = []
        for box in self.boxes:
            chars.extend(box.get_all_chars())
        return chars

    def to_json(self):
        return {
            "width": self.width,
            "height": self.height,
            "text": self.text,
            "boxes": [b.to_json() for b in self.boxes],
        }

    def __repr__(self):
        return f"Page({self.width}x{self.height}, boxes={len(self.boxes)})"

    @classmethod
    def from_json(cls, json_data):
        page = cls(json_data["width"], json_data["height"])
        for box_data in json_data["boxes"]:
            page.add_box(Box.from_json(box_data))
        return page

    def append_run(self, run):
        last_box = self.boxes[-1]
        new_line = Line(last_box.x0, last_box.y0, last_box.x1, last_box.y1)
        new_line.add_run(run)
        last_box.add_line(new_line)

    def __eq__(self, other):
        if not isinstance(other, Page):
            return False
        return (
            self.width == other.width
            and self.height == other.height
            and len(self.boxes) == len(other.boxes)
            and all(b1 == b2 for b1, b2 in zip(self.boxes, other.boxes))
        )


class Document:
    def __init__(self):
        self.pages = []

    def add_page(self, page):
        self.pages.append(page)

    @property
    def text(self):
        return "\n".join(page.text for page in self.pages)

    @property
    def chars(self):
        return self.get_all_chars()

    def get_all_pages(self):
        return self.pages

    def get_all_boxes(self):
        boxes = []
        for p in self.pages:
            boxes.extend(p.boxes)
        return boxes

    def get_all_lines(self):
        lines = []
        for p in self.pages:
            lines.extend(p.get_all_lines())
        return lines

    def get_all_chars(self):
        chars = []
        for p in self.pages:
            chars.extend(p.get_all_chars())
        return chars

    def get_all_runs(self):
        runs = []
        for line in self.get_all_lines():
            runs.extend(line.runs)
        return runs

    def get_text(self):
        return "\n".join(page.text for page in self.pages)

    def to_json(self):
        return {"text": self.text, "pages": [p.to_json() for p in self.pages]}

    def __repr__(self):
        return f"Document(pages={len(self.pages)})"

    @classmethod
    def from_json(cls, json_data):
        doc = cls()
        for page_data in json_data["pages"]:
            doc.add_page(Page.from_json(page_data))
        return doc

    def __eq__(self, other):
        if not isinstance(other, Document):
            return False
        return len(self.pages) == len(other.pages) and all(
            p1 == p2 for p1, p2 in zip(self.pages, other.pages)
        )
