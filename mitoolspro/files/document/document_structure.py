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


class Line:
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


class Box:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.lines = []

    @property
    def text(self):
        return "\n".join(line.text for line in self.lines)

    def add_line(self, line):
        self.lines.append(line)

    def get_all_lines(self):
        return self.lines

    def get_all_chars(self):
        return [char for line in self.lines for char in line.get_all_chars()]

    def to_json(self):
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "text": self.text,
            "lines": [l.to_json() for l in self.lines],
        }

    def __repr__(self):
        return f"Box(lines={len(self.lines)})"


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

    def append_run(self, run):
        last_box = self.boxes[-1]
        new_line = Line(last_box.x0, last_box.y0, last_box.x1, last_box.y1)
        new_line.add_run(run)
        last_box.add_line(new_line)


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
