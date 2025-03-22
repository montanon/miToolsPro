from mitoolspro.document.document_structure import BBox, Box


def merge_overlapping_boxes(boxes: list[Box]) -> list[Box]:
    if not boxes:
        return boxes

    def merge_two_boxes(box1: Box, box2: Box) -> Box:
        new_bbox = BBox(
            min(box1.bbox.x0, box2.bbox.x0),
            min(box1.bbox.y0, box2.bbox.y0),
            max(box1.bbox.x1, box2.bbox.x1),
            max(box1.bbox.y1, box2.bbox.y1),
        )
        merged_box = Box(new_bbox)

        # Copy content from both boxes
        for line in box1.get_all_lines():
            merged_box.add_line(line)
        for line in box2.get_all_lines():
            merged_box.add_line(line)
        for image in box1.get_all_images():
            merged_box.add_image(image)
        for image in box2.get_all_images():
            merged_box.add_image(image)

        return merged_box

    def merge_overlapping_group(group_boxes: list[Box]) -> list[Box]:
        if not group_boxes:
            return []

        # Keep merging until no more merges are possible
        while True:
            merged = []
            any_merged = False
            i = 0

            while i < len(group_boxes):
                current = group_boxes[i]
                merged_with_current = False

                j = i + 1
                while j < len(group_boxes):
                    if current.bbox.overlaps(group_boxes[j].bbox):
                        current = merge_two_boxes(current, group_boxes[j])
                        group_boxes.pop(j)
                        merged_with_current = True
                        any_merged = True
                    else:
                        j += 1

                merged.append(current)
                i += 1

            group_boxes = merged
            if not any_merged:
                break

        return group_boxes

    # Sort boxes by top edge (y1) descending, then left edge (x0)
    boxes.sort(key=lambda b: (-b.bbox.y1, b.bbox.x0))

    # First pass: merge same-type overlapping boxes
    text_boxes = [b for b in boxes if len(b.get_all_lines()) > 0]
    image_boxes = [b for b in boxes if len(b.get_all_images()) > 0]
    empty_boxes = [b for b in boxes if not b.get_all_lines() and not b.get_all_images()]

    merged_text = merge_overlapping_group(text_boxes)
    merged_images = merge_overlapping_group(image_boxes)
    all_boxes = merged_text + merged_images + empty_boxes

    # Second pass: adjust positions of different-type overlapping boxes
    all_boxes.sort(key=lambda b: (-b.bbox.y1, b.bbox.x0))
    result = []

    if not all_boxes:
        return result

    current_box = all_boxes[0]
    for next_box in all_boxes[1:]:
        if current_box.bbox.overlaps(next_box.bbox):
            # If different types, adjust positions
            if bool(current_box.get_all_images()) != bool(next_box.get_all_images()):
                next_box.bbox.y1 = min(next_box.bbox.y1, current_box.bbox.y0)
        result.append(current_box)
        current_box = next_box

    result.append(current_box)
    return result
