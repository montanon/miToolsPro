from mitoolspro.document.document_structure import BBox, Box


def merge_overlapping_boxes(boxes: list[Box]) -> list[Box]:
    if not boxes:
        return []

    def merge_overlapping_group(group_boxes: list[Box]) -> list[Box]:
        if not group_boxes:
            return []

        while True:
            merged = []
            any_merged = False
            i = 0

            while i < len(group_boxes):
                current = group_boxes[i]
                j = i + 1
                while j < len(group_boxes):
                    if current.bbox.overlaps(group_boxes[j].bbox):
                        current = current.merge(group_boxes[j])
                        group_boxes.pop(j)
                        any_merged = True
                    else:
                        j += 1
                merged.append(current)
                i += 1

            group_boxes = merged
            if not any_merged:
                break

        return group_boxes

    # Sort boxes top-to-bottom, left-to-right
    boxes.sort(key=lambda b: (-b.bbox.y1, b.bbox.x0))

    # Separate by content type
    text_boxes = [b for b in boxes if b.get_all_lines()]
    image_boxes = [b for b in boxes if b.get_all_images()]
    empty_boxes = [b for b in boxes if not b.get_all_lines() and not b.get_all_images()]

    merged_text = merge_overlapping_group(text_boxes)
    merged_images = merge_overlapping_group(image_boxes)
    all_boxes = merged_text + merged_images + empty_boxes

    # Second pass: adjust overlapping mixed-type boxes without merging
    all_boxes.sort(key=lambda b: (-b.bbox.y1, b.bbox.x0))
    result = []

    current_box = all_boxes[0]
    for next_box in all_boxes[1:]:
        if current_box.bbox.overlaps(next_box.bbox):
            if bool(current_box.get_all_images()) != bool(next_box.get_all_images()):
                # Adjust vertical overlap of mixed-type boxes
                new_y1 = min(next_box.bbox.y1, current_box.bbox.y0)
                next_box.bbox = next_box.bbox.clone()
                next_box.bbox.y1 = new_y1
        result.append(current_box)
        current_box = next_box

    result.append(current_box)
    return result
