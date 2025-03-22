from mitoolspro.document.document_structure import BBox, Box


def merge_overlapping_boxes(boxes: list[Box]) -> list[Box]:
    if not boxes:
        return boxes

    # Sort boxes by top edge (y1) descending, then left edge (x0)
    boxes.sort(key=lambda b: (-b.bbox.y1, b.bbox.x0))

    merged = []
    current_box = boxes[0]

    for next_box in boxes[1:]:
        if current_box.bbox.overlaps(next_box.bbox):
            # If boxes overlap and are of the same type (both text or both image)
            if bool(current_box.get_all_images()) == bool(next_box.get_all_images()):
                # Merge boxes by extending the bbox
                new_bbox = BBox(
                    min(current_box.bbox.x0, next_box.bbox.x0),
                    min(current_box.bbox.y0, next_box.bbox.y0),
                    max(current_box.bbox.x1, next_box.bbox.x1),
                    max(current_box.bbox.y1, next_box.bbox.y1),
                )
                merged_box = Box(new_bbox)

                # Copy content from both boxes
                for line in current_box.get_all_lines():
                    merged_box.add_line(line)
                for line in next_box.get_all_lines():
                    merged_box.add_line(line)
                for image in current_box.get_all_images():
                    merged_box.add_image(image)
                for image in next_box.get_all_images():
                    merged_box.add_image(image)

                current_box = merged_box
            else:
                # If different types, keep them separate but adjust positions
                next_box.bbox.y1 = min(next_box.bbox.y1, current_box.bbox.y0)
                merged.append(current_box)
                current_box = next_box
        else:
            merged.append(current_box)
            current_box = next_box

    merged.append(current_box)
    return merged
