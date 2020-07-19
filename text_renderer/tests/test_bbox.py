from text_renderer.utils.bbox import BBox


def test_offset():
    bbox1 = BBox(0, 0, 100, 32)
    bbox2 = BBox(0, 0, 50, 16)

    bbox2.offset_(bbox2.left_cnt, bbox1.right_cnt)
    assert bbox2 == BBox(100, 8, 150, 24)
