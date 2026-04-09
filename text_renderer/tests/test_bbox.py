import numpy as np

from text_renderer.utils.bbox import BBox, CharBBox


def test_offset():
    bbox1 = BBox(0, 0, 100, 32)
    bbox2 = BBox(0, 0, 50, 16)

    bbox2.offset_(bbox2.left_cnt, bbox1.right_cnt)
    assert bbox2 == BBox(100, 8, 150, 24)


def test_charbbox_scale_encloses_non_integer_result():
    """scale() must return a box that encloses the scaled glyph; int()
    truncation toward zero would under-bound negative left after scaling
    (e.g. -1*0.5=-0.5 -> 0) and fractional right (21*0.5=10.5 -> 10)."""
    bb = CharBBox("x", -1, 0, 21, 10)
    scaled = bb.scale(0.5, 0.5)
    assert scaled.left <= -1 * 0.5
    assert scaled.top <= 0 * 0.5
    assert scaled.right >= 21 * 0.5
    assert scaled.bottom >= 10 * 0.5


def test_charbbox_from_pnts_encloses_non_integer_points():
    """from_pnts must return a box that encloses all supplied points;
    int() truncation toward zero would under-bound negatives (-0.4 -> 0)
    and max fractions (20.9 -> 20), clipping transformed character boxes."""
    pnts = np.array(
        [
            [-0.4, 0.2],
            [20.9, 0.2],
            [20.9, 10.7],
            [-0.4, 10.7],
        ]
    )
    bb = CharBBox.from_pnts("x", pnts)
    assert bb.left <= -0.4
    assert bb.top <= 0.2
    assert bb.right >= 20.9
    assert bb.bottom >= 10.7
