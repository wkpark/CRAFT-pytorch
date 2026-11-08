"""getDetBoxes must give the boxes, labels, mapper and polygons the original whole-map version gave.

Random heatmaps cover the edge cases: words on the map's edges, slanted words, link areas reaching
out of words and speckle under the size filter.
"""

import math

import cv2
import numpy as np
import pytest
from scipy.ndimage import label

from craft.craft_utils import getDetBoxes, getPoly_core


def reference_getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, estimate_num_chars=False):
    """getDetBoxes_core before it worked in windows, unchanged."""
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    det = []
    mapper = []
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue
        if np.max(textmap[labels == k]) < text_threshold:
            continue
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        if estimate_num_chars:
            _, character_locs = cv2.threshold((textmap - linkmap) * segmap / 255., text_threshold, 1, 0)
            _, n_chars = label(character_locs)
            mapper.append(n_chars)
        else:
            mapper.append(k)
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)
        det.append(box)
    return det, labels, mapper


def _heatmaps(rng, h, w):
    text = np.zeros((h, w), np.float32)
    link = np.zeros((h, w), np.float32)
    for _ in range(rng.integers(5, 60)):
        y, x = rng.integers(-5, h), rng.integers(-5, w)
        bh, bw = rng.integers(2, 30), rng.integers(2, 60)
        text[max(y, 0):y + bh, max(x, 0):x + bw] = rng.uniform(0.3, 1.0)
        if rng.random() < 0.5:  # a link reaching out of the word
            link[max(y, 0):y + bh, max(x + bw // 2, 0):x + bw + rng.integers(0, 20)] = rng.uniform(0.3, 1.0)
        if rng.random() < 0.4:  # a slanted word
            corners = cv2.boxPoints(((float(x), float(y)), (float(bw), float(bh)), float(rng.uniform(5, 85))))
            cv2.fillPoly(text, [corners.astype(np.int32)], float(rng.uniform(0.3, 1.0)))
    text += rng.uniform(0, 0.45, (h, w)).astype(np.float32) * (rng.random((h, w)) < 0.02)  # speckle
    return text, link


@pytest.mark.parametrize("estimate_num_chars", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_matches_whole_map_version(seed, estimate_num_chars):
    rng = np.random.default_rng(seed)
    textmap, linkmap = _heatmaps(rng, int(rng.integers(40, 300)), int(rng.integers(40, 300)))
    args = (textmap, linkmap, 0.7, 0.4, 0.4)
    ref_boxes, ref_labels, ref_mapper = reference_getDetBoxes_core(*args, estimate_num_chars=estimate_num_chars)
    boxes, polys, mapper = getDetBoxes(*args, poly=not estimate_num_chars, estimate_num_chars=estimate_num_chars)
    assert mapper == ref_mapper
    assert len(boxes) == len(ref_boxes)
    for box, ref in zip(boxes, ref_boxes):
        assert box.dtype == ref.dtype
        np.testing.assert_array_equal(box, ref)
    if not estimate_num_chars:
        ref_polys = getPoly_core(ref_boxes, ref_labels, ref_mapper, linkmap)
        for poly, ref in zip(polys, ref_polys):
            assert (poly is None) == (ref is None)
            if ref is not None:
                np.testing.assert_array_equal(poly, ref)
