import sys
import os
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.features.text_extraction.preprocess import orient as orient_mod
from app.features.text_extraction.preprocess import preprocess as preprocess_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gray(h=100, w=80):
    """Return a random uint8 grayscale image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=100, w=80):
    """Return a random uint8 BGR image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _bgra(h=100, w=80):
    """Return a random uint8 BGRA image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 4), dtype=np.uint8)


# ---------------------------------------------------------------------------
# orient module tests
# ---------------------------------------------------------------------------

class TestOrientRotate(unittest.TestCase):

    def test_rotate_returns_same_shape(self):
        binary = _gray(200, 200)
        rotated = orient_mod._rotate(binary, 15)
        self.assertEqual(rotated.shape, binary.shape)

    def test_rotate_zero_degrees_unchanged(self):
        binary = _gray(100, 100)
        rotated = orient_mod._rotate(binary, 0)
        np.testing.assert_array_equal(rotated, binary)

    def test_rotate_output_dtype(self):
        binary = _gray(100, 100)
        rotated = orient_mod._rotate(binary, 45)
        self.assertEqual(rotated.dtype, np.uint8)

    def test_rotate_non_square(self):
        binary = _gray(80, 200)
        rotated = orient_mod._rotate(binary, 10)
        self.assertEqual(rotated.shape, binary.shape)


class TestOrientScore(unittest.TestCase):

    def test_score_returns_numeric(self):
        binary = _gray(100, 100)
        score = orient_mod._score(binary, 0)
        self.assertIsInstance(float(score), float)

    def test_score_non_negative(self):
        binary = _gray(100, 100)
        self.assertGreaterEqual(orient_mod._score(binary, 0), 0.0)

    def test_score_varies_with_angle(self):
        """Score should not be identical for all angles on a non-uniform image."""
        # Build an image with clear horizontal lines so variance changes with angle
        binary = np.full((100, 100), 255, dtype=np.uint8)
        binary[20::20, :] = 0
        scores = {a: orient_mod._score(binary, a) for a in (0, 30, 60)}
        self.assertGreater(len(set(scores.values())), 1)


class TestOrient(unittest.TestCase):

    def test_returns_tuple_of_two(self):
        binary = _gray(100, 100)
        result = orient_mod.orient(binary)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_output_image_same_shape(self):
        binary = _gray(100, 100)
        corrected, _ = orient_mod.orient(binary)
        self.assertEqual(corrected.shape, binary.shape)

    def test_output_image_dtype_uint8(self):
        binary = _gray(100, 100)
        corrected, _ = orient_mod.orient(binary)
        self.assertEqual(corrected.dtype, np.uint8)

    def test_angle_is_float(self):
        binary = _gray(100, 100)
        _, angle = orient_mod.orient(binary)
        self.assertIsInstance(angle, float)

    def test_horizontal_lines_returns_zero_angle(self):
        """An image with clear horizontal lines should need ~0° correction."""
        binary = np.full((100, 200), 255, dtype=np.uint8)
        for row in range(10, 90, 15):
            binary[row, :] = 0
        _, angle = orient_mod.orient(binary)
        self.assertAlmostEqual(angle, 0.0, places=0)

    def test_wide_image_still_works(self):
        """Images wider than 800px should be handled without error."""
        binary = _gray(200, 1000)
        corrected, angle = orient_mod.orient(binary)
        self.assertIsNotNone(corrected)
        self.assertIsInstance(angle, float)

    def test_near_zero_angle_skips_rotation(self):
        """When best angle < 0.5°, the original binary is returned unchanged."""
        binary = np.full((100, 200), 255, dtype=np.uint8)
        for row in range(10, 90, 15):
            binary[row, :] = 0
        corrected, angle = orient_mod.orient(binary)
        self.assertEqual(angle, 0.0)
        np.testing.assert_array_equal(corrected, binary)


# ---------------------------------------------------------------------------
# preprocess helpers
# ---------------------------------------------------------------------------

class TestToGrayscale(unittest.TestCase):

    def test_passthrough_for_already_gray(self):
        img = _gray()
        result = preprocess_mod._to_grayscale(img)
        self.assertEqual(result.ndim, 2)
        np.testing.assert_array_equal(result, img)

    def test_bgr_to_gray(self):
        img = _bgr()
        result = preprocess_mod._to_grayscale(img)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (img.shape[0], img.shape[1]))

    def test_bgra_to_gray(self):
        img = _bgra()
        result = preprocess_mod._to_grayscale(img)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (img.shape[0], img.shape[1]))


class TestNormalizeScale(unittest.TestCase):

    def test_scales_up_to_target_height(self):
        gray = _gray(500, 400)
        result = preprocess_mod._normalize_scale(gray, target_dpi_height=1000)
        self.assertEqual(result.shape[0], 1000)

    def test_scales_down_to_target_height(self):
        gray = _gray(2000, 1600)
        result = preprocess_mod._normalize_scale(gray, target_dpi_height=1000)
        self.assertEqual(result.shape[0], 1000)

    def test_passthrough_when_already_target(self):
        gray = _gray(1000, 800)
        result = preprocess_mod._normalize_scale(gray, target_dpi_height=1000)
        np.testing.assert_array_equal(result, gray)

    def test_output_dtype_preserved(self):
        gray = _gray(200, 160)
        result = preprocess_mod._normalize_scale(gray, target_dpi_height=400)
        self.assertEqual(result.dtype, np.uint8)


class TestDenoise(unittest.TestCase):

    def test_output_shape_unchanged(self):
        gray = _gray(100, 80)
        self.assertEqual(preprocess_mod._denoise(gray).shape, gray.shape)

    def test_output_dtype_uint8(self):
        gray = _gray(100, 80)
        self.assertEqual(preprocess_mod._denoise(gray).dtype, np.uint8)


class TestNormalizeBackground(unittest.TestCase):

    def test_output_shape_unchanged(self):
        gray = _gray(100, 80)
        self.assertEqual(preprocess_mod._normalize_background(gray).shape, gray.shape)

    def test_output_dtype_uint8(self):
        gray = _gray(100, 80)
        self.assertEqual(preprocess_mod._normalize_background(gray).dtype, np.uint8)

    def test_values_in_valid_range(self):
        gray = _gray(100, 80)
        result = preprocess_mod._normalize_background(gray)
        self.assertGreaterEqual(int(result.min()), 0)
        self.assertLessEqual(int(result.max()), 255)


class TestBinarize(unittest.TestCase):

    def test_output_shape(self):
        gray = _gray(200, 160)
        self.assertEqual(preprocess_mod._binarize(gray).shape, gray.shape)

    def test_only_binary_values(self):
        gray = _gray(200, 160)
        unique = set(np.unique(preprocess_mod._binarize(gray)).tolist())
        self.assertTrue(unique.issubset({0, 255}))


class TestMorphologicalClean(unittest.TestCase):

    def test_output_shape_unchanged(self):
        binary = np.zeros((200, 160), dtype=np.uint8)
        binary[50:150, 40:120] = 255
        self.assertEqual(preprocess_mod._morphological_clean(binary).shape, binary.shape)

    def test_output_dtype_uint8(self):
        binary = np.zeros((200, 160), dtype=np.uint8)
        self.assertEqual(preprocess_mod._morphological_clean(binary).dtype, np.uint8)


class TestInvertIfDarkBackground(unittest.TestCase):

    def test_all_black_gets_inverted(self):
        binary = np.zeros((100, 100), dtype=np.uint8)
        result = preprocess_mod._invert_if_dark_background(binary)
        self.assertGreater(np.sum(result == 255) / result.size, 0.5)

    def test_all_white_not_inverted(self):
        binary = np.full((100, 100), 255, dtype=np.uint8)
        result = preprocess_mod._invert_if_dark_background(binary)
        np.testing.assert_array_equal(result, binary)


class TestSeparateCharacters(unittest.TestCase):

    def test_output_shape_unchanged(self):
        binary = _gray(200, 160)
        self.assertEqual(preprocess_mod._separate_characters(binary).shape, binary.shape)

    def test_output_dtype_uint8(self):
        self.assertEqual(preprocess_mod._separate_characters(_gray(200, 160)).dtype, np.uint8)


# ---------------------------------------------------------------------------
# preprocess() pipeline (end-to-end)
# ---------------------------------------------------------------------------

class TestPreprocessPipeline(unittest.TestCase):

    def test_bgr_input_returns_ndarray(self):
        self.assertIsInstance(preprocess_mod.preprocess(_bgr(200, 160)), np.ndarray)

    def test_gray_input_returns_ndarray(self):
        self.assertIsInstance(preprocess_mod.preprocess(_gray(200, 160)), np.ndarray)

    def test_bgra_input_returns_ndarray(self):
        self.assertIsInstance(preprocess_mod.preprocess(_bgra(200, 160)), np.ndarray)

    def test_output_is_2d(self):
        self.assertEqual(preprocess_mod.preprocess(_bgr(200, 160)).ndim, 2)

    def test_output_dtype_uint8(self):
        self.assertEqual(preprocess_mod.preprocess(_bgr(200, 160)).dtype, np.uint8)

    def test_save_output_calls_imwrite_three_times(self):
        with patch("cv2.imwrite") as mock_write:
            preprocess_mod.preprocess(_bgr(200, 160), save_output=True, frame_number=7)
            self.assertEqual(mock_write.call_count, 3)
            names = [c.args[0] for c in mock_write.call_args_list]
            self.assertTrue(any("denoised_7" in n for n in names))
            self.assertTrue(any("binarized_7" in n for n in names))
            self.assertTrue(any("cleaned_7" in n for n in names))

    def test_no_save_never_calls_imwrite(self):
        with patch("cv2.imwrite") as mock_write:
            preprocess_mod.preprocess(_bgr(200, 160), save_output=False)
            mock_write.assert_not_called()


if __name__ == "__main__":
    unittest.main()