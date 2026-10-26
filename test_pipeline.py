import os
import unittest
import tempfile
import cv2
import numpy as np
from PIL import Image

from note_compiler import (
    SyntheticNoteGenerator,
    NoteCompilerPipeline,
    ProcessingConfig,
    NotePDFCompiler,
    compute_image_metrics,
    ensure_bgr_image,
    stage_1_rectify,
    stage_2_normalize,
    stage_3_color_preserve,
    stage_4_denoise_sharpen,
    stage_5_compile_pdf,
)

class TestNoteCompilerPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = cls._temp_dir.name

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def test_01_synthetic_generator_variants(self):
        for n_type in ["lined_paper", "whiteboard", "chalkboard"]:
            img = SyntheticNoteGenerator.generate(
                width=800,
                height=600,
                note_type=n_type,
                add_shadow=True,
                add_perspective=True,
                seed=42,
            )
            self.assertEqual(img.shape, (600, 800, 3))
            self.assertEqual(img.dtype, np.uint8)
            self.assertGreater(img.mean(), 10.0)

    def test_02_stage_1_perspective_rectification(self):
        raw = SyntheticNoteGenerator.generate(width=800, height=600, add_perspective=True, seed=12)
        pipeline = NoteCompilerPipeline()
        res1 = pipeline.stage_1_rectify_perspective(raw)
        self.assertIsNotNone(res1.image)
        self.assertEqual(res1.image.ndim, 3)
        self.assertGreater(res1.execution_time_ms, 0)
        self.assertIn("quad_detected", res1.metadata)

    def test_03_stage_2_illumination_methods(self):
        raw = SyntheticNoteGenerator.generate(width=600, height=450, add_shadow=True, add_perspective=False, seed=99)
        pipeline = NoteCompilerPipeline()

        pipeline.config.illumination_method = "division"
        res_div = pipeline.stage_2_normalize_illumination(raw)
        self.assertEqual(res_div.image.shape, raw.shape)

        pipeline.config.illumination_method = "homomorphic"
        res_homo = pipeline.stage_2_normalize_illumination(raw)
        self.assertEqual(res_homo.image.shape, raw.shape)

        pipeline.config.illumination_method = "hybrid"
        res_hyb = pipeline.stage_2_normalize_illumination(raw)
        self.assertEqual(res_hyb.image.shape, raw.shape)

    def test_04_stage_3_color_preservation_and_bw(self):
        raw = SyntheticNoteGenerator.generate(width=600, height=450, add_shadow=False, add_perspective=False, seed=77)
        pipeline = NoteCompilerPipeline()

        pipeline.config.mode = "color_preserve"
        res_color = pipeline.stage_3_color_preservation_and_binarize(raw)
        self.assertEqual(res_color.image.dtype, np.uint8)

        pipeline.config.mode = "black_and_white"
        res_bw = pipeline.stage_3_color_preservation_and_binarize(raw)
        self.assertEqual(res_bw.image.dtype, np.uint8)

    def test_05_stage_4_denoise_and_sharpen(self):
        raw = SyntheticNoteGenerator.generate(width=600, height=450, add_shadow=False, add_perspective=False, seed=55)
        pipeline = NoteCompilerPipeline()
        res4 = pipeline.stage_4_denoise_and_sharpen(raw)
        self.assertEqual(res4.image.shape, raw.shape)
        self.assertEqual(res4.image.dtype, np.uint8)

    def test_06_stage_5_pdf_compilation(self):
        img1 = SyntheticNoteGenerator.generate(width=600, height=450, note_type="lined_paper", seed=1)
        img2 = SyntheticNoteGenerator.generate(width=600, height=450, note_type="whiteboard", seed=2)

        pdf_path = os.path.join(self.test_dir, "test_output.pdf")
        stats = NotePDFCompiler.compile_pdf(
            images=[img1, img2],
            output_pdf_path=pdf_path,
            title="Automated Test PDF",
        )

        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 1024)
        self.assertEqual(stats["total_pages"], 2)
        self.assertGreater(stats["compression_ratio_pct"], 0)

    def test_07_chalkboard_inversion(self):
        chalk_raw = SyntheticNoteGenerator.generate(
            width=600,
            height=450,
            note_type="chalkboard",
            add_shadow=True,
            add_perspective=False,
            seed=33,
        )
        pipeline = NoteCompilerPipeline(ProcessingConfig(mode="chalkboard_invert", auto_perspective=False))
        result = pipeline.process_image(chalk_raw)
        self.assertTrue(result.success)
        self.assertGreater(result.final_image.mean(), 180)

    def test_08_grayscale_and_rgba_channels(self):
        # 2D grayscale input
        gray = np.full((300, 400), 200, dtype=np.uint8)
        cv2.putText(gray, "Test Grayscale", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 30, 2)
        pipeline = NoteCompilerPipeline(ProcessingConfig(auto_perspective=False))
        res_gray = pipeline.process_image(gray)
        self.assertTrue(res_gray.success)
        self.assertEqual(res_gray.final_image.ndim, 3)
        self.assertEqual(res_gray.final_image.shape, (300, 400, 3))

        # 4-channel BGRA input (with alpha transparency)
        bgra = np.full((300, 400, 4), 220, dtype=np.uint8)
        bgra[:, :, 3] = 255
        cv2.circle(bgra, (150, 150), 50, (0, 0, 255, 255), -1)
        res_bgra = pipeline.process_image(bgra)
        self.assertTrue(res_bgra.success)
        self.assertEqual(res_bgra.final_image.shape, (300, 400, 3))

    def test_09_empty_and_corrupt_inputs(self):
        pipeline = NoteCompilerPipeline()
        with self.assertRaises(ValueError):
            pipeline.process_image(np.array([], dtype=np.uint8))
        with self.assertRaises(ValueError):
            pipeline.process_image(None)

    def test_10_extreme_and_nonstandard_dimensions(self):
        pipeline = NoteCompilerPipeline(ProcessingConfig(auto_perspective=False))

        # Tiny square image
        tiny = np.full((32, 32, 3), 220, dtype=np.uint8)
        res_tiny = pipeline.process_image(tiny)
        self.assertTrue(res_tiny.success)
        self.assertEqual(res_tiny.final_image.shape, (32, 32, 3))

        # Wide aspect ratio image
        wide = np.full((80, 800, 3), 230, dtype=np.uint8)
        res_wide = pipeline.process_image(wide)
        self.assertTrue(res_wide.success)
        self.assertEqual(res_wide.final_image.shape, (80, 800, 3))

    def test_11_perspective_fallback_on_flat_image(self):
        # A completely flat image has no quadrilateral boundary
        flat = np.full((400, 400, 3), 200, dtype=np.uint8)
        out, detected, meta = stage_1_rectify(flat, auto=True, return_meta=True)
        self.assertFalse(detected)
        self.assertEqual(out.shape, flat.shape)
        self.assertIn("reason", meta)

    def test_12_denoise_median_filter_strength(self):
        # Image with synthetic salt-and-pepper noise
        noisy = np.full((200, 200, 3), 220, dtype=np.uint8)
        np.random.seed(42)
        noise_mask = np.random.rand(200, 200) < 0.05
        noisy[noise_mask] = 0

        denoised = stage_4_denoise_sharpen(noisy, median_k=3, boost=0.0)
        # Median filter should eliminate isolated pepper noise pixels
        black_pixels_before = np.count_nonzero(np.all(noisy == 0, axis=2))
        black_pixels_after = np.count_nonzero(np.all(denoised == 0, axis=2))
        self.assertLess(black_pixels_after, black_pixels_before)

    def test_13_pdf_page_sizes_a4_and_letter(self):
        img = SyntheticNoteGenerator.generate(width=500, height=400, seed=10)

        # A4 compilation
        pdf_a4 = os.path.join(self.test_dir, "test_a4.pdf")
        stats_a4 = stage_5_compile_pdf([img], pdf_a4, page_size="A4")
        self.assertTrue(os.path.exists(pdf_a4))
        self.assertEqual(stats_a4["page_size"], "A4")

        # Letter compilation
        pdf_letter = os.path.join(self.test_dir, "test_letter.pdf")
        stats_letter = stage_5_compile_pdf([img], pdf_letter, page_size="Letter")
        self.assertTrue(os.path.exists(pdf_letter))
        self.assertEqual(stats_letter["page_size"], "Letter")

        # Original compilation
        pdf_orig = os.path.join(self.test_dir, "test_orig.pdf")
        stats_orig = stage_5_compile_pdf([img], pdf_orig, page_size="Original")
        self.assertTrue(os.path.exists(pdf_orig))
        self.assertEqual(stats_orig["page_size"], "Original")

    def test_14_grayscale_and_adaptive_modes(self):
        img = SyntheticNoteGenerator.generate(width=400, height=300, seed=15)

        # Grayscale mode
        p_gray = NoteCompilerPipeline(ProcessingConfig(mode="grayscale", auto_perspective=False))
        r_gray = p_gray.process_image(img)
        self.assertTrue(r_gray.success)
        # In grayscale, all 3 channels should be identical
        ch0, ch1, ch2 = cv2.split(r_gray.final_image)
        np.testing.assert_array_equal(ch0, ch1)
        np.testing.assert_array_equal(ch1, ch2)

        # Black and white with adaptive threshold
        p_adapt = NoteCompilerPipeline(ProcessingConfig(mode="black_and_white", adaptive_threshold=True, auto_perspective=False))
        r_adapt = p_adapt.process_image(img)
        self.assertTrue(r_adapt.success)
        unique_vals = np.unique(r_adapt.final_image)
        # Should only contain binary values (0 and 255)
        self.assertTrue(set(unique_vals).issubset({0, 255}))

    def test_15_image_metrics_calculation(self):
        # Plain white image has near zero Laplacian variance and high white ratio
        blank = np.full((200, 200, 3), 255, dtype=np.uint8)
        m_blank = compute_image_metrics(blank)
        self.assertEqual(m_blank["white_background_pct"], 100.0)
        self.assertEqual(m_blank["contrast_std"], 0.0)

        # Image with high-contrast text has positive sharpness index and contrast
        text_img = blank.copy()
        cv2.putText(text_img, "CRISP TEXT", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        m_text = compute_image_metrics(text_img)
        self.assertGreater(m_text["sharpness_index"], 0.0)
        self.assertGreater(m_text["contrast_std"], 0.0)

    def test_16_tonal_parameter_inversion_safety(self):
        # Inverted white_point and black_point should not cause ZeroDivisionError
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        out = stage_3_color_preserve(img, white_pt=50.0, black_pt=200.0)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_17_input_immutability_and_memory_safety(self):
        orig = SyntheticNoteGenerator.generate(width=400, height=300, seed=7)
        copy_orig = orig.copy()
        pipeline = NoteCompilerPipeline()
        _ = pipeline.process_image(orig)
        # Ensure original input buffer was not modified in-place
        np.testing.assert_array_equal(orig, copy_orig)

    def test_18_multi_page_pdf_ordering_and_integrity(self):
        pages = [
            SyntheticNoteGenerator.generate(width=300, height=200, note_type="lined_paper", seed=i)
            for i in range(4)
        ]
        out_pdf = os.path.join(self.test_dir, "multipage_test.pdf")
        stats = stage_5_compile_pdf(pages, out_pdf, compress=True, quality=80)

        self.assertEqual(stats["total_pages"], 4)
        self.assertTrue(os.path.exists(out_pdf))
        self.assertGreater(os.path.getsize(out_pdf), 1000)

        # Verify PDF header and page objects in PDF stream
        with open(out_pdf, "rb") as f:
            pdf_data = f.read()
        self.assertTrue(pdf_data.startswith(b"%PDF"))
        page_objects_count = pdf_data.count(b"/Type /Page") - pdf_data.count(b"/Type /Pages")
        self.assertEqual(page_objects_count, 4)

    def test_19_ensure_bgr_image_helper(self):
        # 3-channel returns unchanged
        img3 = np.zeros((50, 50, 3), dtype=np.uint8)
        self.assertEqual(ensure_bgr_image(img3).shape, (50, 50, 3))

        # 2-channel raises ValueError
        with self.assertRaises(ValueError):
            ensure_bgr_image(np.zeros((50, 50, 2), dtype=np.uint8))

    def test_20_config_overrides_and_defaults(self):
        cfg_default = ProcessingConfig()
        self.assertEqual(cfg_default.mode, "color_preserve")
        self.assertEqual(cfg_default.white_point, 210.0)
        self.assertEqual(cfg_default.black_point, 60.0)
        self.assertEqual(cfg_default.denoise_ksize, 1)

        cfg_custom = ProcessingConfig(mode="grayscale", denoise_ksize=3, white_point=220.0)
        self.assertEqual(cfg_custom.mode, "grayscale")
        self.assertEqual(cfg_custom.denoise_ksize, 3)
        self.assertEqual(cfg_custom.white_point, 220.0)


if __name__ == "__main__":
    unittest.main()
