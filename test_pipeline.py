import os
import unittest
import tempfile
import numpy as np

from note_compiler import (
    SyntheticNoteGenerator,
    NoteCompilerPipeline,
    ProcessingConfig,
    NotePDFCompiler,
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


if __name__ == "__main__":
    unittest.main()
