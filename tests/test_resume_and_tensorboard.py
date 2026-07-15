import tempfile
import unittest
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

from main import parse_args
from monarch_distill.config import default_config
from monarch_distill.io import consolidate_tensorboard_scalars


class ResumeCliTest(unittest.TestCase):
    def test_resume_arguments_are_paired(self):
        args = parse_args(["--resume-from-checkpoint", "checkpoint.pt", "--resume-start-module-index", "4"])
        self.assertEqual(args.resume_from_checkpoint, "checkpoint.pt")
        self.assertEqual(args.resume_start_module_index, 4)

    def test_all35_defaults_are_fresh(self):
        config = default_config()
        self.assertEqual(config.max_modules, 35)
        self.assertIsNone(config.resume_from_checkpoint)
        self.assertEqual(config.resume_start_module_index, 0)
        self.assertIn("all35mlp", config.tensorboard_log_dir)
        self.assertIn("all35mlp", config.save_dir)


class TensorBoardConsolidationTest(unittest.TestCase):
    def test_latest_duplicate_wins_and_output_is_one_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            output = root / "canonical"

            writer = SummaryWriter(first)
            writer.add_scalar("ValidationLoss/eval512_distill", 1.2, 1, walltime=10.0)
            writer.add_scalar("Phase2/layer34_kl_loss", 2.0, 0, walltime=11.0)
            writer.close()

            writer = SummaryWriter(second)
            writer.add_scalar("Phase2/layer34_kl_loss", 1.5, 0, walltime=20.0)
            writer.add_scalar("ValidationLoss/eval512_distill", 1.0, 2, walltime=21.0)
            writer.close()

            result = consolidate_tensorboard_scalars([first, second], output)
            self.assertEqual(result["scalar_count"], 3)
            self.assertEqual(len(list(output.glob("events.out.tfevents.*"))), 1)

            accumulator = event_accumulator.EventAccumulator(
                str(output),
                size_guidance={event_accumulator.SCALARS: 0},
            )
            accumulator.Reload()
            validation = accumulator.Scalars("ValidationLoss/eval512_distill")
            self.assertEqual([event.step for event in validation], [1, 2])
            self.assertEqual([round(event.value, 4) for event in validation], [1.2, 1.0])
            phase2 = accumulator.Scalars("Phase2/layer34_kl_loss")
            self.assertEqual(len(phase2), 1)
            self.assertAlmostEqual(phase2[0].value, 1.5)


if __name__ == "__main__":
    unittest.main()
