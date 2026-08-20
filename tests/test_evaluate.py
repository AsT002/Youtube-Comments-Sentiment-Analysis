import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from youtube_sentiment import evaluate


class FinalEvaluationTests(unittest.TestCase):
    def test_any_existing_dataset_report_keeps_test_globally_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            (reports / "final_tweeteval_old_model_metrics.json").write_text("{}", encoding="utf-8")
            manifest = reports / "manifest.json"
            manifest.write_text('{"name":"new_model","artifact":"unused"}', encoding="utf-8")
            with patch.object(evaluate, "REPORTS_DIR", reports):
                with self.assertRaises(SystemExit) as context:
                    evaluate.main(["tweeteval", "--manifest", str(manifest), "--confirm-final-test"])
            self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
