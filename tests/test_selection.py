import unittest

from youtube_sentiment.select_model import choose_candidate


def candidate(name, tweet_f1):
    return {"name": name, "kind": "test", "artifact": name, "metrics": {"tweeteval": {"macro_f1": tweet_f1}}}


class SelectionTests(unittest.TestCase):
    def test_highest_tweeteval_macro_f1_wins(self):
        selected = choose_candidate([candidate("baseline_v2", 0.64), candidate("model_c_v2", 0.68)])
        self.assertEqual(selected["name"], "model_c_v2")

    def test_selection_requires_a_candidate(self):
        with self.assertRaises(RuntimeError):
            choose_candidate([])


if __name__ == "__main__":
    unittest.main()
