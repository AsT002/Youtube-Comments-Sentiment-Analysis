import unittest

from sklearn.exceptions import NotFittedError

from youtube_sentiment.train_linear_svm import build_svm_pipeline


class SVMTrainingTests(unittest.TestCase):
    def test_feature_extraction_stays_inside_unfitted_cv_pipeline(self):
        model = build_svm_pipeline(0.25, word_features=100, character_features=100, seed=42)
        self.assertEqual(list(model.named_steps), ["features", "classifier"])
        word_vectorizer = dict(model.named_steps["features"].transformer_list)["word"]
        with self.assertRaises(NotFittedError):
            word_vectorizer.transform(["example"])

    def test_word_features_casefold_and_keep_sentiment_tokens(self):
        model = build_svm_pipeline(0.25, word_features=100, character_features=100, seed=42)
        word_vectorizer = dict(model.named_steps["features"].transformer_list)["word"]
        features = word_vectorizer.build_analyzer()("Great GREAT! 😊")
        self.assertEqual(features[:4], ["great", "great", "!", "😊"])
        self.assertNotIn("Great", features)


if __name__ == "__main__":
    unittest.main()
