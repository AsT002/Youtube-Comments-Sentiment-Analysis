import unittest

from youtube_sentiment.text import is_probably_english, normalize_label, normalize_text, normalize_text_lower, sentiment_tokenize


class TextTests(unittest.TestCase):
    def test_normalize_text_preserves_sentiment_signals(self):
        result = normalize_text("  @somebody  NOT good!!! 😡 https://example.com/x  ")
        self.assertEqual(result, "<USER> NOT good!!! 😡 <URL>")

    def test_casefold_and_sentiment_tokenizer_are_explicit(self):
        normalized = normalize_text_lower("Great GREAT! 😊 @Person")
        self.assertEqual(normalized, "great great! 😊 <user>")
        self.assertEqual(sentiment_tokenize(normalized), ["great", "great", "!", "😊", "<user>"])

    def test_normalize_label_supports_declared_schema(self):
        self.assertEqual(normalize_label("negative"), 0)
        self.assertEqual(normalize_label("NEUTRAL"), 1)
        self.assertEqual(normalize_label("2"), 2)
        self.assertIsNone(normalize_label("mixed"))
        self.assertIsNone(normalize_label(True))

    def test_english_prefilter_rejects_non_latin_only_text(self):
        self.assertTrue(is_probably_english("This is a useful comment"))
        self.assertFalse(is_probably_english("这是一个评论"))


if __name__ == "__main__":
    unittest.main()
