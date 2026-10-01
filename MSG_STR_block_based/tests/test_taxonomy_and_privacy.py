import unittest

import pandas as pd

from msg_str.config import PrivacyConfig
from msg_str.llm_annotation import parse_multi_label, parse_single_label
from msg_str.privacy import require_text_allowed
from msg_str.taxonomy import build_seed_topic_list, normalize_text


class TaxonomyAndPrivacyTests(unittest.TestCase):
    def test_seed_topics_are_deduplicated_and_grouped(self) -> None:
        seed = pd.DataFrame(
            {
                "main": ["Cardiovascular", "Cardiovascular", "Neurologic"],
                "sub1": ["Blood pressure", "Blood pressure", "Dizziness"],
            }
        )
        labels, keywords, deduplicated = build_seed_topic_list(seed)
        self.assertEqual(labels, ["Cardiovascular", "Neurologic"])
        self.assertEqual(len(deduplicated), 2)
        self.assertIn("pressure", keywords[0])

    def test_safe_mode_blocks_message_text(self) -> None:
        with self.assertRaises(RuntimeError):
            require_text_allowed(PrivacyConfig(safe_mode=True, allow_text_input=False))

    def test_label_parsers_enforce_ranges_and_uniqueness(self) -> None:
        label, reason = parse_single_label("Classification: 2\nReason: symptom update")
        self.assertEqual(label, 2)
        self.assertEqual(reason, "symptom update")

        labels, _ = parse_multi_label("Classification: 14, 14, 35, 99, 101\nReason: x")
        self.assertEqual(labels, [14, 35, 99])

    def test_normalize_text_handles_missing_values(self) -> None:
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text("  blood   pressure "), "blood pressure")


if __name__ == "__main__":
    unittest.main()
