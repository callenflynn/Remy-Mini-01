#!/usr/bin/env python3
"""Tests for wordnet_to_jsonl.py. Stdlib only: `python3 scripts/test_wordnet_to_jsonl.py`."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import wordnet_to_jsonl as w  # noqa: E402


class RenderTests(unittest.TestCase):
    def test_full_entry_reads_like_a_dictionary_line(self):
        text = w.render(
            {
                "partOfSpeech": "a",
                "members": ["able"],
                "definition": ["having the necessary means or skill to do something"],
                "example": ["able to swim", "she was able to program"],
            }
        )
        self.assertEqual(
            text,
            "able (adjective): having the necessary means or skill to do something. "
            "Examples: able to swim; she was able to program.",
        )

    def test_example_given_as_object_uses_its_text(self):
        self.assertEqual(
            w.render(
                {
                    "partOfSpeech": "n",
                    "members": ["cat", "true cat"],
                    "definition": ["feline mammal"],
                    "example": [{"source": "somewhere", "text": "the cat sat"}],
                }
            ),
            "cat, true cat (noun): feline mammal. Examples: the cat sat.",
        )

    def test_synset_without_definition_is_skipped(self):
        self.assertEqual(w.render({"partOfSpeech": "n", "members": ["x"]}), "")

    def test_multiple_definitions_are_joined_and_trailing_periods_dropped(self):
        text = w.render(
            {
                "partOfSpeech": "v",
                "members": ["run"],
                "definition": ["move fast.", "operate"],
            }
        )
        self.assertEqual(text, "run (verb): move fast. operate.")

    def test_unknown_part_of_speech_falls_back_to_its_code(self):
        self.assertEqual(w.render({"partOfSpeech": "zz", "members": ["y"], "definition": ["d"]}), "y (zz): d.")


class FileSelectionTests(unittest.TestCase):
    def test_only_lexicographer_class_files_are_read(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d)
            for name in ["noun.food.json", "adj.all.json", "entries-a.json", "frames.json"]:
                (src / name).write_text("{}", encoding="utf-8")
            self.assertEqual(
                [p.name for p in w.synset_files(src)], ["adj.all.json", "noun.food.json"]
            )


class EndToEndTests(unittest.TestCase):
    def run_script(self, src: Path) -> str:
        argv = sys.argv
        sys.argv = ["wordnet_to_jsonl.py", str(src)]
        try:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(w.main(), 0)
        finally:
            sys.argv = argv
        return (src / "wordnet.jsonl").read_text(encoding="utf-8")

    def test_output_is_valid_jsonl_deterministic_and_skips_empty_synsets(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d)
            (src / "noun.food.json").write_text(
                json.dumps(
                    {
                        "1": {"partOfSpeech": "n", "members": ["apple"], "definition": ["a fruit"]},
                        "2": {"partOfSpeech": "n", "members": ["void"]},
                    }
                ),
                encoding="utf-8",
            )
            (src / "verb.motion.json").write_text(
                json.dumps(
                    {"3": {"partOfSpeech": "v", "members": ["walk"], "definition": ["move on foot"]}}
                ),
                encoding="utf-8",
            )
            (src / "entries-a.json").write_text('{"ignored": {"definition": ["x"]}}', encoding="utf-8")

            first = self.run_script(src)
            second = self.run_script(src)
            self.assertEqual(first, second, "output must be deterministic")

            lines = [json.loads(line) for line in first.splitlines()]
            self.assertEqual([l["id"] for l in lines], ["1", "3"])
            self.assertTrue(all(isinstance(l["text"], str) and l["text"] for l in lines))
            self.assertIn("apple (noun): a fruit.", first)
            self.assertFalse(any("ignored" in l["id"] for l in lines))

    def test_missing_directory_is_an_error_not_a_crash(self):
        argv = sys.argv
        sys.argv = ["wordnet_to_jsonl.py", "/nonexistent/path"]
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(w.main(), 1)
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main(verbosity=2)
