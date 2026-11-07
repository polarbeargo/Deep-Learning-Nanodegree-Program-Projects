"""Exercise the actual Streamlit UI without calling a generator API."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class TestStreamlitDemo(unittest.TestCase):
    def test_all_retrievers_and_shared_metrics(self):
        app_path = Path(__file__).resolve().parents[1] / "streamlit" / "app.py"
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            app = AppTest.from_file(str(app_path), default_timeout=180).run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertFalse(app.error, [error.value for error in app.error])
            self.assertFalse(app.sidebar.checkbox[0].value)
            app.sidebar.slider[0].set_value(5)
            app.text_input[0].set_value("What products does TechFlow AI offer?").run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertFalse(app.error, [error.value for error in app.error])
            rendered = "\n".join(element.value for element in app.markdown)
            for method in ("BM25", "Word2Vec", "Transformer"):
                self.assertIn(method, rendered)
            self.assertEqual(rendered.count("Rank 5:"), 3)
            self.assertEqual(rendered.count("Precision@5:"), 3)
            self.assertEqual(rendered.count("Recall@5:"), 3)
            self.assertEqual(rendered.count("Reciprocal Rank@5:"), 3)


if __name__ == "__main__":
    unittest.main()
