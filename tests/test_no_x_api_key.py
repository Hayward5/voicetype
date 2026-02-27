import os
import unittest


class TestNoXApiKey(unittest.TestCase):
    """Test that no Python source files contain the API key header string."""

    EXCLUDE_DIRS = {".git", ".sisyphus", "build", "dist", "__pycache__"}

    def test_no_x_api_key_in_source_files(self):
        """Fail if any .py file contains the API key header string."""
        # Build the needle at runtime to avoid literal in this file
        needle = "x" + "-api-key"
        offending_files = []

        for root, dirs, files in os.walk("."):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in self.EXCLUDE_DIRS]

            for filename in files:
                if filename.endswith(".py"):
                    filepath = os.path.join(root, filename)
                    # Skip this test file itself
                    if filename == "test_no_x_api_key.py":
                        continue
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                            if needle in content:
                                offending_files.append(filepath)
                    except (IOError, UnicodeDecodeError):
                        # Skip files that can't be read
                        continue

        if offending_files:
            self.fail(
                f"Found {needle!r} in {len(offending_files)} file(s):\n"
                + "\n".join(f"  - {f}" for f in offending_files)
            )


if __name__ == "__main__":
    unittest.main()
