#!/usr/bin/env python3
"""Interactive review tool for LLM-generated outputs.

This script provides a command-line interface to review, edit, and approve
LLM-generated steering texts, question rephrases, and ambiguity resolutions.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


class LLMOutputReviewer:
    """Interactive reviewer for LLM outputs."""

    def __init__(self, input_path: Path, output_path: Optional[Path] = None):
        """Initialize reviewer."""
        self.input_path = input_path
        self.output_path = output_path or input_path.with_suffix(".reviewed.json")
        self.data = self._load_data()
        self.current_index = 0
        self.reviewed_data = []
        self.stats = {"approved": 0, "edited": 0, "rejected": 0, "skipped": 0}

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(self.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]

    def _save_progress(self):
        """Save current progress to output file."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.reviewed_data, f, indent=2, ensure_ascii=False)
        LOGGER.info(f"Progress saved to {self.output_path}")

    def _display_item(self, item: Dict[str, Any], index: int):
        """Display an item for review."""
        print("\n" + "=" * 80)
        print(f"Item {index + 1}/{len(self.data)}")
        print("=" * 80)
        print(json.dumps(item, indent=2, ensure_ascii=False))
        print("=" * 80)

    def _get_user_action(self) -> str:
        """Get user action choice."""
        print("\nActions: [a]pprove [e]dit [r]eject [s]kip [q]uit [h]elp")
        while True:
            action = input("\nChoose action: ").strip().lower()
            if action in ["a", "e", "r", "s", "q", "h"]:
                return action
            print("Invalid action.")


if __name__ == "__main__":
    print("LLM Output Review Tool")
