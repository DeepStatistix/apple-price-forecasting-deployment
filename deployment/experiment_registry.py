"""Registry: maps market/variety/grade to data files and NHITS model paths.

Scans the experiments/ folder to find the latest NHITS .darts.pt model
for each deployed market/variety/grade combination.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DataEntry:
    """Entry pointing to a CSV dataset and optionally a saved NHITS model."""
    market: str
    variety: str
    grade: str
    data_path: str
    nhits_model_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Markets, varieties, and grades to expose in the UI
# ---------------------------------------------------------------------------
DEPLOYED_MARKETS = {"Azadpur", "Shopian", "Sopore"}
DEPLOYED_VARIETIES = {"American", "Delicious"}
DEPLOYED_GRADES = {"A", "B"}


def _build_data_path(project_root: str, market: str, variety: str, grade: str) -> str:
    """Build path to the processed CSV dataset."""
    filename = f"{variety}_{grade}_dataset.csv"
    return os.path.join(project_root, "data", "raw", "processed", market, filename)


def _find_nhits_model(experiments_root: str, market: str, variety: str, grade: str) -> Optional[str]:
    """Find the latest NHITS .darts.pt model in the experiments folder.

    Looks for experiment folders matching the pattern:
        {timestamp}_{market}_{variety}_{grade}_all_precut_*
    and returns the path to the latest nhits_*.darts.pt file.
    """
    if not os.path.isdir(experiments_root):
        return None

    pattern = re.compile(rf"^\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}_{market}_{variety}_{grade}_all_precut_.*$")

    matching_dirs = []
    for d in os.listdir(experiments_root):
        if pattern.match(d):
            matching_dirs.append(d)

    if not matching_dirs:
        return None

    # Sort to get the latest experiment (timestamp is at the start)
    matching_dirs.sort()
    latest_dir = matching_dirs[-1]
    models_dir = os.path.join(experiments_root, latest_dir, "models")

    if not os.path.isdir(models_dir):
        return None

    # Find the latest nhits_*.darts.pt file
    nhits_files = [f for f in os.listdir(models_dir) if f.startswith("nhits_") and f.endswith(".darts.pt")]
    if not nhits_files:
        return None

    nhits_files.sort()
    return os.path.join(models_dir, nhits_files[-1])


class ExperimentRegistry:
    """Registry that maps market/variety/grade to data files and NHITS models."""

    def __init__(self, experiments_root: str = "experiments"):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.experiments_root = os.path.join(self.project_root, experiments_root)
        self.entries: List[DataEntry] = []
        self._index: Dict[str, DataEntry] = {}
        self._build_index()

    def _build_index(self):
        """Create entries for all deployed market/variety/grade combinations."""
        for market in DEPLOYED_MARKETS:
            for variety in DEPLOYED_VARIETIES:
                for grade in DEPLOYED_GRADES:
                    path = _build_data_path(self.project_root, market, variety, grade)
                    if os.path.exists(path):
                        nhits_path = _find_nhits_model(self.experiments_root, market, variety, grade)
                        entry = DataEntry(
                            market=market,
                            variety=variety,
                            grade=grade,
                            data_path=path,
                            nhits_model_path=nhits_path,
                        )
                        self.entries.append(entry)
                        key = f"{market}_{variety}_{grade}"
                        self._index[key] = entry

    def get_entry(self, market: str, variety: str, grade: str, horizon: int = None) -> Optional[DataEntry]:
        """Get entry for a specific market/variety/grade."""
        key = f"{market}_{variety}_{grade}"
        return self._index.get(key)

    def list_markets(self) -> List[str]:
        return sorted({e.market for e in self.entries})

    def list_varieties(self, market: str) -> List[str]:
        return sorted({e.variety for e in self.entries if e.market == market})

    def list_grades(self, market: str, variety: str) -> List[str]:
        return sorted({e.grade for e in self.entries if e.market == market and e.variety == variety})

    def list_horizons(self, market: str, variety: str, grade: str) -> List[int]:
        """Return available horizons. Always [7, 15, 30] for NHITS."""
        entry = self.get_entry(market, variety, grade)
        return [7, 15, 30] if entry else []

    def get_all_entries(self) -> List[DataEntry]:
        return self.entries


if __name__ == "__main__":
    reg = ExperimentRegistry()
    print(f"Registered {len(reg.entries)} data entries.")
    for e in reg.entries:
        print(f"  {e.market} | {e.variety} | Grade {e.grade}")
        print(f"    Data: {e.data_path}")
        print(f"    NHITS: {e.nhits_model_path or 'NOT FOUND'}")

