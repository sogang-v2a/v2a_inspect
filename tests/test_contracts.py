from __future__ import annotations

import importlib
import unittest


class ContractImportTests(unittest.TestCase):
    def test_contract_modules_import_without_server_runtime(self) -> None:
        contracts = importlib.import_module("v2a_inspect.contracts")
        bundle = importlib.import_module("v2a_inspect.contracts.bundle")
        gold_set = importlib.import_module("v2a_inspect.contracts.gold_set")

        self.assertTrue(hasattr(contracts, "MultitrackDescriptionBundle"))
        self.assertTrue(hasattr(bundle, "PhysicalSourceTrack"))
        self.assertTrue(hasattr(gold_set, "load_gold_set_manifest"))


if __name__ == "__main__":
    unittest.main()
