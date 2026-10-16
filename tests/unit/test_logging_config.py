import logging

from demand_engine.logging_config import configure_logging


def test_configure_logging_sets_level_and_quiets_noisy_libs() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("lightgbm").level == logging.WARNING
    configure_logging("INFO")
    assert logging.getLogger().level == logging.INFO
