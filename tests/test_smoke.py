"""Smoke test: the package and its modules import. Wires pytest before real tests exist."""

import importlib


def test_package_imports():
    assert importlib.import_module("miniserve").__version__


def test_modules_import():
    for name in ("model", "kv_cache", "scheduler", "engine", "server", "metrics", "bench"):
        importlib.import_module(f"miniserve.{name}")
