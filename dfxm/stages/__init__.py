"""Analysis stages.

Each stage module mirrors one family of the legacy standalone scripts and
exposes a ``run(params, progress=None) -> result`` entry point. ``progress``
defaults to a no-op so every stage stays runnable headless and unit-testable.
"""
