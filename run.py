#!/usr/bin/env python3
"""Convenience entry point.

``python run.py [options]`` is equivalent to ``python -m quantcam [options]``.
Run ``python run.py --help`` for the full list of options.
"""

from quantcam.cli import main

if __name__ == "__main__":
    main()
