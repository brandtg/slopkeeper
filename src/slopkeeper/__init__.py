# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""slopkeeper — inventory, partition and scaffold messy file corpora.

Agent-first CLI: the calling agent is the brain; this tool is the hands
and the seatbelt. No LLM calls, no network, stdlib only.
"""

# pyproject.toml is the single source of truth for the version; it is
# read from installed package metadata at runtime. Never hardcode it.
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("slopkeeper")
except PackageNotFoundError:  # running from an uninstalled checkout
    __version__ = "unknown"

VERSION = __version__
