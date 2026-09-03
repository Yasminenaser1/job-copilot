"""Job sources. Each module knows one API; the sweep lives in feeds.py.

Order matters: when the same job is cross-posted, the first source listed here
is the one whose link gets kept.
"""
from sources import arbeitnow, remoteok, remotive

ALL_SOURCES = [remoteok, remotive, arbeitnow]
