"""Gap report: which missing keywords show up most across all tracked applications.

Read-only over tracker.db. This is the deterministic baseline the Insights agent
has to beat: it counts raw keyword strings, which is exactly why almost everything
comes back 1x. Turning that noise into themes is the model's job, not this file's.
"""
from collections import Counter
from tracker import list_applications

def keyword_rows() -> list[tuple[int, str, str, list[str]]]:
    """(id, company, role, missing keywords) for every posting that logged any."""
    rows = []
    for app in list_applications():
        keywords = [k.strip() for k in (app["missing_keywords"] or "").split(",") if k.strip()]
        if keywords:
            rows.append((app["id"], app["company"], app["role"], keywords))
    return rows

def count_gaps() -> Counter:
    counts = Counter()
    for _, _, _, keywords in keyword_rows():
        # set(): one posting listing the same gap twice is still one posting, which is
        # the unit the Insights agent thresholds on
        counts.update(set(k.lower() for k in keywords))
    return counts

def show_gaps():
    counts = count_gaps()
    if not counts:
        print("No missing keywords logged yet.")
        return
    for keyword, count in counts.most_common(10):
        print(f"{count}x  {keyword}")

if __name__ == "__main__":
    show_gaps()
