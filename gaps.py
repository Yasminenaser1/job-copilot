"""Gap report: which missing keywords show up most across all tracked applications."""
from collections import counter
from tracker import list_applications

def show_gaps():
    apps = list_applications()
    all_keywords = []
    for app in apps:
        keywords = app["missing_keywords"].split(", ")
        all_keywords.append(keywords)
    counts = Counter(all_keywords)
    for keyword, count in counts.most_common(10):
        print(f"{count}x  {keyword}")

show_gaps()