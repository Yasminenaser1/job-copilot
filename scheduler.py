"""Run the scout on a schedule. Start it, leave it; Ctrl+C stops it. Nothing runs in the background."""
import time
from datetime import datetime
from scout_run import run_scout

INTERVAL_HOURS = 6

if __name__ == "__main__":
    print(f"⏰ Scheduler up: scout runs now, then every {INTERVAL_HOURS}h. Ctrl+C to stop.\n")
    while True:
        print(f"—— {datetime.now().strftime('%b %d %H:%M')} ——")
        try:
            run_scout()
        except Exception as e:
            print(f"⚠️  Run failed ({e}); will retry next cycle.")
        print(f"😴 Sleeping {INTERVAL_HOURS}h...\n")
        time.sleep(INTERVAL_HOURS * 3600)
