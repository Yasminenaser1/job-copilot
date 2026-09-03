# v3 Ideas - parked, not promised

Rules: ideas land here instead of into scope. Nothing on this list is a commitment.
If v3 ever gets signed, it gets its own contract with its own finish line.

## Parked from v2 development:
- "Rejected memory": Scout re-judges known-rejected postings every run (wasted model
  calls). Store rejections so they skip.
- Status/cache oddities in pipeline table noticed Sep 2 (blank statuses on some rows).

## Ideas:
-
- ~~More job feeds beyond RemoteOK~~ SHIPPED Sep 3 as v2.1: Remotive + Arbeitnow alongside
  RemoteOK (sources/, 14 model-free evals). Still parked: HN "Who's Hiring" - the monthly
  thread is comments, not a job API, so it needs its own parser rather than an adapter.
- Top Picks ranking now that the funnel is wide - noticed Sep 3: a fresh 79% match with a
  link ranked *below* three older link-less rows, one of them the "Course Director UX/UI
  and AI" posting the Scout itself rejects. Three slots, and stale rows are sitting in
  them. Needs a triage rule (recency? require a link? re-judge old rows against Scout?),
  not a bigger list.
- "Top Picks" daily digest view - the 3 best new matches surfaced with letters attached, instead of a table
- Smarter thresholds - only surface 70%+, auto-bury the rest
- ~~Ask: free-text questions over the tracker~~ SHIPPED Sep 2 (ask_agent.py, 12/12 evals).
  Known trade-off: the scope gate over-refuses advice-shaped questions ("which roles
  should I focus on this week?"). Refusing is the safe side of that line; revisit only
  if it gets annoying in real use.
- ~~AI insights agent~~ SHIPPED Sep 2 as skill-gap themes only (insights_agent.py, 7/7 evals).
  Still parked: outcome/funnel insights - blocked until rejection memory exists and
  enough applications are logged to have a funnel at all.
- Real-opening verification: detect ghost/fake/stale postings (age, repost patterns, company legitimacy signals) before they enter the pipeline - parked Sep 3, arrived suspiciously at application time :)
