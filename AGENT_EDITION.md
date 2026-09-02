# Job Copilot: Agent Edition (v2) - Build Contract

Signed: Sep 2, 2026. Target: ~3 weeks.

## DONE means:
1. Scout agent checks permitted job feeds (RSS/APIs) for AI/automation roles on a schedule
2. Finds are auto-scored through the existing matcher and logged with status "scouted"
3. Letter agent pre-drafts cover letters for postings scoring 65+
4. "Scouted" feed view in the existing UI
5. GUARDRAIL: the system never sends, submits, or touches any account. It prepares; the human decides.
6. 3 eval cases for Scout relevance filtering
7. README updated with agent architecture diagram

## NOT in scope (v3-ideas.md material, refused on sight):
LinkedIn anything, auto-applying, email parsing, browser extensions, notifications.

## Build order:
- Week 1: feed ingestion + auto-scoring (plumbing, no agents)
- Week 2: CrewAI agent layer (Scout + Letter)
- Week 3: UI feed, scheduling, evals, README -> DONE, then STOP.
