# Squad pipeline

```
                ┌─────────────┐
   hackathon    │  Strategist │  reads brief, scouts angle
   URL ───────► └──────┬──────┘
                       │
                       ▼ (user approval)
                ┌─────────────┐
                │     PM      │  scopes MVP, sets time budget
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │     Dev     │  delegates to claude-code subagent
                └──────┬──────┘
                       ▼
              ┌────────┴────────┐
              ▼                 ▼
        ┌──────────┐      ┌─────────────┐
        │ Designer │      │ Storyteller │  parallel spawn
        └────┬─────┘      └──────┬──────┘
             └────────┬──────────┘
                      ▼
                ┌─────────────┐
                │  Submitter  │  creates/updates DRAFT
                └──────┬──────┘
                       │
                       ▼ (user approval)
                  hackclaw_submit
                       │
                       ▼
                  ACTIVE on platform
```

## Approval gates

Two human-in-the-loop gates:

1. **After Strategist**: confirm the chosen angle. The Strategist's suggestion is high-leverage; getting it wrong wastes the whole run. A 30-second human check here is worth it.
2. **Before publish**: confirm the DRAFT submission. Publishing is irreversible on most platforms (you can't unsubmit), so we never auto-publish.

## Parallel phases

Designer and Storyteller run in parallel because they don't depend on each other. The Designer reads the chosen angle. The Storyteller reads the chosen angle plus the Dev outputs. Neither reads the other's output.

## Subagent return contract

Each subagent returns a JSON object as its final response. The orchestrator parses it and stores it in session memory under a known key. See each role's SKILL.md for its return shape.

## Time budgets

The orchestrator passes a time budget to PM. PM scopes the build to fit. Dev respects the budget and ships what it has when time runs out. Other roles run as fast as they run.

Default budget for a 24h hackathon: 360 minutes for the Dev phase, leaving buffer for the rest. Scale proportionally for shorter or longer hackathons.
