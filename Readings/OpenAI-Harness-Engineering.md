# OpenAI Harness Engineering
## AI Engineering Notes

## Takeaways
- The role of a software engineering team is shifting. Humans are no longer mainly writing code; they are designing systems, specifying intent, and building feedback loops that let agents work reliably.
- Humans steer. Agents execute.
- A strong AI engineering workflow depends on creating the right environment for the agent, not just giving it prompts.
- The best systems are built with clear scaffolding, reusable patterns, and good observability.
- As agent output grows, the bottleneck becomes human review and quality assurance.
- Good AI engineering requires strong tooling for logs, metrics, traces, and debugging so agents can inspect and improve their own work.
- Documentation should be designed for progressive disclosure. Instead of overwhelming the agent with everything at once, give it a small stable entry point and teach it where to look next.
- The goal is not perfect human-style code. The goal is code that is correct, maintainable, and understandable for future agent runs.
- In agent-first systems, correction is cheap and waiting is expensive. That means fast iteration and strong feedback loops matter a lot.
- Full autonomy can create drift because agents tend to copy existing patterns, even imperfect ones. This means engineering discipline is still necessary, but it is expressed through scaffolding and system design rather than manual code writing.

## Simple Understanding
- OpenAI’s idea is that AI engineering is less about writing every line manually and more about building an environment where agents can operate effectively.
- The real work is in creating tools, instructions, structure, and feedback loops that make agent behavior reliable.
- The system should help agents learn the repository gradually, inspect context intelligently, and produce useful outputs without constant human intervention.

## For an AI Engineer
- Focus on environment design: repository structure, instructions, templates, and workflows matter as much as the model itself.
- Use progressive disclosure in documentation and agent guidance so the agent starts with the essentials and expands its context as needed.
- Build observability into the system so agents can understand failures, inspect logs, and improve their behavior.
- Treat quality assurance as a first-class part of the system. Fast feedback loops reduce human overhead.
- Accept that generated code may not match human style exactly, but ensure it remains correct and maintainable.
- Keep the system coherent by defining architecture patterns, conventions, and review mechanisms that prevent drift over time.

