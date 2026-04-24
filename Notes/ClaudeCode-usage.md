### Claude Code: Core Features & Architecture
1. **CLAUDE.md (Project Memory)**:
A markdown file in your project root that stores high-level context. Claude reads this at the start of every session to understand project structure, tech stacks, and specific coding guidelines (e.g., "Use functional components" or "Maintain 100% test coverage").

2. **Permissions & Safety**:
Claude Code operates with a "human-in-the-loop" philosophy. It requires explicit approval before executing bash commands or writing to files. You can use `/permissions` to customize these rules, allowing for "auto-approve" on safe commands (like ls or npm test) while keeping strict blocks on destructive actions.

3. **Plan Mode**:
A specialized state toggled via Shift + Tab where Claude acts as a consultant rather than a coder. In this mode, Claude uses read-only tools to analyze the codebase and propose a multi-step strategy. This prevents "hallucinated" edits and saves tokens by ensuring you approve the logic before any files are modified.

4. **Checkpoints & Rewind**:
Claude automatically creates a "snapshot" of your filesystem before making edits. If a refactor breaks the build or goes in the wrong direction, the `/rewind` command allows you to view a timeline of changes and instantly restore your code to a previous working state.

5. **Skills**:
Predefined, reusable instruction sets for specific, complex workflows. Skills are stored as `skill.md` files and act as "runbooks" for the agent. When Claude recognizes a task that matches a skill, it invokes that specialized logic to ensure the task is handled according to your team's standard operating procedures.

6. **Hooks**:
Deterministic scripts that trigger automatically at specific points in the agentic loop. For example, a "post-save" hook can be set to automatically run a linter or formatter every time Claude finishes editing a file, ensuring code quality without manual intervention.

7. **Model Context Protocol (MCP)**:
An open standard that bridges the gap between the AI and external data. MCP allows Claude to securely connect to third-party tools (Google Drive, Slack, Figma, Jira) or local databases. It fetches only the necessary context on-demand rather than overwhelming the context window upfront.

8. **Plugins**:
Standardized packages that bundle together multiple skills, hooks, and MCP servers. Plugins allow for "Infrastructure as Code" for your AI assistant—one developer can configure a perfect workflow and share it with the entire team as a single installable unit.

9. **Context Management & Compact**:
While Claude has a large 200k+ token window, long sessions can eventually fill it. The `/context` command provides a breakdown of what is consuming memory. The `/compact` command allows Claude to summarize the session history, discarding the "noise" while retaining key decisions and architectural changes to free up space.

10. **Slash Commands**:
Terminal shortcuts designed to trigger common workflows quickly.

- `/init`: Scaffolds a CLAUDE.md for a new project.

- `/cost`: Displays the token spend for the current session.

- `/compact`: Triggers context compression.

- `/clear`: Resets the current conversation state.

11. **Sub-agents**:
For massive tasks, the main Claude session can spin up "child" sessions to handle specific sub-tasks in parallel. For example, the main agent might fix a feature while a sub-agent is tasked specifically with "Security Review" or "Documentation Update," keeping the primary context window lean and focused.

12. **Agentic Loop (The CLI)**:
The fundamental engine of Claude Code. Unlike a chat-based LLM, the CLI agent operates in a continuous loop: it observes the state, plans an action, executes a tool (terminal/file edit), analyzes the result, and iterates until the objective is reached.

____________

#### Run local Ollama model in Claude Code:

`ollama pull qwen2.5-coder`
`ollama launch claude`

(Then select a model)

#### Sessions in Claude Code:
 - One session = one task (saves context window)
 - Name your session immediately 
 - Use `/btw` for quick question (doesn't become part of main context)
 - Export session before a big refactor 

List sessions: `claude -r` 

Model selection: `/model`
- **Opus** for the
planning phase - where you're thinking through
architecture, writing specs and making decisions. 
- Switch
to **Sonnet** for the implementation phase where you just need reliable code generation.

#### Context window

- Context window = 200k 
Usable = 150k
- Auto-compaction triggers (75-92% full) - summarizes previously conversation 
- Repeated compaction causes corruption 

Best practices:
- Manual compaction: `/compact`
- Use subagent for isolated or exploratory work
- `/clear` or start new session
- Use `.claudeignore` to keep irrelevant files out

#### Subagents
- Subagents get their own isolated context window completely separate from the main session 
- Subagents return only a summary to the main context, not their full working history

#### CLAUDE.md files 
 - Within 200 words 
 - Split into `.claude/rules`
 - Use @imports to reference external docs 
 - Use subdirectory CLAUDE.md files
```
.claude/rules
    |--- code-style.md 
    |--- testing.md    
    |--- security.md 
    |--- api-conventions.md
```

```
API conventions 
See @docs/api-guidelines.md
```

#### Auto Memory
 - Only the first 200 lines of MEMORY.md load automatically 
 - To review or edit what Claude has saved, run `/memory`
 
 Path: `~/.claude/projects/<project>/memory/`

#### Skills

```
.claude/
    ---- skill
        ---- SKILL.md
        --- scripts/
        --- resources/
        --- assets/
```

Progressive Disclosure: 
- Level 1: Description (always visible)
- Level 2: SKILL.md body (loaded on demand)
- Level 3: Referenced resources (loaded if needed)

Claude won't call it (command)
`disable-model-invocation: true`

#### Example: Fix a Jira bug 
- Jira MCP tool - read Jira description 
- Skill `/debug` - loads debugging logic to analyze the codebase and reproduce the error locally 
- Filesystem MCP tool - write the fix to .tsx file 
- Skill `/review` - runs a final pass to ensure the fix follows the project's architectural patterns 

### Sub-agents

Sub-agents solve these problems:
 - Context window overflow 
 - "Lost in the middle" effect

Advantages:
 - context isolation 
 - specialization 
 - modularity 
 - parallelization 

Top use cases: 
 - Codebase exploration (Subagent isolated workspace does scanning, grepping and returns its findings to main - summary that 5 files need changes)
 - Independent code review (Main agent writes code, subagent does code review)
 - Verification & validation (Claude write code, Validation subagent checks input validation rules empty, format etc.)
 - Multi-stage pipelines (API Design -> implement -> test)
 - Parallel independent tasks  (Investigate outage across 3 services AuthService, UserService, OrderService)

