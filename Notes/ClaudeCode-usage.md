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

