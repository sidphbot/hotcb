Read `.claude/plans/STREAMS.md` and help the user manage work streams.

**Behavior based on args:**

- **No args** (`/stream`): Show the roadmap table from STREAMS.md. Ask which stream to attach to, or offer to create a new one.

- **Stream ID** (`/stream fix-error-handling`): Attach to that stream. Read its section, show plan + last log entry, start working on next unchecked item.

- **`new`** (`/stream new`): Ask for: ID, type, priority, summary, goal. Add section + table row to STREAMS.md.

- **`branch <name>`** (`/stream branch feature-x`): Import an existing git branch as a stream.
  1. Run `git log --oneline main..<name>` (or `master..<name>`) to see commits
  2. Run `git diff --stat main..<name>` to see affected files
  3. Propose a stream ID, type, and summary based on the branch content
  4. Ask user to confirm or adjust
  5. Create the stream section in STREAMS.md with the branch noted, committed work as checked items, and uncommitted/remaining as unchecked
  6. If the branch has uncommitted changes (check `git status`), note those in the plan too

- **`status`** (`/stream status`): Show just the table with current statuses.

- **`done`** (`/stream done`): Mark current stream as `done`. Add dated log entry.

- **`release`** (`/stream release`): Set current stream to `planned` (pausing). Log what's left.

**When attaching to a stream:**
1. Set its status to `active` in the table row
2. Note the current git branch in the Branch column
3. Read the stream's plan section
4. Tell the user what's done and what's next
5. Start working on the first unchecked `- [ ]` item

**When finishing work (session ending or switching streams):**
1. Update checkboxes for completed items
2. Append a dated log line summarizing what was done and what's next
3. Update `STREAMS.md` — this is the only file that matters

**New stream section format:**
```
---

## <stream-id>
**Goal:** <one paragraph>
**Branch:** <branch-name> (if from `/stream branch`)
**Files:** <key files>
- [x] Already done thing (from git log)
- [ ] Remaining thing
**Log:** <date> — Created from branch <name>.
```
