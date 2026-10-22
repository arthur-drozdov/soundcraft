# Playbooks

Skills and templates that turn Claude Code into a studio assistant for SoundCraft. They follow section 9 of [the spec](../../docs/agent-studio/SPEC.md) and section 7 of [DESIGN.md](../DESIGN.md).

| Path | What |
|---|---|
| `skills/<name>/SKILL.md` | Step-by-step workflows (YAML front matter `name` and `description`) |
| `templates/CLAUDE.md` | Project instructions for the music agent (placeholders `{{PROJECT}}`, `{{SOUNDCRAFT_PORT}}`, `{{FORGE_URL}}`) |
| `templates/mcp.json` | MCP servers `soundcraft`, `forge` and `atlas` (becomes the project's `.mcp.json`) |
| `new_project.py` | Creates a music project folder |

| Skill | Use it for | Spec |
|---|---|---|
| `studio-verified-edit` | Every change: inspect, checkpoint, batch, re-inspect, bounce, report | 9.1 |
| `studio-sample-composition` | Generate, audition, import, chop, resequence, process, preview | 9.2 |
| `studio-playable-instrument` | A single sample on the Sampler, or a Forge keybed through SFZ | 9.3 |
| `studio-sing-to-midi` | Hummed melody to MIDI to an instrument | 9.4 |
| `studio-section-ops` | Duplicate or move sections, bigger choruses, transitions | 9.5 |
| `studio-plugin-profiling` | Atlas profiles before touching a plugin; profiling; manual ingestion | 8 |
| `studio-session-conventions` | Track roles and colours, markers, tags, limiter and the -1 dBFS cap | 9.6 |

## Create a project

```sh
python playbooks/new_project.py ~/Music/my-song --studio-port 7801 \
    --forge-url http://spark.tailnet:7700 --soundcraft-cli /path/to/soundcraft-cli
```

This creates:

```
my-song/
  CLAUDE.md             rendered from templates/CLAUDE.md
  .mcp.json             rendered from templates/mcp.json
  .claude/skills/       copies of the skills
  Forge Inbox/          forge-mcp downloads (dated sub-folders, sidecar .json next to each file)
  Previews/             bounces
  Checkpoints/          session.checkpoint output
  Template.scraft       only with --soundcraft-cli
```

Options:

- `--studio-port N`: the SoundCraft control port. Without it, `.mcp.json` reads `${SOUNDCRAFT_PORT:-7801}` from the environment when Claude Code starts.
- `--forge-url URL`: the Forge base URL (no credentials in it). Without it, `${FORGE_URL:-http://127.0.0.1:7700}`.
- `--soundcraft-cli PATH`: a `soundcraft-cli` binary (or a name on `PATH`); builds `Template.scraft` headlessly.
- `--force`: write into a folder that is not empty. Without it a non-empty folder is refused. `--force` overwrites only the files this script creates (`CLAUDE.md`, `.mcp.json`, the skills, `Template.scraft`) and deletes nothing.

The Forge token is never written to disk: `.mcp.json` holds `"FORGE_TOKEN": "${FORGE_TOKEN}"`, so export `FORGE_TOKEN` in the shell that starts Claude Code. `AGENT_STUDIO_PROJECT` is set to the project folder, and `ATLAS_PROFILES` to this checkout's `atlas/profiles` when that folder exists.

## Template session

With `--soundcraft-cli`, the script runs `soundcraft-cli script ... --keep-going --save` headlessly and checks that the saved session opens. `Template.scraft` contains:

- audio tracks `DRM`, `BAS`, `SYN`, `FX` (stereo) and `VOX` (mono), coloured by role. Instrument tracks are added when a song needs MIDI;
- section markers Intro (bar 1), Verse (5), Chorus (13), Drop (21) and Outro (29);
- a `Master` track with a `maximizer` limiter in insert slot 0 (threshold 0 dB, ceiling -1.2 dB).

A step that the installed `soundcraft-cli` rejects (for example a command it does not have) is left out and listed in the output. The agent works on a copy of the template (`session.save_as`), never on the template itself.

## Run it

1. Put `soundcraft-cli`, `forge-mcp` and `atlas-mcp` on `PATH` (activate the virtual environment that has `agent-studio` installed).
2. Start SoundCraft with its control channel: `soundcraft --control 7801 ~/Music/my-song/Template.scraft`. Open the window and keep it visible: you watch every edit happen.
3. `export FORGE_TOKEN=...`, then start Claude Code in the project folder. It reads `CLAUDE.md`, `.mcp.json` and `.claude/skills/`.

Headless instead of the window: replace the `soundcraft` server's arguments with `["mcp"]` (add `--automation-read-root DIR --automation-write-root DIR` to confine file access to the project folder). `--connect` and the roots cannot be combined.

## Tests

`tests/test_playbooks.py` checks the front matter of every skill, that every command id the playbooks mention exists (in `crates/engine/src/cmd/*.rs`, `crates/ui-egui/src/menus.rs` or DESIGN.md section 9), that the MCP template renders to valid JSON, and that `new_project.py` builds the tree and refuses a non-empty folder. With `SOUNDCRAFT_CLI=/path/to/soundcraft-cli` it also builds the template session and opens it headlessly.
