"""Prompt builders for the model families. Pure functions, no third-party imports.

Foundation (RoyalCities) models are trained on comma-separated tag prompts. The grammar and the
vocabulary below were read from the RC Stable Audio Tools sources (MIT licensed;
``stable_audio_tools/interface/prompts/{foundation,keybed,oneshot,prompt_common}_prompts.py`` and
``interface/gradio.py::amend_prompt*``, commit 43dcbb4) and are re-implemented here without the Gradio
code. The shapes are:

* loop     ``<family>, <subfamily>, <descriptors>, <behaviour>, Wet|Dry, <fx>, <key>, <N> Bars, <N> BPM``
* one shot ``One Shot, <family>, <subfamily>, <descriptors>, Wet|Dry, <fx>, <Register>, <Note>``
* keybed   ``Keybed, Timbre Profile, <family>, <subfamily>, <descriptors>, Wet|Dry, <fx>, Target Note, <Note>``

Tags arrive from the job as ``tags: dict[str, str | list[str]]`` (DESIGN 3.4). Recognised keys (case
and ``-``/``_`` insensitive; values may be lists or comma separated strings):

=============== =====================================================================================
``family``       one instrument family (``Synth``, ``Keys``, ...)
``subfamily``    one subfamily/instrument (``Synth Lead``, ``Rhodes Piano``; alias ``instrument``)
``descriptors``  timbre/spatial/wave tags (``warm``, ``wide``, ``saw``); aliases ``descriptor``, ``timbre``,
                 ``descriptors_extra`` and every key starting with ``descriptor``
``behaviour``    melodic behaviour (``arp``, ``rising``, ``slow speed``); aliases ``behavior``,
                 ``behaviour_tags``, ``behavior_tags``, ``melody``, ``structure``
``effects``      effect tokens (``Medium Reverb``, ``Ping Pong Delay``); aliases ``effect``, ``fx``
``wetdry``       ``Wet`` or ``Dry`` (alias ``wet``); default: Wet when effects are given, else Dry
``key``/``bpm``/``bars``  optional overrides for the numbers that otherwise come from the job params
=============== =====================================================================================

Any other key (except test hooks starting with ``x-``) contributes its values as extra descriptors,
so an agent can pass vocabulary the table above does not name. Tokens that match the vocabulary
are rewritten to its canonical case; unknown tokens are kept as written and reported by
:func:`unknown_tokens` so a caller can warn.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from forge.music import NOTE_NAMES, key_name, parse_key

MAX_TOKEN_CHARS = 80
MAX_TOKENS = 64

# ---- vocabulary (canonical spellings) -----------------------------------------------------------

FAMILIES: tuple[str, ...] = (
    "Synth", "Keys", "Bass", "Bowed Strings", "Mallet", "Wind", "Guitar", "Brass", "Vocal", "Plucked Strings",
    "Pure Tone", "White Noise",
)
SUBFAMILIES: tuple[str, ...] = (
    "Synth Lead", "Pluck", "Pad", "Supersaw", "FM Synth", "Wavetable Synth", "Atmosphere", "Texture", "Synth Bass",
    "Ensemble", "Bell", "Flute", "Digital Organ", "Digital Strings",
    "Grand Piano", "Digital Piano", "Rhodes Piano", "Felt Piano", "Wurlitzer Piano", "Tack Piano", "Clavinet",
    "Hammond Organ", "Church Organ", "Pipe Organ", "Harpsichord", "Celesta", "Tubular Bells", "Harp", "Harmonium",
    "Wavetable Bass", "Reese Bass", "Sub Bass", "Electric Bass", "Analog Bass", "FM Bass", "Picked Bass",
    "Digital Bass", "Fingered Bass", "808", "FX",
    "Violin", "Cello", "Viola", "Fiddle",
    "Marimba", "Vibraphone", "Glockenspiel", "Kalimba", "Xylophone", "Steel Drums", "Music Box", "Toy Bell",
    "Pan Flute", "Piccolo", "Clarinet", "Oboe", "Bassoon", "Ocarina", "World Winds", "Woodwinds", "Saxophones",
    "Sax", "Alto Sax", "Tenor Sax", "Soprano Sax", "Irish Flute",
    "Electric Guitar", "Acoustic Guitar", "Nylon Guitar", "Ukulele", "Lute", "Banjo",
    "Trumpet", "Brass", "French Horn", "Tuba", "Tenor Trombone", "Bass Trombone", "Trombone", "War Horn",
    "Choir", "Synthetic Choir", "Synthetic",
    "Concert Harp", "Celtic Harp", "Koto", "Sitar",
    "Sine", "Saw", "Triangle", "Pulse",
)
# Loop prompts use lower-case tags, keybed prompts Title Case; matching is case-insensitive and
# the canonical spelling depends on the prompt kind (see ``_canon``).
LOOP_DESCRIPTORS: tuple[str, ...] = (
    "sub", "sub bass", "bass", "low mids", "mids", "upper mids", "highs", "air",
    "wide", "mono", "near", "far", "spacey", "ambient", "distant", "intimate", "small", "big", "deep",
    "saw", "square", "sine", "triangle", "pulse", "analog", "digital", "fm", "supersaw", "reese", "pitch bend",
    "white noise", "filter",
    "dubstep", "chiptune", "acid", "303", "retro", "vintage", "laser", "siren", "fx", "formant vocal", "growl",
    "warm", "bright", "tight", "thick", "airy", "rich", "clean", "gritty", "crisp", "focused", "metallic", "dark",
    "shiny", "present", "silky", "sparkly", "smooth", "cold", "buzzy", "round", "fat", "punchy", "thin", "soft",
    "woody", "hollow", "nasal", "biting", "overdriven", "subdued", "breathy", "glassy", "pizzicato", "staccato",
    "snappy", "full", "harsh", "knock", "muddy", "steel", "veiled", "rubbery", "rumble", "noisy", "boomy",
    "crispy", "dreamy", "heavy", "tiny", "spiccato",
)
KEYBED_DESCRIPTORS: tuple[str, ...] = (
    "Warm", "Bright", "Dark", "Airy", "Rich", "Clean", "Gritty", "Crisp", "Focused", "Metallic", "Smooth", "Cold",
    "Buzzy", "Round", "Fat", "Punchy", "Thin", "Soft", "Woody", "Hollow", "Nasal", "Biting", "Overdriven",
    "Subdued", "Breathy", "Glassy", "Sparkly", "Shiny", "Noisy", "Muffled", "Distant", "Wide", "Mono", "Near",
    "Far", "Spacey", "Ambient", "Intimate", "Small", "Big", "Deep", "Rumble", "Growl", "Wavetable", "Digital",
    "Analog", "Retro", "Vintage", "Dubstep", "Chiptune", "Formant Vocal", "Synthetic Vox", "Choir", "Pluck",
    "Sustained", "Short", "Staccato", "Snappy", "Pizzicato", "Spiccato", "Impact", "Hit", "Swell", "Thick",
    "Present", "Sharp", "Harsh", "Bell", "Full", "Silky", "Square", "Pulse", "Saw", "Sine", "Triangle",
    "White Noise", "Pure Tone", "FM", "Supersaw", "Reese", "Filter",
)
BEHAVIOURS: tuple[str, ...] = (
    "slow speed", "medium speed", "fast speed",
    "off beat", "alternating", "triplets", "strummed", "arp",
    "rising", "falling", "bounce", "rolling", "sustained", "choppy", "top",
    "simple", "repeating", "catchy", "complex", "epic",
    "chord progression", "dance chord progression", "melody", "bassline",
    "16th note", "quarter note",
)
EFFECTS: tuple[str, ...] = (
    "Low Reverb", "Medium Reverb", "High Reverb", "Plate Reverb",
    "Low Delay", "Medium Delay", "Ping Pong Delay", "Stereo Delay", "Cross Delay", "Delay", "High Delay", "Mono Delay",
    "Low Distortion", "Medium Distortion", "High Distortion", "Distortion",
    "Phaser", "Low Phaser", "Medium Phaser", "High Phaser",
    "Bitcrush", "High Bitcrush",
)
FX_WORDS = ("reverb", "delay", "distortion", "phaser", "bitcrush")

# One-shot register bands (lowest MIDI note of the band, label), from oneshot_prompts.ONESHOT_REGISTER_BANDS.
REGISTER_BANDS: tuple[tuple[int, str], ...] = (
    (84, "Top Register"), (72, "High Register"), (48, "Medium Register"), (36, "Low Register"), (0, "Sub Register"),
)
KEYBED_MIN_MIDI = 12  # C0
KEYBED_MAX_MIDI = 108  # C8
ONESHOT_MIN_MIDI = 12  # C0
ONESHOT_MAX_MIDI = 104  # G#7 (ONESHOT_REGISTER_BANDS "Top Register" ends at G#7)

def _table(*groups: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """casefolded word -> every spelling the vocabulary has for it (usually one)."""
    out: dict[str, list[str]] = {}
    for group in groups:
        for word in group:
            spellings = out.setdefault(word.casefold(), [])
            if word not in spellings:
                spellings.append(word)
    return {k: tuple(v) for k, v in out.items()}


_SOURCES = _table(FAMILIES, SUBFAMILIES)
_LOOP_DESC = _table(LOOP_DESCRIPTORS)
_KEYBED_DESC = _table(KEYBED_DESCRIPTORS)
_BEHAVIOUR = _table(BEHAVIOURS)
_EFFECT = _table(EFFECTS)
_ANY_LOOP = _table(FAMILIES, SUBFAMILIES, LOOP_DESCRIPTORS, BEHAVIOURS, EFFECTS)
_ANY_KEYBED = _table(FAMILIES, SUBFAMILIES, KEYBED_DESCRIPTORS, EFFECTS)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_BPM_TOKEN = re.compile(r"^\d+(\.\d+)?\s*bpm$", re.IGNORECASE)
_BARS_TOKEN = re.compile(r"^\d+\s*bars?$", re.IGNORECASE)
_KEY_TOKEN = re.compile(r"^[A-Ga-g][#b]?\s*(major|minor|maj|min)$", re.IGNORECASE)
_NOTE_TOKEN = re.compile(r"^[A-Ga-g][#b]?-?\d+$")
_WETDRY = {"wet", "dry"}
_RESERVED = {"one shot", "oneshot", "one-shot", "keybed", "timbre profile", "target note", "sequence", "chromatic chunk",
             "note sequence"} | {label.casefold() for _, label in REGISTER_BANDS}

_KEY_ALIASES = {
    "family": "family", "subfamily": "subfamily", "sub_family": "subfamily", "instrument": "subfamily",
    "descriptors": "descriptors", "descriptor": "descriptors", "timbre": "descriptors",
    "behaviour": "behaviour", "behavior": "behaviour", "behaviour_tags": "behaviour", "behavior_tags": "behaviour",
    "melody": "behaviour", "structure": "behaviour",
    "effects": "effects", "effect": "effects", "fx": "effects",
    "wetdry": "wetdry", "wet": "wetdry", "wet_dry": "wetdry",
    "key": "key", "bpm": "bpm", "bars": "bars",
}


# ---- spec ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FoundationSpec:
    """The structured content of a Foundation prompt (everything except the numbers a job carries)."""

    family: str = ""
    subfamily: str = ""
    descriptors: tuple[str, ...] = ()
    behaviours: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    wet: bool | None = None  # None: Wet when effects are present, else Dry
    key: str | None = None
    bpm: float | None = None
    bars: int | None = None
    extra: tuple[str, ...] = field(default=())  # free text tokens (the job's ``prompt``)

    @property
    def is_wet(self) -> bool:
        return bool(self.effects) if self.wet is None else self.wet

    def sources(self) -> tuple[str, ...]:
        return tuple(t for t in (self.family, self.subfamily) if t)


def clean_token(text: Any) -> str:
    """One prompt token: control characters removed, whitespace collapsed, commas dropped, length-capped."""
    if not isinstance(text, (str, int, float)) or isinstance(text, bool):
        return ""
    cleaned = _CONTROL.sub(" ", str(text)).replace(",", " ")
    return re.sub(r"\s+", " ", cleaned).strip()[:MAX_TOKEN_CHARS]


def split_tokens(value: Any) -> list[str]:
    """Tokens of a tag value: a string is split on commas, a list is flattened."""
    if isinstance(value, str):
        parts: Iterable[Any] = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = (piece for item in value for piece in (str(item).split(",") if isinstance(item, str) else [item]))
    else:
        parts = [value]
    return [t for t in (clean_token(p) for p in parts) if t]


def dedupe(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        key = token.casefold()
        if token and key not in seen:
            seen.add(key)
            out.append(token)
    return out


def _normal_key(key: str) -> str:
    return re.sub(r"[\s\-]+", "_", key.strip().lower())


def _parse_wet(value: Any) -> bool | None:
    first = split_tokens(value)
    if not first:
        return None
    word = first[0].casefold()
    if word in {"wet", "fx", "with fx", "true", "yes", "1", "effects"}:
        return True
    if word in {"dry", "no fx", "none", "clean", "false", "no", "0"}:
        return False
    return None


def _strip_authoritative(tokens: Iterable[str]) -> list[str]:
    """Drop tokens the builders set themselves (Wet/Dry, ``N Bars``, ``N BPM``, a key, grammar words)."""
    out = []
    for token in tokens:
        low = token.casefold()
        if low in _WETDRY or low in _RESERVED or _BPM_TOKEN.match(token) or _BARS_TOKEN.match(token):
            continue
        if _KEY_TOKEN.match(token) or _NOTE_TOKEN.match(token):
            continue
        out.append(token)
    return out


def spec_from_tags(tags: Mapping[str, Any] | None, prompt: str = "") -> FoundationSpec:
    """Build a :class:`FoundationSpec` from a job's ``tags`` and free-text ``prompt``."""
    groups: dict[str, list[str]] = {k: [] for k in ("family", "subfamily", "descriptors", "behaviour", "effects")}
    extra_descriptors: list[str] = []
    wet: bool | None = None
    numbers: dict[str, Any] = {}
    for raw_key, value in (tags or {}).items():
        norm = _normal_key(str(raw_key))
        if norm.startswith("x_") or norm.startswith("x-"):
            continue  # test hooks, not vocabulary
        kind = _KEY_ALIASES.get(norm)
        if kind is None and norm.startswith("descriptor"):
            kind = "descriptors"
        if kind == "wetdry":
            wet = _parse_wet(value)
        elif kind in ("key", "bpm", "bars"):
            numbers[kind] = split_tokens(value)[:1]
        elif kind in groups:
            groups[kind].extend(split_tokens(value))
        else:
            extra_descriptors.extend(split_tokens(value))
    key = numbers.get("key", [None])[0] if numbers.get("key") else None
    bpm = _to_float(numbers["bpm"][0]) if numbers.get("bpm") else None
    bars = _to_float(numbers["bars"][0]) if numbers.get("bars") else None
    prompt_tokens = _strip_authoritative(split_tokens(prompt))
    return FoundationSpec(
        family=(groups["family"] or [""])[0],
        subfamily=(groups["subfamily"] or [""])[0],
        descriptors=tuple(dedupe([*groups["descriptors"], *extra_descriptors])[:MAX_TOKENS]),
        behaviours=tuple(dedupe(groups["behaviour"])[:MAX_TOKENS]),
        effects=tuple(dedupe(groups["effects"])[:16]),
        wet=wet,
        key=key,
        bpm=bpm if bpm and math.isfinite(bpm) and bpm > 0 else None,
        bars=int(bars) if bars and math.isfinite(bars) and 1 <= bars <= 64 else None,
        extra=tuple(dedupe(prompt_tokens)[:MAX_TOKENS]),
    )


def _to_float(text: Any) -> float | None:
    try:
        value = float(str(text).strip().lower().removesuffix("bpm").strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


# ---- numbers ------------------------------------------------------------------------------------

def format_bpm(bpm: float) -> str:
    """``128`` for whole tempos, one decimal otherwise (the models were trained on whole BPM values)."""
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("bpm must be positive and finite")
    return str(int(round(bpm))) if abs(bpm - round(bpm)) < 0.05 else f"{bpm:.1f}"


def format_key(key: str | None) -> str | None:
    """``"F minor"``; flats are folded to sharps because the model's key vocabulary is sharp-only."""
    if not key:
        return None
    parsed = parse_key(key)
    if parsed is None:
        raise ValueError(f"unrecognised key {key!r} (expected e.g. 'F minor')")
    return key_name(*parsed)


def midi_to_note(midi: int) -> str:
    """``60`` -> ``"C4"`` (C4 = 60, sharps only)."""
    if isinstance(midi, bool) or not isinstance(midi, int) or not 0 <= midi <= 127:
        raise ValueError("MIDI note must be an integer in 0..127")
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def register_label(midi: int) -> str:
    for low, label in REGISTER_BANDS:
        if midi >= low:
            return label
    return REGISTER_BANDS[-1][1]  # pragma: no cover - the last band starts at 0


# ---- Foundation builders ------------------------------------------------------------------------

def _canon(token: str, table: Mapping[str, tuple[str, ...]]) -> str:
    """The vocabulary's spelling of ``token``; unchanged when unknown or when the vocabulary has several
    spellings and ``token`` is none of them (``bass`` the band tag versus ``Bass`` the family)."""
    spellings = table.get(token.casefold())
    if not spellings or token in spellings:
        return token
    return spellings[0] if len(spellings) == 1 else token


def _body(spec: FoundationSpec, *, keybed: bool, behaviours: bool) -> list[str]:
    """Family, subfamily, descriptors (and behaviours) in prompt order, canonically spelled, minus effect
    tokens (they are emitted after Wet/Dry) and minus anything the builders set themselves."""
    desc, free = (_KEYBED_DESC, _ANY_KEYBED) if keybed else (_LOOP_DESC, _ANY_LOOP)
    effect_names = {e.casefold() for e in spec.effects}
    tokens = [_canon(t, _SOURCES) for t in spec.sources()]
    tokens += [_canon(t, desc) for t in spec.descriptors]
    if behaviours:
        tokens += [_canon(t, _BEHAVIOUR) for t in spec.behaviours]
    tokens += [_canon(t, free) for t in spec.extra]
    return dedupe(_strip_authoritative(t for t in tokens if t.casefold() not in effect_names))


def _effects(spec: FoundationSpec) -> list[str]:
    return dedupe(_canon(t, _EFFECT) for t in spec.effects) if spec.is_wet else []


def build_loop_prompt(spec: FoundationSpec, *, bars: int, bpm: float, key: str | None = None) -> str:
    """``<family>, <subfamily>, <descriptors>, <behaviour>, Wet|Dry, <fx>, <key>, <N> Bars, <N> BPM``."""
    if isinstance(bars, bool) or not isinstance(bars, int) or bars < 1:
        raise ValueError("bars must be a positive integer")
    tokens = _body(spec, keybed=False, behaviours=True)
    tokens.append("Wet" if spec.is_wet else "Dry")
    tokens += _effects(spec)
    chosen_key = format_key(key or spec.key)
    if chosen_key:
        tokens.append(chosen_key)
    tokens += [f"{bars} Bars", f"{format_bpm(bpm)} BPM"]
    return ", ".join(dedupe(tokens))


def build_descriptor_prompt(spec: FoundationSpec) -> str:
    """The body of a loop prompt without key, bars or BPM (used for audio-to-audio, which has no tempo)."""
    tokens = _body(spec, keybed=False, behaviours=True)
    tokens.append("Wet" if spec.is_wet else "Dry")
    tokens += _effects(spec)
    if spec.key:
        tokens.append(format_key(spec.key) or "")
    if spec.bars:
        tokens.append(f"{spec.bars} Bars")
    if spec.bpm:
        tokens.append(f"{format_bpm(spec.bpm)} BPM")
    return ", ".join(dedupe(tokens))


def build_oneshot_prompt(spec: FoundationSpec, *, midi_note: int) -> str:
    """``One Shot, <family>, <subfamily>, <descriptors>, Wet|Dry, <fx>, <Register>, <Note>``."""
    if not ONESHOT_MIN_MIDI <= midi_note <= ONESHOT_MAX_MIDI:
        raise ValueError(
            f"MIDI note {midi_note} is outside the one-shot range {ONESHOT_MIN_MIDI}..{ONESHOT_MAX_MIDI} (C0..G#7)"
        )
    tokens = ["One Shot", *_body(spec, keybed=False, behaviours=False)]
    tokens.append("Wet" if spec.is_wet else "Dry")
    tokens += _effects(spec)
    tokens += [register_label(midi_note), midi_to_note(midi_note)]
    return ", ".join(dedupe(tokens))


def build_keybed_prompt(spec: FoundationSpec, *, midi_note: int) -> str:
    """``Keybed, Timbre Profile, <family>, <subfamily>, <descriptors>, Wet|Dry, <fx>, Target Note, <Note>``."""
    if not KEYBED_MIN_MIDI <= midi_note <= KEYBED_MAX_MIDI:
        raise ValueError(
            f"MIDI note {midi_note} is outside the keybed range {KEYBED_MIN_MIDI}..{KEYBED_MAX_MIDI} (C0..C8)"
        )
    tokens = ["Keybed", "Timbre Profile", *_body(spec, keybed=True, behaviours=False)]
    tokens.append("Wet" if spec.is_wet else "Dry")
    tokens += _effects(spec)
    tokens += ["Target Note", midi_to_note(midi_note)]
    return ", ".join(dedupe(tokens))


def unknown_tokens(spec: FoundationSpec, *, keybed: bool = False) -> list[str]:
    """Tokens of ``spec`` that are not in the Foundation vocabulary (kept in the prompt as written)."""
    desc, free = (_KEYBED_DESC, _ANY_KEYBED) if keybed else (_LOOP_DESC, _ANY_LOOP)
    checks: list[tuple[str, Mapping[str, tuple[str, ...]]]] = [(t, _SOURCES) for t in spec.sources()]
    checks += [(t, desc) for t in spec.descriptors]
    checks += [(t, free) for t in spec.extra]
    checks += [(t, _EFFECT) for t in spec.effects]
    if not keybed:
        checks += [(t, _BEHAVIOUR) for t in spec.behaviours]
    return dedupe(t for t, table in checks if t.casefold() not in table and not _KEY_TOKEN.match(t))


# ---- Stable Audio 3 -----------------------------------------------------------------------------

_SA3_KEYS = {"genre": "Genre", "instruments": "Instruments", "instrument": "Instruments", "format": "Format",
             "tracktype": "TrackType", "track_type": "TrackType", "vocaltype": "VocalType", "mood": "Mood"}


def sa3_tag_tokens(tags: Mapping[str, Any] | None) -> list[str]:
    """AudioSparx-style ``Key: value`` tokens from the tags Stable Audio 3 was trained with; other keys
    contribute their values as plain words. Test hooks (``x-...``) are ignored."""
    tokens: list[str] = []
    for raw_key, value in (tags or {}).items():
        norm = _normal_key(str(raw_key))
        if norm.startswith("x_"):
            continue
        label = _SA3_KEYS.get(norm.replace("_", ""), _SA3_KEYS.get(norm))
        for item in split_tokens(value):
            tokens.append(f"{label}: {item}" if label else item)
    return dedupe(tokens)


def _sa3(*parts: Iterable[str]) -> str:
    return ", ".join(dedupe(t for part in parts for t in part))


def sa3_sfx_prompt(prompt: str, tags: Mapping[str, Any] | None = None) -> str:
    """``TrackType: SFX, <prompt>, <tags>`` (the model card's recommended prefix for effects)."""
    return _sa3(["TrackType: SFX"], [clean_prompt(prompt)], sa3_tag_tokens(tags))


def sa3_loop_prompt(prompt: str, tags: Mapping[str, Any] | None, *, bpm: float, key: str | None, bars: int) -> str:
    """``TrackType: Music, VocalType: Instrumental, <prompt>, <tags>, <key>, <N> BPM, <N> bars``."""
    key_text = format_key(key) if key else None
    return _sa3(
        ["TrackType: Music", "VocalType: Instrumental"], [clean_prompt(prompt)], sa3_tag_tokens(tags),
        [f"Key: {key_text}"] if key_text else [], [f"{format_bpm(bpm)} BPM", f"{bars} bars"],
    )


def sa3_sample_prompt(prompt: str, tags: Mapping[str, Any] | None, *, midi_note: int, sfx_model: bool) -> str:
    """A pitched one-shot: ``TrackType: Instrument|SFX, <prompt>, <tags>, single note <Note>``."""
    track = "TrackType: SFX" if sfx_model else "TrackType: Instrument"
    return _sa3([track], [clean_prompt(prompt)], sa3_tag_tokens(tags), [f"single note {midi_to_note(midi_note)}"])


def sa3_edit_prompt(prompt: str, tags: Mapping[str, Any] | None = None) -> str:
    """Free text for audio-to-audio, inpainting and continuation (may be empty: the audio carries the context)."""
    return _sa3([clean_prompt(prompt)], sa3_tag_tokens(tags))


def clean_prompt(text: str) -> str:
    """Free text with control characters removed and whitespace collapsed (commas are kept)."""
    return re.sub(r"\s+", " ", _CONTROL.sub(" ", str(text or ""))).strip()[:2000]


# ---- ACE-Step -----------------------------------------------------------------------------------

ACE_TRACKS: tuple[str, ...] = (
    "woodwinds", "brass", "fx", "synth", "strings", "percussion", "keyboard", "guitar", "bass", "drums",
    "backing_vocals", "vocals",
)
_ACE_ALIASES: dict[str, str] = {
    "vocal": "vocals", "voice": "vocals", "vox": "vocals", "singing": "vocals", "backing vocals": "backing_vocals",
    "backing-vocals": "backing_vocals", "harmony": "backing_vocals", "drum": "drums", "kit": "drums",
    "piano": "keyboard", "keys": "keyboard", "organ": "keyboard", "rhodes": "keyboard", "keyboards": "keyboard",
    "strings": "strings", "string": "strings", "violin": "strings", "cello": "strings", "orchestra": "strings",
    "pad": "synth", "lead": "synth", "arp": "synth", "synthesizer": "synth", "synthesiser": "synth",
    "guitars": "guitar", "bass guitar": "bass", "sub": "bass", "808": "bass", "horns": "brass", "trumpet": "brass",
    "woodwind": "woodwinds", "flute": "woodwinds", "sax": "woodwinds", "clarinet": "woodwinds", "oboe": "woodwinds",
    "perc": "percussion", "shaker": "percussion", "conga": "percussion", "tambourine": "percussion",
    "riser": "fx", "effects": "fx", "sfx": "fx", "texture": "fx", "noise": "fx", "impact": "fx",
}


def ace_track(name: str) -> str:
    """A valid ACE-Step track name for ``name`` (aliases accepted), else ``ValueError``."""
    key = re.sub(r"\s+", " ", str(name).strip().lower())
    if key in ACE_TRACKS:
        return key
    mapped = _ACE_ALIASES.get(key) or _ACE_ALIASES.get(key.replace("_", " "))
    if mapped:
        return mapped
    raise ValueError(f"unknown ACE-Step track {name!r}; expected one of {', '.join(ACE_TRACKS)}")


def ace_track_from_description(description: str, tags: Mapping[str, Any] | None = None) -> str | None:
    """The ACE-Step track a layer description asks for: ``tags.track``/``tags.stem`` win, else the first
    recognised word of ``description`` (``"warm analog pad"`` -> ``synth``), else ``None``."""
    for key in ("track", "stem", "layer"):
        for raw_key, value in (tags or {}).items():
            if _normal_key(str(raw_key)) == key:
                for token in split_tokens(value):
                    return ace_track(token)
    words = re.findall(r"[a-z0-9]+", description.lower())
    candidates = [*(f"{a} {b}" for a, b in zip(words, words[1:])), *words]  # two-word names first
    for candidate in candidates:
        if candidate in ACE_TRACKS:
            return candidate
        if candidate in _ACE_ALIASES:
            return _ACE_ALIASES[candidate]
    return None


def ace_key(key: str | None) -> str:
    """``"F minor"`` -> ``"F minor"`` for ACE-Step's ``keyscale`` ('' when unset)."""
    return format_key(key) or ""


def ace_caption(prompt: str, tags: Mapping[str, Any] | None = None, extra: Sequence[str] = ()) -> str:
    """Caption text for ACE-Step (its limit is 512 characters)."""
    tokens = dedupe([clean_prompt(prompt), *extra, *sa3_tag_tokens(tags)])
    return ", ".join(t for t in tokens if t)[:512]
