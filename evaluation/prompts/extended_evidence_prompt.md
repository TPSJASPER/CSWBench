# Extended evidence-construction instruction

You build a descriptive evidence profile only: you do not score and you do not redo event discovery.

**Evidence identifiers.** Every `evidence` and `profile_evidence` entry may only use a `P####` or `T####` identifier that exists in the input directory, preferring one and allowing at most two per item.

**Scenes.** `narrative_system.scenes` is the only scene timeline and is filled in text order, with the number of scenes determined by the article itself.

Each scene must record, within the same entry:

    - the relationship state;
    - the state of character A;
    - the state of character B.

When a character is absent:

    - set `presence=NOT_PRESENT`;
    - fill other inapplicable states with `null` or an empty array;
    - set `change_from_previous` to `NOT_PRESENT`;
    - do not invent goals, emotions, or actions for that character.

When a character is present but nothing changes, set `change_from_previous` to `UNCHANGED`.

The first scene sets `causal_from_previous` to `null`. Later scenes record the causal relation to the immediately preceding scene, or `NO_DIRECT_CAUSAL_LINK` when there is none.

Non-adjacent, delayed, or global causality goes into `global_causal_summary`. No scene may be added merely to express cross-scene causality.

The local program injects `S001`, `S002`, and so on solely according to array position, and the model must not guess or generate these identifiers.

**Licensing views.** For each frozen event, construct an event-time evidence view from the visible Prompt, pre-narrative baselines, and narrative states established before that event.

**Fidelity and brevity.** Preserve:

    - world facts;
    - character baselines;
    - language resources;
    - characters' non-switching behaviour;
    - turning points;
    - the ending;
    - narrative voice.
