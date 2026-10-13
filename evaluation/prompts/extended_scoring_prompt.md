# Extended quality-judge instruction

You evaluate a blinded long-form Chinese-dominant multilingual narrative. Score only from the supplied article, actual visible Prompt, frozen event manifest, and frozen evidence manifest.

**Output identity.** Return exactly one scoring call and no prose. Copy the blinded document identifier and article hash exactly from the input, set the judge model to the exact model identifier used by this request, judge every frozen event exactly once using its frozen identifier, and score exactly the seven whole-text dimensions listed below.

**Whole-text anchor.** The seven whole-text dimensions use anchored integers. 1: seriously unsupported, unstable, ineffective, or damaging. 2: substantial weaknesses with limited successful evidence. 3: adequate and generally supported, with meaningful limitations. 4: strong, sustained, and well-supported execution. 5: exceptional, precise, sustained, and difficult to improve materially.

**Event judgments.** Evaluate every frozen event exactly once.

The *Licensing gate* is not numeric. For each frozen event, Licensing uses its event-time evidence view. It records two independent binary checks: a participant/role check asking whether the attributed speaker or narrative voice has the identity, language history, and resources needed for this particular use of English, and a world check asking whether the addressee, carrier, scene, relationship state, social rules, established facts, and prior discourse state permit the choice. Each check receives `PASS` or `FAIL`, and an event passes only if both pass.

*Contextual and narrative appropriateness* is scored 1--5 for how well the multilingual choice fits its local narrative and discourse context: 1, contextually unsupported or conflicting; 2, weak or partial contextual fit; 3, clear contextual fit supported by concrete local evidence; 4, strong and specific contextual fit; 5, precise contextual realization, for which the particular wording, form, and placement are well supported by the context where relevant. The judgment jointly considers the following evidence checks:

**Voice ownership:** whether the switch is clearly attributable to the current character, interior voice, quotation, or other discourse source, while remaining consistent with surrounding voice boundaries.

**Carrier and focalization:** whether the English fits its current carrier---such as dialogue, writing, mediated text, or interior speech---and is compatible with the narrative viewpoint and information available at that position.

**Addressee and relationship:** whether the switch fits the addressee's language capacity, identity, immediate response, and the established relationship history between the characters.

**Topic:** whether the current topic, concept, source wording, or referent provides a specific contextual basis for using English.

**Emotion and power:** whether the switch is compatible with the current emotional state, intimacy, defensiveness, pressure, or distribution of power between the characters.

**Stance and identity:** whether the switch fits the character's current claim, alignment, concession, boundary-drawing, self-positioning, or identity presentation.

**Register:** whether the formality, professional association, social style, or other register carried by the English fits the current relationship, occasion, and action.

**Local switch boundary:** whether the switch occurs at a position that fits the local syntax, rhythm, turn structure, or narrative boundary.

These are evidence checks that jointly support one Contextual and narrative appropriateness score, not separate scored dimensions.

*Semantic and aesthetic net gain* is scored 1--5 for the added value provided by the multilingual choice within the current event: 1, the relevant function is absent or poorly realized; 2, weak, limited, or largely redundant functional value; 3, a clear, observable, and locatable contribution within the current event; 4, a strong and specific contribution that actively does work in the event; 5, a substantial, precise, and non-redundant contribution tightly integrated into the current event. The judgment jointly considers the following evidence checks:

**Semantic precision:** whether the English preserves or adds word sense, conceptual boundaries, source relations, ambiguity, or verbal effects that are not fully carried by an ordinary Chinese alternative.

**Relational or stance signal:** whether the switch strengthens intimacy, alignment, distance, power, attitude, or other interactional subtext between the characters.

**Identity-cultural connection:** whether the switch connects specifically to the character's language experience, cultural background, social identity, or current self-positioning.

**Stylistic contribution:** whether the switch makes an actual aesthetic or stylistic contribution through rhythm, syntax, voice contrast, imagery, or register contrast.

**Experiential texture:** whether the switch makes an action, memory, bodily experience, medium, institution, or concrete lived experience more specific and credible.

These are evidence checks that jointly support one Semantic and aesthetic net gain score, not separate scored dimensions.

**Event salience.** Each judge assigns its own salience label to each frozen event in the same scoring pass: S1 (weight 1) has little or no consequence beyond the immediate moment; S2 (weight 2) introduces a clear change with consequences for the current scene or subsequent narrative development; S3 (weight 3) introduces a major change whose consequences persist or substantially shape later narrative development. Judges determine salience from the surrounding narrative and the consequences of the event. Salience weights only Contextual Appropriateness and Functional Gain, with each judge's labels remaining paired with that judge's own scores. Each event judgment and each salience label carries at least one short concrete evidence string and a concise rationale, and must not merely cite an identifier or restate a score.

**Whole-text dimensions.** Score exactly seven and do not produce a parent aggregate.

*Multilingual narrative-system stability* asks whether speaker ownership, dialogue, interiority and quotation boundaries, language carriers, focalization, translation policy, and multilingual facts remain intelligible and consistent across the article, including whether later scenes honour or coherently develop earlier language-use rules.

*Character A's bilingual profile and language arc* asks whether A remains the same coherent multilingual person while changing in textually supported ways, considering repertoire, language function, relationship and affect mapping, scene-state responsiveness, transitions, language arc, and motivated non-switching. Mere event count is not rewarded.

*Character B's bilingual profile and language arc* applies the same standard independently to B, considering repertoire, function, relationship and affect mapping, scene-state responsiveness, transitions, language arc, and motivated non-switching. B's score must not be inferred from A's, and symmetrical language use is not required.

*Structure, causality, pacing, and closure* assesses scene progression, causal continuity, escalation, turning points, proportional pacing, and whether the ending reflects the main actions and consequences.

*Character and relationship development completion* assesses whether both characters' goals, knowledge, emotions, actions, and transitions support their development and whether relationship change is earned by prior interaction.

*Scene, dialogue, prose, and emotional execution* assesses scene-to-scene execution, dialogue distinctiveness, prose clarity and control, sensory and atmospheric specificity, emotional credibility, and reader engagement without rewarding ornamental or elaborate prose for its own sake.

*Whole-text multilingual literary integration* assesses whether multilingual resources create sustained character-voice differentiation, cultural and social verisimilitude, stylistic identity, and meaningful relations among social voices across the full narrative.

Every whole-text dimension carries at least one concrete evidence string and a rationale that explains the score against the whole-text anchor.
