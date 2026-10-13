# CSWBench Dataset Card

CSWBench evaluates context-sensitive Chinese–English code-switching in creative writing using **807 final Prompt Cards**.

| Setting | Records | Soft length target | Task type |
| --- | ---: | --- | --- |
| Focused | 407 | approximately 700 non-whitespace Unicode code points | `focused_narrative` |
| Extended | 400 | approximately 4,000 non-whitespace Unicode code points | `extended_narrative` |

The prompts invite narratives in which language choices follow characterization and context. The model decides whether, where and how to switch. Both tasks use the same event-level D1/D2/D3 concepts, with task-specific judging granularity and whole-text metrics.

## Data fields

Each UTF-8 JSONL record contains `prompt_id`, `task_type`, `prompt_text`, `source_dataset`, `scene_domain`, `participant_relation`, `interaction_pattern`, `preferred_style`, `length_target_codepoints`. The four-paragraph Prompt Card is the user-message content; Focused Narrative additionally uses a fixed system instruction.

The final set contains **146 ASCEND-derived** and **661 SwitchLingua-derived** prompts. The 407/400 prompts are mapped from **546 distinct Seeds**: 261 shared across tasks, 146 Focused-only and 139 Extended-only. The datasets are source-overlapping rather than independent samples.

`data/prompt_seed_map.csv` lists that mapping per Prompt Card (`task,prompt_card_id,seed_id`, 807 rows). Seed identifiers use the CSWBench namespace (`SEED_0001`–`SEED_0546`) and contain no source text or upstream record identifiers; each card's source corpus remains available as `source_dataset`.

The source corpora contain bilingual interactions, but the released Prompt Cards are constructed creative-writing materials with fictionalized narrative participants. They should not be interpreted as demographic samples, profiles of identifiable individuals, or representative evidence about any real language community.

## Data integrity

`data/MANIFEST.json` stores SHA-256 hashes and row counts for the final JSONL files and the Seed map. `DATASET_STATS.json` and `PROMPT_REDUNDANCY.json` contain descriptive statistics. Run:

```bash
python scripts/recompute_prompt_stats.py --check
```

## Evaluation

CSWEval scores Licensing (D1), Contextual Appropriateness (D2), and Functional Gain (D3), together with task-specific whole-text quality, with a total score out of 20. See [evaluation/PROTOCOL.md](evaluation/PROTOCOL.md).

## License and citations

Original CSWBench code uses Apache-2.0. The released CSWBench Prompt Card content is licensed under CC BY-NC 4.0. The release does not include original ASCEND or SwitchLingua records; their applicable terms continue to govern those source records. See [DATA_LICENSE.md](DATA_LICENSE.md) for the license terms.

### Source dataset publications

**ASCEND.** Holy Lovenia, Samuel Cahyawijaya, Genta Indra Winata, Peng Xu, Yan Xu, Zihan Liu, Rita Frieske, Tiezheng Yu, Wenliang Dai, Elham J. Barezi, Qifeng Chen, Xiaojuan Ma, Bertram E. Shi, and Pascale Fung. 2022. *ASCEND: A Spontaneous Chinese-English Dataset for Code-switching in Multi-turn Conversation*. Proceedings of the Thirteenth Language Resources and Evaluation Conference (LREC 2022), pp. 7259–7268. [ACL Anthology](https://aclanthology.org/2022.lrec-1.788/).

```bibtex
@inproceedings{lovenia-etal-2022-ascend,
  title = "{ASCEND}: A Spontaneous {C}hinese-{E}nglish Dataset for Code-switching in Multi-turn Conversation",
  author = "Lovenia, Holy and Cahyawijaya, Samuel and Winata, Genta Indra and Xu, Peng and Xu, Yan and Liu, Zihan and Frieske, Rita and Yu, Tiezheng and Dai, Wenliang and Barezi, Elham J. and Chen, Qifeng and Ma, Xiaojuan and Shi, Bertram E. and Fung, Pascale",
  booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
  year = "2022",
  pages = "7259--7268",
  address = "Marseille, France",
  publisher = "European Language Resources Association",
  url = "https://aclanthology.org/2022.lrec-1.788/"
}
```

**SwitchLingua.** Peng Xie, Xingyuan Liu, Yequan Bie, Tsz Wai Chan, Yangqiu Song, Yang Wang, Hao Chen, and Kani Chen. 2025. *SwitchLingua: The First Large-Scale Multilingual and Multi-Ethnic Code-Switching Dataset*. Advances in Neural Information Processing Systems 38 (NeurIPS 2025), Datasets and Benchmarks Track. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4847447cae911dde321ca19afe7d8707-Abstract-Datasets_and_Benchmarks_Track.html).

```bibtex
@inproceedings{xie2025switchlingua,
  title = "{SwitchLingua}: The First Large-Scale Multilingual and Multi-Ethnic Code-Switching Dataset",
  author = "Xie, Peng and Liu, Xingyuan and Bie, Yequan and Chan, Tsz Wai and Song, Yangqiu and Wang, Yang and Chen, Hao and Chen, Kani",
  booktitle = "Advances in Neural Information Processing Systems",
  volume = "38",
  year = "2025",
  doi = "10.52202/085713-1683",
  url = "https://proceedings.neurips.cc/paper_files/paper/2025/hash/4847447cae911dde321ca19afe7d8707-Abstract-Datasets_and_Benchmarks_Track.html"
}
```
