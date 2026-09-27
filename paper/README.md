# Spatial Atlas manuscript

This folder holds the revised manuscript "Spatial Atlas: Compute-Grounded Reasoning for Spatial-Aware Research Agent Benchmarks," by Arun Sharma. I prepared it as arXiv replacement v3 of 2604.12102, and v3 is not yet on arXiv. The file `spatial_atlas.pdf` is the compiled manuscript, and the folder `source/` holds its LaTeX source, its three TikZ figures, and the style file it uses.

## What the manuscript reports

The manuscript describes compute-grounded reasoning, a design pattern in which code computes selected sub-problems from an explicit intermediate representation before a language model answers. It reports one private label-free operational run, called V37, in which four paths each wrote eight prediction rows with zero retries. Labels stayed sealed and no score was computed, so that run establishes operational integrity only. The manuscript reports no FieldWorkArena result, because the benchmark data were not accessible, and it reports no accuracy, latency, or resource-use result.

This version removes the evaluation tables of the earlier version, because no run artifact backs the values they reported. It also corrects the system description and the bibliography against the released code. The earlier version is on arXiv as [2604.12102v2](https://arxiv.org/abs/2604.12102v2), and this repository withdraws the values it reports.

The root [README](../README.md) follows one task through the code, and it cites the manuscript's equation numbers for the rules that the code applies.

## Folder structure

```text
paper/
├── README.md                     # this file
├── source/                       # LaTeX source, prepared for the arXiv v3 replacement
│   ├── 00README.json             # arXiv build settings (pdflatex, with main.tex as the top-level file)
│   ├── figures/                  # TikZ sources of the three manuscript figures
│   │   ├── fig_architecture.tex  # Figure 1, the system architecture
│   │   ├── fig_arms.tex          # Figure 3, the four V37 arms
│   │   └── fig_bridge.tex        # Figure 2, the strict metric bridge
│   ├── main.tex                  # manuscript text, equations, tables, and bibliography
│   └── neurips_2020.sty          # NeurIPS style file for the page layout
└── spatial_atlas.pdf             # compiled manuscript, 20 pages, and the shipped copy
```

## Build the PDF

Run `pdflatex` three times from `paper/source/`. The source carries its own bibliography, so no BibTeX step is needed. The build uses standard TeX Live packages, including TikZ, pgfplots, siunitx, and cleveref.

```bash
cd paper/source
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
```

The build writes `main.pdf` in `paper/source/`, along with the auxiliary files `main.aux`, `main.log`, and `main.out`. It does not replace `paper/spatial_atlas.pdf`, which is the shipped copy of the manuscript.

## License

The manuscript, its LaTeX source, and its figures are copyright Arun Sharma, and all rights are reserved. The MIT License at the root of this repository covers the code, and it does not cover the manuscript. The file `source/neurips_2020.sty` is the NeurIPS style file. It is not my work, and neither the MIT License nor my reserved rights cover it. The earlier version of the manuscript on arXiv (arXiv:2604.12102) carries a CC BY 4.0 license, and that license still applies to that version.
