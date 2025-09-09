# Cataract‑LMM — Dataset Layout & Access

> **At a glance**
>
> - **Raw source videos:** 3,000  
> - **Instance segmentation frames:** 6,094  
> - **Phase‑recognition clips:** 150  
> - **Segmentation source clips:** 150  
> - **Instrument‑tracking clips:** 170  
> - **Skill‑assessment clips:** 170

> **Procedures per site:** 3,000 procedures, comprising 2,930 from Farabi Hospital and 70 from Noor Hospital.

This repository provides the **directory structure and documentation** for the Cataract‑LMM dataset. **It does not include any media or annotations.** Access to the actual data is described below.

---

## 🔑 Access & Availability

- **Production data hosting:** The dataset will be made available **via Synapse after the publication of the article**.  
- **Pre‑publication access request:** Complete the Google Form to request access:  
  https://docs.google.com/forms/d/e/1FAIpQLSfmyMAPSTGrIy2sTnz0-TMw08ZagTimRulbAQcWdaPwDy187A/viewform?usp=dialog
- See [`data/README.md`](data/README.md) for a concise access note.

---

## 🧭 Global Naming Convention

**Hospitals (site codes) 🏥**
- `S1` → **Farabi Hospital**
- `S2` → **Noor Hospital**

**Subset prefixes**
- `RV` = Raw Video  
- `PH` = Phase Recognition  
- `SE` = Instance Segmentation  
- `TR` = Instrument Tracking
- `SK` = Skill Assessment

**Indexing & zero‑padding**
- All counters are **1‑based** and **zero‑padded**  
  - `RawVideoID` = 4 digits (e.g., `0002`)  
  - `ClipID`     = 4 digits (subset‑local index)  
  - `SubclipOrder` = 4 digits (temporal order in the **original raw video**)  
  - `PhaseID`    = 2 digits (e.g., `P01`)  
  - `PhaseOccurrence` = 2 digits (the *n*‑th time that phase appears in the **original raw video**)  
  - `FrameIndex` = 7 digits (frame number **within the clip**)

> **Important logic for phase sub‑clips**  
> In filenames like `PH_0001_0002_S1_0001_P01_01.mp4`:
> - The `0001` **after `S1`** is `SubclipOrder` → the **temporal order of the sub‑clip in the original raw video** (first extracted sub‑clip).
> - The trailing `01` is `PhaseOccurrence` → the **first time phase `P01` occurs** in that raw video.  
> Phases may recur; occurrences are counted even if other phases appear in between.

**Canonical patterns**

- **Raw video:**  
  `RV_<RawVideoID>_S<Site>.mp4`

- **Subset clip (videos/ & full‑video CSVs):**  
  `<PREFIX>_<ClipID>_<RawVideoID>_S<Site>.<ext>`

- **Phase sub‑clip (phase‑specific segments):**  
  `PH_<ClipID>_<RawVideoID>_S<Site>_<SubclipOrder>_P<PhaseID>_<PhaseOccurrence>.mp4`

- **Per‑frame images (segmentation):**  
  `SE_<ClipID>_<RawVideoID>_S<Site>_<FrameIndex>.jpg`

- **Tracking video:**  
  `TR_<ClipID>_S<Site>_P03.mp4`

- **Tracking annotation folder:**  
  `TR_S<Site>_<ClipID>_P03/`  *(note: folder puts Site before ClipID)*

- **Tracking frames inside a tracking folder:**  
  `TR_<ClipID>_S<Site>_P03_<FrameIndex>.jpg`

- **Skill video:**  
  `SK_<ClipID>_S<Site>_P03.mp4`

---

## 🧩 Taxonomies (authoritative names)

### Surgical phases
The phase names used throughout the repository are:

- Incision
- Viscoelastic
- Capsulorhexis
- Hydrodissection
- Phacoemulsification
- Irrigation-Aspiration
- Capsule Polishing
- Lens Implantation
- Lens Positioning
- Viscoelastic Suction
- Anterior Chamber Flushing
- Tonifying-Antibiotics
- Idle

> **Note:** In the tracking and skill‑assessment subsets, `P03` denotes **Capsulorhexis**.

### Instance‑segmentation classes
12 classes: two ocular structures (**pupil**, **cornea**) and ten surgical instruments (**Primary knife**, **Secondary knife**, **Capsulorhexis cystotome**, **Capsulorhexis forceps**, **Phaco handpiece**, **I/A handpiece**, **Second instrument**, **Forceps**, **Cannula**, and **Lens injector**).

### Skill‑assessment metrics (names only)
The following **six** indicators are used to score the **capsulorhexis (P03)** clips:
- Instrument Handling
- Motion
- Tissue Handling
- Microscope Use
- Commencement of Flap
- Circular Completion

*(Only metric names are disclosed here; detailed scoring methods are in the paper.)*

---

## 📁 Dataset Structure (overview)

```text
Cataract-LMM/
│
├── README.md
│   └─ 📄 Main README with overview, setup, and citation instructions.
│
├── LICENSE.md
│   └─ 📄 Custom data usage license (full text).
│
├── 1_Raw_Videos/
│   │  Raw, unprocessed source videos (total: 3,000).
│   │  🧭 Pattern: RV_<RawVideoID>_S<Site>.mp4
│   │  • Decoding: RV_0001_S1.mp4 → RawVideoID=0001; Site=S1 (Farabi).
│   │
│   ├── README.md
│   │  └─ 📄 Notes specific to the raw collection.
│   └── videos/
│       ├── RV_0001_S1.mp4
│       │   └─ 🎬 Raw Video 0001 @ Farabi (S1).
│       ├── RV_0002_S2.mp4
│       │   └─ 🎬 Raw Video 0002 @ Noor (S2).
│       └── ...  (3,000 total)
│
├── 2_Phase_Recognition/
│   │  Surgical workflow (phase) analysis.
│   │
│   ├── README.md
│   │  └─ 📄 Subset details and phase taxonomy notes.
│   ├── videos/
│   │  │  Curated clips used for phase recognition (150 total).
│   │  │  🧭 Pattern: PH_<ClipID>_<RawVideoID>_S<Site>.mp4
│   │  │  • <ClipID> is the subset-local index within this folder (1…150).
│   │  ├── PH_0001_0002_S1.mp4
│   │  │   └─ 🎬 Subset ClipID=0001; RawVideoID=0002; Site=S1 (Farabi).
│   │  └── ...
│   │
│   ├── annotations_full_video/
│   │  │  One CSV per subset clip with frame-level phase labels (150 CSVs).
│   │  │  🧭 Pattern: PH_<ClipID>_<RawVideoID>_S<Site>.csv  (same stem as its video)
│   │  ├── PH_0001_0002_S1.csv
│   │  │   └─ 📄 Labels for PH_0001_0002_S1.mp4 (full-clip timeline; one label per frame).
│   │  └── ...
│   │
│   └── annotations_sub_clips/
│       │  Phase-specific segments extracted from each subset clip.
│       │  📁 One folder per subset clip:
│       │  🧭 Folder pattern: PH_<ClipID>_<RawVideoID>_S<Site>/
│       │  🧭 File pattern:   PH_<ClipID>_<RawVideoID>_S<Site>_<SubclipOrder>_P<PhaseID>_<PhaseOccurrence>.mp4
│       │   • <SubclipOrder> (4 digits) = temporal order of this sub-clip in the **original raw video**.
│       │   • <PhaseOccurrence> (2 digits) = the n-th time that phase appears in the **original raw video**.
│       ├── PH_0001_0002_S1/
│       │   ├── PH_0001_0002_S1_0001_P01_01.mp4
│       │   │   └─ 🎬 ClipID=0001; RawVideoID=0002; Site=S1; SubclipOrder=0001 (first sub-clip in raw timeline);
│       │   │      PhaseID=P01; PhaseOccurrence=01 (first occurrence of P01 in that raw video).
│       │   └── ...
│       └── ...
│
├── 3_Instance_Segmentation/
│   │  Instrument/tissue instance segmentation resources.
│   │
│   ├── README.md
│   │  └─ 📄 Subset details and annotation schema.
│   ├── videos/
│   │  │  Source clips from which frames were sampled (150 total).
│   │  │  🧭 Pattern: SE_<ClipID>_<RawVideoID>_S<Site>.mp4
│   │  ├── SE_0001_0002_S1.mp4
│   │  │   └─ 🎬 ClipID=0001; RawVideoID=0002; Site=S1.
│   │  └── ...
│   │
│   └── annotations/
│       ├── coco_json/
│       │   ├── images/
│       │   │   │  All annotated frames: 6,094 images.
│       │   │   │  🧭 Pattern: SE_<ClipID>_<RawVideoID>_S<Site>_<FrameIndex>.jpg
│       │   │   ├── SE_0002_0003_S2_0000045.jpg
│       │   │   │   └─ 🖼 ClipID=0002; RawVideoID=0003; Site=S2; FrameIndex=0000045 (45th frame of this clip).
│       │   │   └── ... (6,094 total)
│       │   └── annotations.json
│       │       └─ 📄 COCO annotations enumerating all 6,094 frames.
│       │
│       └── yolo_txt/
│           ├── images/
│           │   ├── SE_0002_0003_S2_0000045.jpg
│           │   │   └─ 🖼 Same image split as COCO/images.
│           │   └── ... (6,094 total)
│           └── labels/
│               ├── SE_0002_0003_S2_0000045.txt
│               │   └─ 🏷 Label file matching the image stem (one-to-one).
│               └── ... (6,094 total)
│
├── 4_Instrument_Tracking/
│   │  Spatiotemporal analysis of instruments (capsulorhexis).
│   │
│   ├── README.md
│   │  └─ 📄 Subset details and tracking format.
│   ├── videos/
│   │  │  170 tracking clips.
│   │  │  🧭 Pattern: TR_<ClipID>_S<Site>_P03.mp4
│   │  ├── TR_0003_S1_P03.mp4
│   │  │   └─ 🎬 ClipID=0003; Site=S1; P03 indicates capsulorhexis phase.
│   │  └── ...
│   │
│   └── annotations/
│       │  One folder per tracking clip (170 total).
│       │  🧭 Folder pattern: TR_<ClipID>_S<Site>_P03/
│       │  Files inside follow the standard TR_<ClipID>_S<Site>_P03_<FrameIndex>.jpg stem.
│       ├── TR_0003_S1_P03/
│       │   ├── TR_0003_S1_P03.json
│       │   │   └─ 📄 Dense frame-by-frame tracking for this clip.
│       │   ├── TR_0003_S1_P03_0000001.jpg
│       │   │   └─ 🖼 FrameIndex=0000001 (first frame of the clip).
│       │   ├── TR_0003_S1_P03_0000002.jpg
│       │   │   └─ 🖼 FrameIndex=0000002 (second frame), etc.
│       │   └── ...
│       └── ...
│
└── 5_Skill_Assessment/
    │  Objective surgical skill ratings.
    │
    ├── README.md
    │  └─ 📄 Subset details and scoring rubric.
    ├── videos/
    │  │  🧭 Pattern: SK_<ClipID>_S<Site>_P03.mp4
    │  ├── SK_0003_S1_P03.mp4
    │  │   └─ 🎬 Skill-assessment clip; ClipID=0003; Site=S1; P03 indicates capsulorhexis phase.
    │  └── ...
    │
    └── annotations/
        └── skill_scores.csv
            └─ 📄 One row per clip (170 rows), keyed by SK/TR stem; columns = skill metrics.
```


## 🔍 Worked Examples (decoding)

- `RV_0002_S2.mp4` → `RV` (raw) • `RawVideoID=0002` • `S2` (Noor).  
- `PH_0001_0002_S1.mp4` → `PH` (phase) • `ClipID=0001` • `RawVideoID=0002` • `S1` (Farabi).  
- `PH_0001_0002_S1.csv` → CSV labels aligned to `PH_0001_0002_S1.mp4`.  
- `PH_0001_0002_S1_0001_P01_01.mp4` → Sub‑clip for **phase `P01`**.  
- `SE_0001_0002_S1.mp4` → Segmentation source clip.  
- `SE_0002_0003_S2_0000045.jpg` → Segmentation frame.  
- `TR_0003_S1_P03.mp4` → Tracking clip; `ClipID=0003`; `S1`; `P03` indicates **Capsulorhexis**.  
- `TR_S1_0003_P03/` → Annotation folder for the same clip.  
  - `TR_0003_S1_P03_0000001.jpg` → Frame 1 (`FrameIndex=0000001`).  
- `SK_0003_S1_P03.mp4` → Skill‑assessment video; `ClipID=0003`; `S1`; `P03` indicates **Capsulorhexis**.  
- `skill_scores.csv` → One record per `SK`/`TR` stem.

## 🧠 Quick Reference

| Element | Meaning | Scope |
|---|---|---|
| `RawVideoID` (`0001`…`3000`) | ID of the **source raw video** | Global across all raw videos |
| `ClipID` (`0001`…) | Index **within a subset’s `videos/` folder** | Local to each subset |
| `S1` / `S2` | **S1 = Farabi**, **S2 = Noor** | Site/Hospital |
| `SubclipOrder` (`0001`…) | Temporal order of the **sub‑clip** in the **original raw video** | Per raw video |
| `Pxx` | Surgical **phase** code (e.g., `P01`, `P03`) | Phase taxonomy |
| `PhaseOccurrence` (`01`…) | The n‑th time **that phase** appears in the **original raw video** | Per raw video & phase |
| `FrameIndex` (`0000001`…) | Frame number **within a given clip** | Per clip |

---

## 📜 License & Attribution

This dataset is released under **CC BY 4.0**. Proper attribution is required. 

**For detailed ownership and licensing information, see:**
- [`LICENSE.md`](LICENSE.md) - Local license file
- [`../DATA_LICENSE.md`](../DATA_LICENSE.md) - Complete data ownership and attribution requirements

---

## 🏥 Site Distribution

3,000 procedures, comprising 2,930 from Farabi Hospital and 70 from Noor Hospital.

---

## 📣 Citation

For citation information, please refer to the [main repository README](../README.md#-citation).
