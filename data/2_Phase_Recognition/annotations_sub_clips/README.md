# Phase Recognition — `annotations_sub_clips/`

Phase‑specific segments extracted from the curated clips.

- **Folder pattern per clip:** `PH_<ClipID>_<RawVideoID>_S<Site>/`
- **File pattern:** `PH_<ClipID>_<RawVideoID>_S<Site>_<SubclipOrder>_P<PhaseID>_<PhaseOccurrence>.mp4`

**Semantics**
- `SubclipOrder` (4 digits): temporal order of the sub‑clip in the **original raw video**.  
- `PhaseOccurrence` (2 digits): n‑th time a phase appears in the **original raw video**.

Media files are not included in this package.
