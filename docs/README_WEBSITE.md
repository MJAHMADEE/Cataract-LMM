# Cataract-LMM Project Website

This directory contains the source code for the breathtaking, ultra-professional Github Pages website for **Cataract-LMM: A Large-Scale, Multi-Source, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis**.

## Architecture & Stack
- Single-page HTML5 architecture (`index.html`)
- **Tailwind CSS** (via CDN for zero-build-step deployment)
- Vanilla JavaScript for IntersectionObserver scroll animations and number counters
- Custom "Sci-Tech Minimalist" styling with glassmorphism effects

## How to Customize and Deploy

### 1. Replace Placeholder Assets
The website uses placeholder images to maintain its structure. You need to replace these with actual figures from your paper or dataset. 
Place your high-resolution images in the `docs/assets/` folder, ensuring they match these exact filenames (or update `index.html` to reflect your new filenames):

* `docs/assets/hero-bg.mp4` (Optional: Background video for the hero section. Uncomment the `<video>` tag in `index.html` if used).
* `docs/assets/teaser-figure.png` (Used in the Abstract section).
* `docs/assets/task-1-phase.png` (Used in the Bento Box for Phase Recognition).
* `docs/assets/task-2-seg.png` (Used in the Bento Box for Instance Segmentation).
* `docs/assets/task-3-track.png` (Used in the Bento Box for Object Tracking).
* `docs/assets/task-4-skill.png` (Used in the Bento Box for Skill Assessment).

*Tip: For the bento box images, images with dark backgrounds or transparent PNGs blend best with the glowing card effects.*

### 2. Update Links
Open `index.html` and search for `#` or `href=""` to update the actual links when they become available:
* ArXiv Paper Link
* Hugging Face Dataset Link
* Update the `assets/ZipContentsReport.csv` link in the Modular Data Access section if needed.

### 3. Deploy via GitHub Pages
Since the site is completely static, deploying is trivial:

1. Push these files to your repository (make sure they are in the `docs/` folder or root depending on your preference. Currently setup for `docs/`).
2. Go to your GitHub repository -> **Settings** -> **Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Select the `main` branch and the `/docs` folder.
5. Click **Save**.

GitHub actions will automatically build and deploy your site within a few minutes.