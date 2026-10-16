# Research Assistant — frontend concept v1

Generated using the built-in image generation tool on 2026-10-16.

## Design direction

A calm research workspace that preserves the existing Research Assistant name, teal identity, and six navigation sections. Modern sans-serif typography, selected-paper context, cited answers, and an integrated source inspector. Based on the design-taste-frontend and redesign-existing-projects skills, applying product-appropriate typography, spacing, and hierarchy guidance.

Screen: Ask your papers. The displayed content illustrates the existing research workflow.

Image: `frontend-concept-v1.png`

## Generation prompt

Use case: ui-mockup.
Asset type: one high-fidelity desktop web application design concept for the existing Research Assistant project.
Primary request: Generate a polished, practical frontend redesign image showing the real research workflow: select papers, ask a question, read a cited answer, inspect source evidence. This is a functional research application for students and researchers.

Composition: a single wide landscape desktop screen, approximately 16:10, straight-on and edge-to-edge. Crisp screen-design rendering, exceptionally readable UI text, meticulous spacing and aligned baselines. No laptop or phone device, no photography, no perspective, no external presentation captions. One cohesive screen, not a collage.

Design direction: calm, precise, contemporary, restrained. Evolve the existing teal identity. White surfaces, very pale cool gray-green workspace, deep charcoal text, deep teal #0e6b66 as the consistent action/selection color, fine quiet gray-green separators. Pale yellow is used ONLY to highlight a supporting phrase in the source excerpt. Geist or Satoshi-like sans-serif typography with clear medium/semibold hierarchy, small monospaced citation/page metadata. No serif headings. Comfortable whitespace, nuanced 6–12px corner radii, extremely restrained shadows. Product UI, not a marketing landing page. Do not add a hero, sales slogan, vanity metric cards, purple gradients, glass effects, colorful charts, glowing AI ornaments, or decorative illustrations.

Information architecture: preserve these exact six primary nav labels: "Library", "Ask", "Compare", "Gaps", "Discover", "Review". A clean 68px top navigation bar, brand "Research Assistant" on the left with a small simple outline book icon; six navigation items across the middle, "Ask" active with a subtle teal underline. A compact "OM" profile control at the far right. No permanent global navigation sidebar.

Below navigation: left-aligned page title "Ask your papers", one short muted sentence "Answers grounded in your selected research." A quiet secondary "Upload PDFs" control at right.

Main workspace: three asymmetrical functional areas, approximately 22% paper selection, 47% answer workspace, 31% source inspector. Keep meaningful content large enough to read. The left paper panel and right inspector use subtle surface tints, while the central content has open whitespace rather than many bordered cards.

LEFT PAPER PANEL:
Heading "Selected papers" with "3 papers" as small contextual text.
A small search input "Filter papers".
Three selected research-paper rows with checked teal checkboxes, small PDF glyphs, restrained ready status text, year, and title. Use these genuine titles, wrapped naturally:
"Detecting Parkinson's Disease From an Online Speech-task" — 2020
"Parkinsonian Chinese Speech Analysis towards Automatic Classification of Parkinson's Disease" — 2021
"Assessment of Parkinson's Disease Medication State through Automatic Speech Analysis" — 2020.
Highlight the first row with a very subtle teal tint because its source is open.
Bottom text link "Back to library".
Do not add folders, collaboration, notes, or unimplemented project management features.

CENTER ANSWER WORKSPACE:
A generous question composer with exact question: "Which speech representations and models are used to classify Parkinson's disease?"
At composer bottom, muted "Scope: 3 selected papers", an unchecked option "Route automatically", and a compact teal "Ask" button with a simple arrow icon.
Below, a clear "Answer" heading and a small factual status "Source support checked" with a restrained check icon. No percentage confidence score.
An easily read answer with three short paragraphs or bullets, using modest bold emphasis only for key model names:
"PASE encodes raw audio into feature vectors. SVM, Random Forest, LightGBM and XGBoost then classify PD and non-PD speech. [1]"
"The Mandarin Chinese speech study evaluates classical machine learning, convolutional networks and recurrent networks. [2]"
"The medication-state study uses acoustic-prosodic markers with a deep neural network to classify ON and OFF states. [3]"
Render [1], [2], [3] as small square interactive teal citation controls, [1] active.
Under answer, small muted text "Checked against cited passages. Open a source to inspect the evidence."
A simple "Sources" section showing three compact source rows with citation number, short paper title, and actual page/section cues. Row [1] selected and linked visually to the source inspector. Avoid giant repeated cards and avoid an additional chat conversation or a second composer.

RIGHT SOURCE INSPECTOR:
Clearly presented as an integrated panel with "Source passage" heading and close control.
A small [1] marker.
Title "Detecting Parkinson's Disease From an Online Speech-task".
Metadata "Introduction · Page 2".
A typographically clear excerpt in a clean paper-like reading area, with this actual supporting content:
"Encoder (PASE) represents the information contained in a raw audio instance through a list of encoded vectors. These features are modeled with four different machine learning models — Support-Vector-Machine, Random Forest, LightGBM, and XGBoost — to classify individuals with and without PD."
Use a soft pale-yellow highlight just behind "Support-Vector-Machine, Random Forest, LightGBM, and XGBoost", not the whole excerpt.
Small text link "Open full passage".
At bottom of this panel, a quiet citation navigation control "1 of 3 sources" with previous/next arrows.

Quality constraints: make this look like a carefully art-directed, buildable web product with consistent spacing, elegant typography, real readable content, clear active states and useful evidence traceability. Keep all six navigation labels and the existing Research Assistant name. Do not introduce new branding. No fabricated accuracy percentages, novelty guarantees, research performance graphs, provider API details, watermarks, or promotional copy. The chosen paper and source panel must agree.

