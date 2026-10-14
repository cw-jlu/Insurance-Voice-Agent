# Consultation demo design

## Scene and register

Product interface for an operator presenting in a well-lit meeting room. Light surfaces support projection, while one red accent anchors the current action. The page exposes a working consultation rather than a marketing story.

## Palette

Restrained palette. Primary seed `oklch(0.563 0.223 11)`; neutral surface `oklch(1 0 0)`, background `oklch(0.975 0.003 11)`, ink `oklch(0.245 0.012 11)`, muted text `oklch(0.48 0.014 11)`. Green and ochre carry semantic completion and missing-condition states.

## Layout and typography

Single system sans family with Chinese fallbacks. 14px base, 20px page title, 16px conversation text (15px on mobile). The default screen is one centered conversation, up to 860px wide. Only voice on/off, stop, playback speed and text submission remain beside the composer. Settings and evidence are separate, initially hidden right panels. On large screens the open panel narrows the conversation; on smaller screens it overlays, with an explicit close control and backdrop. Measurements, provenance, profile, TODOs, history and rehearsal tools stay available through progressive disclosure.

## Components and behavior

Native buttons, selects, textarea and disclosures. 8–14px radii, solid separators, no decorative shadows. Muted controls still use readable contrast. 180ms state transitions, reduced-motion support and visible keyboard focus. Continuous microphone status and partial transcription are visible in the composer. Local sustained speech pauses playback; recognized substantive speech cancels the old generation, while backchannels and likely echo resume it. Loading, cancellation, service errors, unavailable microphone and text-only playback confirmations have explicit states.

## Content rules

Replay is labeled historical and retains its original failures. Real-time Qwen is a named comparison mode and never presented as the frozen DeepSeek benchmark. Actions without real integrations say only that a request was recorded. Version labels identify reproducible engineering artifacts, not validated completion of the competition goals.
