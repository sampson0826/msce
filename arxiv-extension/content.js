// MSCE arXiv Extension — Content Script
// Detects cosmology papers and displays constraint conflict status.

(function () {
  "use strict";

  const ABS_URL = window.location.href;

  // Only run on abstract pages
  if (!ABS_URL.match(/arxiv\.org\/abs\//)) return;

  // Extract arXiv ID
  const arxivId = ABS_URL.split("/abs/")[1]?.split(/[?#]/)[0];
  if (!arxivId) return;

  // Check if paper is in cosmology categories
  const subjectsEl = document.querySelector(".subjects");
  const subjects = subjectsEl?.textContent || "";
  const isCosmology =
    /astro-ph\.CO|gr-qc|hep-ph|hep-th/.test(subjects) &&
    /cosmolog|dark energy|dark matter|inflation|CMB|Hubble/.test(
      document.title + " " + (document.querySelector(".abstract")?.textContent || "")
    );

  if (!isCosmology) return;

  // Create banner
  const banner = document.createElement("div");
  banner.id = "msce-banner";
  banner.innerHTML = `
    <div style="
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      border: 1px solid #e74c3c;
      border-radius: 8px;
      padding: 12px 16px;
      margin: 12px 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 18px;">🔍</span>
        <div>
          <span style="color: #e74c3c; font-weight: 700; font-size: 14px;">MSCE</span>
          <span style="color: #aaa; font-size: 12px; margin-left: 8px;">Constraint Conflict Detector</span>
        </div>
      </div>
      <p style="color: #ccc; font-size: 13px; margin: 8px 0 0 0;">
        This paper makes claims about cosmology.
        <a href="https://github.com/msce-ai/msce" target="_blank" style="color: #3498db;">
          Check constraint consistency →
        </a>
      </p>
    </div>
  `;

  // Insert after abstract
  const abstractEl = document.querySelector(".abstract");
  if (abstractEl) {
    abstractEl.parentNode.insertBefore(banner, abstractEl.nextSibling);
  }
})();
