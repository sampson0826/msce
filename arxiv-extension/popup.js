// MSCE Popup — stats display

document.addEventListener('DOMContentLoaded', () => {
  chrome.storage.local.get(['papersScanned', 'conflictsFound'], (data) => {
    document.getElementById('papers-scanned').textContent = data.papersScanned || 0;
    document.getElementById('conflicts-found').textContent = data.conflictsFound || 0;
  });
});
