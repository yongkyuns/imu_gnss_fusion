(function () {
  const TOLERANCE_PX = 8;

  function updateMathOverflow() {
    document.querySelectorAll(".math-wrapper").forEach((wrapper) => {
      wrapper.classList.remove("math-no-scroll");
      const overflowPx = wrapper.scrollWidth - wrapper.clientWidth;
      if (overflowPx > 0 && overflowPx <= TOLERANCE_PX) {
        wrapper.classList.add("math-no-scroll");
      }
    });
  }

  function scheduleMathOverflowUpdate() {
    window.requestAnimationFrame(() => {
      updateMathOverflow();
      window.setTimeout(updateMathOverflow, 250);
    });
  }

  function observeMathJaxMutations() {
    if (!window.MutationObserver || !document.body) {
      return;
    }
    let mutationTimer = 0;
    const observer = new MutationObserver(() => {
      window.clearTimeout(mutationTimer);
      mutationTimer = window.setTimeout(updateMathOverflow, 80);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    window.setTimeout(() => observer.disconnect(), 8000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      scheduleMathOverflowUpdate();
      observeMathJaxMutations();
    }, { once: true });
  } else {
    scheduleMathOverflowUpdate();
    observeMathJaxMutations();
  }

  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(scheduleMathOverflowUpdate).catch(() => {});
  }

  if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
    window.MathJax.startup.promise.then(scheduleMathOverflowUpdate).catch(() => {});
  }

  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(updateMathOverflow, 120);
  });
})();
