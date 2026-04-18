// Landing-page motion and scroll logic.
//
// Three responsibilities, kept deliberately minimal:
//   1. Fade each section in once when it first enters the viewport.
//   2. Animate the stat counters ($33B / 19 / 71 / 114 / 80%) on first view.
//   3. Drive a thin left-edge section index that highlights the active
//      section and lets the user click to jump.
//
// No parallax, no scroll-linked transforms, no motion budget above ~1.5 s
// per interaction. This is the ICRC/DeepMind register — motion exists to
// aid comprehension, not to ornament.

(function () {
    "use strict";

    // ─── 1. Fade-in on scroll ──────────────────────────────────────────
    const faders = document.querySelectorAll("[data-fade]");
    if ("IntersectionObserver" in window) {
        const fadeObs = new IntersectionObserver((entries) => {
            entries.forEach((e) => {
                if (e.isIntersecting) {
                    e.target.classList.add("is-visible");
                    fadeObs.unobserve(e.target);
                }
            });
        }, { threshold: 0.12, rootMargin: "0px 0px -60px 0px" });
        faders.forEach((el) => fadeObs.observe(el));
    } else {
        faders.forEach((el) => el.classList.add("is-visible"));
    }

    // ─── 2. Counter animation ──────────────────────────────────────────
    // Elements carry:
    //   data-count-target   numeric target, e.g. "33"
    //   data-count-prefix   optional, e.g. "$"
    //   data-count-suffix   optional, e.g. "B" or "%"
    //   data-count-duration optional, ms (default 1100)
    const counters = document.querySelectorAll("[data-count-target]");
    const counterObs = "IntersectionObserver" in window
        ? new IntersectionObserver((entries) => {
            entries.forEach((e) => {
                if (e.isIntersecting) {
                    runCounter(e.target);
                    counterObs.unobserve(e.target);
                }
            });
        }, { threshold: 0.4 })
        : null;

    function runCounter(el) {
        const target   = parseFloat(el.dataset.countTarget);
        const prefix   = el.dataset.countPrefix || "";
        const suffix   = el.dataset.countSuffix || "";
        const duration = parseInt(el.dataset.countDuration || "1100", 10);
        const isInt    = Number.isInteger(target);
        const start    = performance.now();

        function frame(now) {
            const t = Math.min(1, (now - start) / duration);
            // ease-out cubic
            const eased = 1 - Math.pow(1 - t, 3);
            const v = target * eased;
            el.textContent = prefix + (isInt ? Math.round(v) : v.toFixed(1)) + suffix;
            if (t < 1) requestAnimationFrame(frame);
            else       el.textContent = prefix + (isInt ? target : target.toFixed(1)) + suffix;
        }
        el.textContent = prefix + (isInt ? 0 : "0.0") + suffix;
        requestAnimationFrame(frame);
    }

    if (counterObs) {
        counters.forEach((el) => counterObs.observe(el));
    } else {
        counters.forEach(runCounter);
    }

    // ─── 3. Section index on the left edge ─────────────────────────────
    // Highlights the section currently dominating the viewport and lets the
    // user click any dot to jump. Keeps the bar hidden below 900 px because
    // the single-column mobile layout doesn't need it.
    const sections = document.querySelectorAll("section[data-index]");
    const indexBar = document.getElementById("section-index");
    if (sections.length && indexBar) {
        sections.forEach((sec) => {
            const dot = document.createElement("a");
            dot.className = "section-dot";
            dot.href = "#" + sec.id;
            dot.setAttribute("aria-label", "Jump to " + (sec.dataset.indexLabel || sec.id));
            const label = document.createElement("span");
            label.className = "section-dot-label";
            label.textContent = sec.dataset.indexLabel || "";
            dot.appendChild(label);
            indexBar.appendChild(dot);
            sec.dataset.dotId = sec.id;
        });

        const dots = indexBar.querySelectorAll(".section-dot");
        const byId = new Map();
        sections.forEach((sec, i) => byId.set(sec.id, dots[i]));

        if ("IntersectionObserver" in window) {
            const secObs = new IntersectionObserver((entries) => {
                entries.forEach((e) => {
                    const dot = byId.get(e.target.id);
                    if (!dot) return;
                    if (e.isIntersecting && e.intersectionRatio >= 0.35) {
                        dots.forEach((d) => d.classList.remove("is-active"));
                        dot.classList.add("is-active");
                    }
                });
            }, { threshold: [0.35, 0.5, 0.75] });
            sections.forEach((sec) => secObs.observe(sec));
        }
    }
})();
