// Geo-Insight — interactive orthographic globe for the landing page.
//
// D3 v7 + TopoJSON. Renders Earth as a thin-line vector globe with the
// top-ten overlooked crises as oxblood hotspots. Drag to rotate. Click a
// hotspot for a sidecar country card. Design language matches
// shared.css: single accent, no textures, rule-coloured boundaries.

(function () {
    "use strict";

    // ─── Top-10 overlooked (HRP-eligible pool, n=22, 2025 cycle).
    //     `rank` is position in the posterior-median ordering (1 = most
    //     overlooked). `ci_width` is the 90 % credible interval width on
    //     the latent — the model's uncertainty about this country.
    //     `cerf` is membership in any CERF UFE allocation 2024–2025. ──
    const OVERLOOKED = [
        { iso: "HND", name: "Honduras",      rank:  1, ci_width: 0.55, type: "contested · sector-starved", cerf: true,  lat: 15.2,  lon: -86.2 },
        { iso: "SLV", name: "El Salvador",   rank:  2, ci_width: 0.53, type: "contested · sector-starved", cerf: false, lat: 13.8,  lon: -88.9 },
        { iso: "MOZ", name: "Mozambique",    rank:  3, ci_width: 0.52, type: "contested · balanced",       cerf: true,  lat: -18.7, lon:  35.5 },
        { iso: "SOM", name: "Somalia",       rank:  4, ci_width: 0.51, type: "consensus · overlooked",     cerf: true,  lat:  5.2,  lon:  46.2 },
        { iso: "GTM", name: "Guatemala",     rank:  5, ci_width: 0.58, type: "contested · sector-starved", cerf: false, lat: 15.5,  lon: -90.3 },
        { iso: "NER", name: "Niger",         rank:  6, ci_width: 0.50, type: "consensus · overlooked",     cerf: true,  lat: 17.6,  lon:   8.1 },
        { iso: "HTI", name: "Haiti",         rank:  7, ci_width: 0.53, type: "contested · sector-starved", cerf: true,  lat: 18.9,  lon: -72.3 },
        { iso: "CMR", name: "Cameroon",      rank:  8, ci_width: 0.51, type: "consensus · sector-starved", cerf: true,  lat:  7.4,  lon:  12.4 },
        { iso: "VEN", name: "Venezuela",     rank:  9, ci_width: 0.49, type: "consensus · overlooked",     cerf: true,  lat:  6.4,  lon: -66.6 },
        { iso: "TCD", name: "Chad",          rank: 10, ci_width: 0.48, type: "consensus · overlooked",     cerf: true,  lat: 15.5,  lon:  18.7 },
    ];

    const POOL_SIZE = 22;  // HRP-eligible pool, 2025 cycle

    // Default rotation: anchored on Honduras (#1), then drifting east on
    // load. Frames the Latin-America / West-Africa cluster the model
    // surfaces.
    const DEFAULT_ROTATION = [80, -10, 0];

    const COLORS = {
        ocean:     "#ffffff",
        land:      "#fafafa",
        landStroke:"#e8e8e8",
        graticule: "rgba(0, 0, 0, 0.035)",
        edge:      "#888888",
        hotspot:   "#7c1d1d",
        hotspotSel:"#5c1414",
        hotspotDim:"#c9554f",
        label:     "#111111",
        labelMuted:"#555555",
    };

    const container = document.getElementById("globe");
    const panel     = document.getElementById("globe-panel");
    if (!container) return;

    // Sizing — responsive: the container sets the width; height matches.
    const W = container.clientWidth;
    const H = Math.min(W, 560);
    const R = Math.min(W, H) / 2 - 12;
    const cx = W / 2;
    const cy = H / 2;

    const projection = d3.geoOrthographic()
        .scale(R)
        .translate([cx, cy])
        .clipAngle(90)
        .rotate(DEFAULT_ROTATION);

    const path = d3.geoPath(projection);
    const graticule = d3.geoGraticule10();

    const svg = d3.select(container).append("svg")
        .attr("viewBox", `0 0 ${W} ${H}`)
        .attr("width", "100%")
        .attr("height", H)
        .style("display", "block")
        .style("touch-action", "none");

    // Sphere outline + subtle drop
    const defs = svg.append("defs");
    const glow = defs.append("filter").attr("id", "hotspot-glow").attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
    glow.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "b");
    const glowMerge = glow.append("feMerge");
    glowMerge.append("feMergeNode").attr("in", "b");
    glowMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Layers (order matters for z-stacking)
    const gSphere    = svg.append("path").attr("id", "sphere");
    const gGraticule = svg.append("path").attr("id", "graticule");
    const gLand      = svg.append("g").attr("id", "land");
    const gEdge      = svg.append("path").attr("id", "edge");
    const gHotspots  = svg.append("g").attr("id", "hotspots");

    gSphere.datum({type: "Sphere"})
        .attr("fill", COLORS.ocean)
        .attr("stroke", "none")
        .attr("d", path);

    gGraticule.datum(graticule)
        .attr("fill", "none")
        .attr("stroke", COLORS.graticule)
        .attr("stroke-width", 0.6)
        .attr("d", path);

    gEdge.datum({type: "Sphere"})
        .attr("fill", "none")
        .attr("stroke", COLORS.edge)
        .attr("stroke-width", 1)
        .attr("d", path);

    // Selected-country fill layer (rendered behind hotspots)
    const selectedFill = svg.insert("path", "#hotspots")
        .attr("fill", "rgba(124, 29, 29, 0.12)")
        .attr("stroke", "none");

    let selectedIso = "HND";

    function render() {
        gSphere.attr("d", path({type: "Sphere"}));
        gGraticule.attr("d", path(graticule));
        gEdge.attr("d", path({type: "Sphere"}));
        gLand.selectAll("path").attr("d", path);

        // Hotspots: project each centroid; fade if on far side.
        const rot = projection.rotate();
        const [lambda, phi] = [-rot[0] * Math.PI / 180, -rot[1] * Math.PI / 180];
        const hotspots = gHotspots.selectAll("g.hotspot")
            .data(OVERLOOKED, d => d.iso);

        const hotspotEnter = hotspots.enter().append("g")
            .attr("class", "hotspot")
            .style("cursor", "pointer")
            .on("click", (_ev, d) => select(d.iso))
            .on("mouseenter", (ev, d) => tooltipShow(ev, d))
            .on("mouseleave", tooltipHide)
            .on("mousemove", tooltipMove);

        hotspotEnter.append("circle").attr("r", 6)
            .attr("fill", COLORS.hotspot)
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5);

        hotspotEnter.append("circle").attr("class", "halo")
            .attr("r", 10)
            .attr("fill", "none")
            .attr("stroke", COLORS.hotspot)
            .attr("stroke-width", 1.2)
            .attr("opacity", 0.35);

        hotspotEnter.append("text")
            .attr("x", 10).attr("y", 3)
            .attr("font-size", 10)
            .attr("font-family", "Inter, sans-serif")
            .attr("font-weight", 600)
            .attr("fill", COLORS.label)
            .text(d => d.iso);

        const all = hotspots.merge(hotspotEnter);
        all.each(function (d) {
            const coords = [d.lon, d.lat];
            const lonRad = d.lon * Math.PI / 180;
            const latRad = d.lat * Math.PI / 180;
            // Angular distance from current centre (to decide front/back):
            const centerLon = -projection.rotate()[0] * Math.PI / 180;
            const centerLat = -projection.rotate()[1] * Math.PI / 180;
            const cosc = Math.sin(centerLat) * Math.sin(latRad) +
                         Math.cos(centerLat) * Math.cos(latRad) * Math.cos(lonRad - centerLon);
            const visible = cosc > 0.02;
            const [x, y] = projection(coords);
            const isSel  = d.iso === selectedIso;
            d3.select(this)
                .attr("transform", `translate(${x}, ${y})`)
                .attr("display", visible ? null : "none");
            d3.select(this).select("circle:not(.halo)")
                .attr("r", isSel ? 8 : 6)
                .attr("fill", isSel ? COLORS.hotspotSel : COLORS.hotspot);
            d3.select(this).select(".halo")
                .attr("opacity", isSel ? 0.6 : 0.0);
            d3.select(this).select("text")
                .attr("fill", isSel ? COLORS.label : COLORS.labelMuted)
                .attr("font-weight", isSel ? 700 : 600);
        });

        // Selected country outline
        if (selectedCountryFeature) {
            selectedFill.datum(selectedCountryFeature).attr("d", path);
        }
    }

    // ─── Drag to rotate ────────────────────────────────────────────────
    let dragStart = null;
    let rotationStart = null;
    svg.call(
        d3.drag()
            .on("start", (ev) => {
                dragStart = [ev.x, ev.y];
                rotationStart = projection.rotate();
            })
            .on("drag", (ev) => {
                if (!dragStart) return;
                const dx = ev.x - dragStart[0];
                const dy = ev.y - dragStart[1];
                const sensitivity = 0.3;
                const lambda = rotationStart[0] + dx * sensitivity;
                const phi = Math.max(-70, Math.min(70, rotationStart[1] - dy * sensitivity));
                projection.rotate([lambda, phi, rotationStart[2]]);
                render();
            })
            .on("end", () => { dragStart = null; })
    );

    // ─── Slow idle rotation, pauses when user interacts ───────────────
    let autoRotating = true;
    let lastTs = 0;
    svg.on("pointerdown", () => { autoRotating = false; });

    function tick(ts) {
        if (!lastTs) lastTs = ts;
        const dt = ts - lastTs;
        lastTs = ts;
        if (autoRotating) {
            const rot = projection.rotate();
            projection.rotate([rot[0] + dt * 0.008, rot[1], rot[2]]);
            render();
        }
        requestAnimationFrame(tick);
    }

    // ─── Load world topology + render land ─────────────────────────────
    let countries = null;
    let selectedCountryFeature = null;

    d3.json("world-110m.json").then((world) => {
        countries = topojson.feature(world, world.objects.countries);
        gLand.selectAll("path")
            .data(countries.features)
            .join("path")
            .attr("fill", COLORS.land)
            .attr("stroke", COLORS.landStroke)
            .attr("stroke-width", 0.6)
            .attr("d", path);
        select(selectedIso);
        render();
        requestAnimationFrame(tick);
    });

    // ─── Selection + sidecar panel ─────────────────────────────────────
    const ISO_NUMERIC = {
        HND: "340", SLV: "222", MOZ: "508", SOM: "706", GTM: "320",
        NER: "562", HTI: "332", CMR: "120", VEN: "862", TCD: "148",
    };

    function select(iso) {
        selectedIso = iso;
        if (countries) {
            const num = ISO_NUMERIC[iso];
            selectedCountryFeature = countries.features.find((f) => f.id === num) || null;
        }
        const d = OVERLOOKED.find((x) => x.iso === iso);
        if (panel && d) {
            panel.innerHTML = panelHTML(d);
        }
        render();
    }

    function panelHTML(d) {
        const cerf = d.cerf
            ? `<span class="panel-chip chip-on">CERF UFE pick (2024–25)</span>`
            : `<span class="panel-chip chip-off">Not picked by CERF UFE</span>`;
        return `
            <div class="panel-eyebrow">Selected country</div>
            <div class="panel-name">${d.name}</div>
            <div class="panel-iso">${d.iso}</div>

            <div class="panel-grid">
                <div class="panel-metric">
                    <div class="panel-metric-label">Posterior rank</div>
                    <div class="panel-metric-value">${d.rank}<span style="font-size: 0.55em; color: var(--muted); font-weight: 400; margin-left: 4px;">of ${POOL_SIZE}</span></div>
                </div>
                <div class="panel-metric">
                    <div class="panel-metric-label">90 % CI width</div>
                    <div class="panel-metric-value">${d.ci_width.toFixed(2)}</div>
                </div>
            </div>

            <div class="panel-type">${d.type}</div>
            <div class="panel-chips">${cerf}</div>

            <p class="panel-note">
                Position in the 2025-cycle Bayesian posterior across ${POOL_SIZE} HRP-eligible
                countries (those with an active humanitarian response plan). The 90 % CI width
                on the latent says how confidently the model places this crisis — wide bands
                mean the data alone cannot reliably distinguish this country from its neighbours
                in the ranking.
            </p>
        `;
    }

    // ─── Tooltip (hover) ───────────────────────────────────────────────
    const tt = document.getElementById("globe-tooltip");
    function tooltipShow(ev, d) {
        if (!tt) return;
        tt.textContent = `${d.name} · rank ${d.rank} of ${POOL_SIZE}`;
        tt.style.opacity = "1";
        tooltipMove(ev);
    }
    function tooltipMove(ev) {
        if (!tt) return;
        const rect = container.getBoundingClientRect();
        tt.style.left = (ev.clientX - rect.left + 12) + "px";
        tt.style.top  = (ev.clientY - rect.top  - 8)  + "px";
    }
    function tooltipHide() {
        if (!tt) return;
        tt.style.opacity = "0";
    }
})();
