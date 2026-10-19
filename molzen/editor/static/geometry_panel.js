/* Live geometry readout for one to four selected atoms.
   The formulas match molzen.geometry.describe_geometry. */
(function (root) {
  const MIN_LENGTH = 1e-6;
  const STYLE_ID = "molzen-geometry-style";
  const STYLE = `
.mz-geometry { box-sizing: border-box; padding: 8px 10px; border-top: 1px solid #dde2e6; background: #f7faf8; font: 13px system-ui, sans-serif; color: #263238; }
.mz-geometry-kicker { font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase; color: #526570; }
.mz-geometry-atoms { margin-top: 2px; font-weight: 600; color: #1d7a48; }
.mz-geometry-row { display: flex; justify-content: space-between; gap: 16px; margin-top: 2px; font-variant-numeric: tabular-nums; }
.mz-geometry-row span:first-child { color: #526570; }
.mz-geometry-empty, .mz-geometry-undefined { margin: 2px 0 0; color: #526570; }
`;

  function subtract(a, b) {
    return [a.x - b.x, a.y - b.y, a.z - b.z];
  }

  function dot(u, v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  }

  function cross(u, v) {
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  }

  function norm(vector) {
    return Math.hypot(vector[0], vector[1], vector[2]);
  }

  function finitePoint(point) {
    return point && [point.x, point.y, point.z].every(Number.isFinite);
  }

  function formatNumber(value, digits) {
    const text = value.toFixed(digits);
    const negativeZero = `-0.${"0".repeat(digits)}`;
    return text === negativeZero ? text.slice(1) : text;
  }

  function describe(points) {
    if (!points.length) return { kind: "empty" };
    if (points.length > 4 || points.some(point => !finitePoint(point))) return { kind: "undefined" };
    if (points.length === 1) {
      return { kind: "coordinates", value: [points[0].x, points[0].y, points[0].z] };
    }
    if (points.length === 2) {
      const separation = norm(subtract(points[1], points[0]));
      return separation < MIN_LENGTH ? { kind: "undefined" } : { kind: "distance", value: separation };
    }
    if (points.length === 3) {
      const towardA = subtract(points[0], points[1]);
      const towardC = subtract(points[2], points[1]);
      const lengthA = norm(towardA);
      const lengthC = norm(towardC);
      if (lengthA < MIN_LENGTH || lengthC < MIN_LENGTH) return { kind: "undefined" };
      const cosine = Math.min(1, Math.max(-1, dot(towardA, towardC) / (lengthA * lengthC)));
      return { kind: "angle", value: Math.acos(cosine) * 180 / Math.PI };
    }
    const towardA = subtract(points[0], points[1]);
    const axis = subtract(points[2], points[1]);
    const towardD = subtract(points[3], points[2]);
    const axisLength = norm(axis);
    if (axisLength < MIN_LENGTH) return { kind: "undefined" };
    const unitAxis = axis.map(component => component / axisLength);
    const planeA = towardA.map((component, index) => component - dot(towardA, unitAxis) * unitAxis[index]);
    const planeD = towardD.map((component, index) => component - dot(towardD, unitAxis) * unitAxis[index]);
    if (norm(planeA) < MIN_LENGTH || norm(planeD) < MIN_LENGTH) return { kind: "undefined" };
    const sine = dot(cross(unitAxis, planeA), planeD);
    return { kind: "dihedral", value: Math.atan2(sine, dot(planeA, planeD)) * 180 / Math.PI };
  }

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = STYLE;
    document.head.appendChild(style);
  }

  function addRow(container, label, value) {
    const row = document.createElement("div");
    row.className = "mz-geometry-row";
    const name = document.createElement("span");
    name.textContent = label;
    const reading = document.createElement("span");
    reading.textContent = value;
    row.append(name, reading);
    container.appendChild(row);
  }

  function render(container, atoms) {
    if (!container) return;
    ensureStyle();
    const points = Array.isArray(atoms) ? atoms.slice(0, 4) : [];
    const described = describe(points);
    container.replaceChildren();
    const kicker = document.createElement("div");
    kicker.className = "mz-geometry-kicker";
    kicker.textContent = "Geometry";
    container.appendChild(kicker);
    if (described.kind === "empty") {
      const empty = document.createElement("p");
      empty.className = "mz-geometry-empty";
      empty.textContent = "Select 1–4 atoms.";
      container.appendChild(empty);
      return;
    }
    const sequence = document.createElement("div");
    sequence.className = "mz-geometry-atoms";
    sequence.textContent = points.map(point => point.label || "atom").join(" – ");
    container.appendChild(sequence);
    if (described.kind === "undefined") {
      const undefinedReading = document.createElement("p");
      undefinedReading.className = "mz-geometry-undefined";
      undefinedReading.textContent = "Undefined for these atoms.";
      container.appendChild(undefinedReading);
      return;
    }
    if (described.kind === "coordinates") {
      ["x", "y", "z"].forEach((axis, index) => addRow(container, axis, `${formatNumber(described.value[index], 3)} Å`));
      return;
    }
    const digits = described.kind === "distance" ? 3 : 2;
    const unit = described.kind === "distance" ? "Å" : "°";
    addRow(container, described.kind, `${formatNumber(described.value, digits)} ${unit}`);
  }

  root.molzenGeometry = { maxAtoms: 4, describe, render };
})(typeof globalThis !== "undefined" ? globalThis : window);
