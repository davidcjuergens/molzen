/* The renderer is prepended from the pinned local 3Dmol distribution. */
const ATOM_STYLE = { stick: { radius: 0.16 }, sphere: { scale: 0.22 } };
const HALO_SCALE = 0.36;
const HALO_OPACITY = 0.32;
const HALO_COLOR = "#6fce96";
// Same radii as the bundled 3Dmol build, so the halo stays a fixed step larger than the drawn atom.
const VDW_RADII = { H: 1.2, He: 1.4, Li: 1.82, Be: 1.53, B: 1.92, C: 1.7, N: 1.55, O: 1.52, F: 1.47, Ne: 1.54, Na: 2.27, Mg: 1.73, Al: 1.84, Si: 2.1, P: 1.8, S: 1.8, Cl: 1.75, Ar: 1.88, K: 2.75, Ca: 2.31, Ni: 1.63, Cu: 1.4, Zn: 1.39, Ga: 1.87, Ge: 2.11, As: 1.85, Se: 1.9, Br: 1.85, Kr: 2.02, Rb: 3.03, Sr: 2.49, Pd: 1.63, Ag: 1.72, Cd: 1.58, In: 1.93, Sn: 2.17, Sb: 2.06, Te: 2.06, I: 1.98, Xe: 2.16, Cs: 3.43, Ba: 2.68, Pt: 1.75, Au: 1.66, Hg: 1.55, Tl: 1.96, Pb: 2.02, Bi: 2.07, Po: 1.97, At: 2.02, Rn: 2.2, Fr: 3.48, Ra: 2.83, U: 1.86 };

function vdwRadius(element) {
  if (VDW_RADII[element]) return VDW_RADII[element];
  const normalized = element.length > 1 ? element[0].toUpperCase() + element.slice(1).toLowerCase() : element;
  return VDW_RADII[normalized] || 1.5;
}

function render({ model, el }) {
  const root = document.createElement("div");
  root.className = "molzen-editor";
  root.style.width = model.get("width");
  root.innerHTML = `
    <div class="mz-toolbar">
      <button data-action="undo">Undo</button><button data-action="redo">Redo</button>
      <button data-action="reset-view">Center</button>
      <span class="mz-summary"></span>
    </div>
    <div class="mz-viewport"></div>
    <div class="mz-geometry" aria-live="polite"></div>
    <div class="mz-controls">
      <label>Selected atoms <select class="mz-atoms" multiple size="4" aria-label="Selected atoms"></select></label>
      <div>
        <div class="mz-selection">Click atoms to select up to four. Select two to edit a bond. Click empty space to clear.</div>
        <div class="mz-toolbar">
          <label>Element <select class="mz-element" aria-label="Element"></select></label>
          <button class="mz-accent-green" data-action="set_element">Change element</button>
          <button data-action="delete_atom">Delete atom</button>
        </div>
        <div class="mz-toolbar">
          <label>Bond <select class="mz-order" aria-label="Bond order">
            <option value="1">Single</option><option value="2">Double</option>
            <option value="3">Triple</option><option value="ar">Aromatic</option>
            <option value="unknown">Unknown</option>
          </select></label>
          <button class="mz-accent-blue" data-action="set_bond">Set bond</button>
          <button data-action="remove_bond">Remove bond</button>
        </div>
        <div class="mz-toolbar">
          <label>Torsion (°) <input class="mz-torsion" aria-label="Methyl torsion" type="number" value="0" step="15"></label>
          <button class="mz-accent-violet" data-action="replace_hydrogen">Replace H with methyl</button>
        </div>
      </div>
    </div>
    <div class="mz-toolbar">
      <button class="mz-accent-gold" data-action="suggest_bonds">Suggest bonds</button>
      <button class="mz-accent-green" data-action="accept_bonds">Accept suggestions</button>
      <span class="mz-preview"></span>
    </div>
    <div class="mz-status" role="status" aria-live="polite"></div>
    <div class="mz-diagnostics"></div>
  `;
  el.appendChild(root);
  const viewport = root.querySelector(".mz-viewport");
  viewport.style.height = model.get("height");
  const viewer = molzen3Dmol.createViewer(viewport, { backgroundColor: "white" });
  const atomList = root.querySelector(".mz-atoms");
  const elementSelect = root.querySelector(".mz-element");
  const orderSelect = root.querySelector(".mz-order");
  const status = root.querySelector(".mz-status");
  let selected = [];
  let pending = null;
  let firstDraw = true;
  let state = model.get("state");
  let pointerOrigin = null;
  let atomClicked = false;

  function paintShapes() {
    viewer.removeAllShapes();
    const byId = new Map(state.atoms.map(atom => [atom.id, atom]));
    for (const edge of state.suggestions) {
      const start = byId.get(edge.a);
      const end = byId.get(edge.b);
      if (start && end) viewer.addCylinder({ start, end, radius: 0.07, color: "#d29220", opacity: 0.5 });
    }
    for (const id of selected) {
      const atom = byId.get(id);
      if (!atom) continue;
      const shape = viewer.addSphere({
        center: { x: atom.x, y: atom.y, z: atom.z },
        radius: vdwRadius(atom.elem) * HALO_SCALE,
        color: HALO_COLOR,
        opacity: HALO_OPACITY,
      });
      // 3Dmol rebuilds shape materials on every render and writes depth, which would hide the atom inside the halo.
      const build = shape.globj.bind(shape);
      shape.globj = (group, extensions) => {
        const result = build(group, extensions);
        const stack = shape.renderedShapeObj ? [shape.renderedShapeObj] : [];
        while (stack.length) {
          const node = stack.pop();
          for (const material of node.material == null ? [] : [].concat(node.material)) {
            material.depthWrite = false;
            material.transparent = true;
            material.opacity = HALO_OPACITY;
          }
          if (node.children) stack.push(...node.children);
        }
        return result;
      };
    }
  }

  function updateSelection() {
    for (const option of atomList.options) option.selected = selected.includes(Number(option.value));
    viewer.setStyle({}, { stick: { ...ATOM_STYLE.stick }, sphere: { ...ATOM_STYLE.sphere } });
    for (const id of selected) {
      const atom = state.atoms.find(candidate => candidate.id === id);
      if (!atom) continue;
      viewer.setStyle({ index: state.atoms.indexOf(atom) }, {
        stick: { ...ATOM_STYLE.stick },
        sphere: { ...ATOM_STYLE.sphere },
        clicksphere: { radius: vdwRadius(atom.elem) * HALO_SCALE },
      });
    }
    paintShapes();
    const atom = state.atoms.find(atom => atom.id === selected[0]);
    if (atom) elementSelect.value = atom.elem;
    const edge = state.bonds.find(b => selected.includes(b.a) && selected.includes(b.b));
    if (edge) orderSelect.value = ["1", "2", "3", "ar"].includes(edge.order) ? edge.order : "unknown";
    const selectionLabel = root.querySelector(".mz-selection");
    selectionLabel.classList.toggle("mz-has-selection", selected.length > 0);
    selectionLabel.textContent = selected.length
      ? `Selected: ${selected.join(", ")}${edge ? ` · bond ${edge.order} (${edge.source})` : ""}`
      : "Click atoms to select up to four. Select two to edit a bond. Click empty space to clear.";
    const geometryAtoms = selected.flatMap(id => {
      const match = state.atoms.find(candidate => candidate.id === id);
      return match ? [{ label: `${match.elem} ${match.id}`, x: match.x, y: match.y, z: match.z }] : [];
    });
    globalThis.molzenGeometry.render(root.querySelector(".mz-geometry"), geometryAtoms);
    for (const button of root.querySelectorAll("button")) {
      const action = button.dataset.action;
      let disabled = Boolean(pending);
      if (["set_element", "delete_atom", "replace_hydrogen"].includes(action)) disabled ||= selected.length !== 1;
      if (["set_bond", "remove_bond"].includes(action)) disabled ||= selected.length !== 2;
      if (action === "replace_hydrogen") disabled ||= atom?.elem !== "H";
      if (action === "undo") disabled ||= !state.can_undo;
      if (action === "redo") disabled ||= !state.can_redo;
      if (action === "accept_bonds") disabled ||= !state.suggestions.length;
      button.disabled = disabled;
    }
    viewer.render();
  }

  function selectAtom(id) {
    atomClicked = true;
    if (pending) return;
    const kept = globalThis.molzenGeometry.maxAtoms - 1;
    selected = selected.includes(id) ? selected.filter(value => value !== id) : [...selected.slice(-kept), id];
    updateSelection();
  }

  function draw() {
    state = model.get("state");
    if (state.request === pending) pending = null;
    const view = firstDraw ? null : viewer.getView();
    viewer.removeAllModels();
    viewer.removeAllShapes();
    const atoms = state.atoms.map((atom, index) => ({ ...atom, index, serial: index + 1, bonds: [], bondOrder: [] }));
    const indices = new Map(atoms.map((atom, index) => [atom.id, index]));
    for (const edge of state.bonds) {
      const a = indices.get(edge.a), b = indices.get(edge.b);
      if (edge.order === "nc") continue;
      const order = edge.order === "ar" ? 1.5 : (["1", "2", "3"].includes(edge.order) ? Number(edge.order) : 1);
      atoms[a].bonds.push(b); atoms[a].bondOrder.push(order);
      atoms[b].bonds.push(a); atoms[b].bondOrder.push(order);
    }
    viewer.addModel().addAtoms(atoms);
    viewer.setClickable({}, true, atom => selectAtom(atom.id));
    viewer.setHoverable({}, true, atom => {
      atom.molzenLabel = viewer.addLabel(`${atom.elem} · ${atom.id}`, { position: atom, fontSize: 12, backgroundColor: "#263238" });
      viewer.render();
    }, atom => { if (atom.molzenLabel) viewer.removeLabel(atom.molzenLabel); viewer.render(); });
    viewer.removeAllLabels();
    atomList.replaceChildren(...state.atoms.map(atom => new Option(`${atom.id}: ${atom.elem} ${atom.name}`, atom.id)));
    if (!elementSelect.options.length) elementSelect.replaceChildren(...state.elements.map(e => new Option(e, e)));
    selected = selected.filter(id => indices.has(id));
    root.querySelector(".mz-summary").textContent = `${atoms.length} atoms · ${state.bonds.length} bonds · ${state.coverage} connectivity`;
    root.querySelector(".mz-preview").textContent = state.suggestions.length ? `${state.suggestions.length} candidate bonds shown in gold` : "";
    root.querySelector(".mz-diagnostics").textContent = state.diagnostics.join("\n");
    status.textContent = state.error || (pending ? "Applying…" : "Ready. Edits update a copy; retrieve it with editor.molecule.");
    status.classList.toggle("mz-error", Boolean(state.error));
    if (view) viewer.setView(view); else viewer.zoomTo();
    firstDraw = false;
    updateSelection();
  }

  // 3Dmol 2.5.4 only invokes the atom callback when a clickable is hit, so empty space is a viewport click.
  viewport.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    pointerOrigin = { x: event.clientX, y: event.clientY };
    atomClicked = false;
  });
  viewport.addEventListener("click", event => {
    if (atomClicked) {
      atomClicked = false;
      return;
    }
    if (pending || !pointerOrigin || !selected.length) return;
    const dx = event.clientX - pointerOrigin.x;
    const dy = event.clientY - pointerOrigin.y;
    if (dx * dx + dy * dy > 9) return;
    selected = [];
    updateSelection();
  });
  atomList.addEventListener("change", () => {
    selected = [...atomList.selectedOptions].slice(-globalThis.molzenGeometry.maxAtoms).map(option => Number(option.value));
    updateSelection();
  });
  root.addEventListener("click", event => {
    const action = event.target.closest("button")?.dataset.action;
    if (!action || pending) return;
    if (action === "reset-view") { viewer.zoomTo(); viewer.render(); return; }
    let args = {};
    if (action === "set_element") args = { atom_id: selected[0], element: elementSelect.value };
    if (action === "delete_atom") args = { atom_id: selected[0] };
    if (action === "replace_hydrogen") args = { atom_id: selected[0], torsion: Number(root.querySelector(".mz-torsion").value) };
    if (action === "set_bond") args = { a: selected[0], b: selected[1], order: orderSelect.value };
    if (action === "remove_bond") args = { a: selected[0], b: selected[1] };
    pending = crypto.randomUUID();
    status.textContent = "Applying…";
    updateSelection();
    model.send({ id: pending, revision: state.revision, action, args });
  });
  model.on("change:state", draw);
  const observer = new ResizeObserver(() => { viewer.resize(); viewer.render(); });
  observer.observe(viewport);
  draw();
  return () => {
    model.off("change:state", draw);
    observer.disconnect();
    viewer.clear();
    root.remove();
  };
}
export default { render };
