/* ==========================================================================
   FaceSearch — local index
   Frontend for the local Flask API. No frameworks, no external requests.
   ========================================================================== */

(function () {
  "use strict";

  const API = {
    persons: "/api/persons",
    person: (id) => `/api/persons/${encodeURIComponent(id)}`,
    recognize: "/api/recognize",
    photo: (id) => `/photo/${encodeURIComponent(id)}`,
  };

  const MATCH_THRESHOLD = 0.3; // mirrors the server-side threshold
  const MAX_SCAN_BYTES = 25 * 1024 * 1024;

  // ------------------------------------------------------------- utilities
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function h(tag, props, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "dataset") Object.assign(node.dataset, value);
      else if (key === "style") node.setAttribute("style", value);
      else if (key.startsWith("on") && typeof value === "function") {
        node.addEventListener(key.slice(2).toLowerCase(), value);
      } else if (key === "html") node.innerHTML = value;
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function svgIcon(path, viewBox = "0 0 16 16") {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", viewBox);
    svg.setAttribute("fill", "none");
    svg.setAttribute("aria-hidden", "true");
    svg.classList.add("chev");
    const p = document.createElementNS(ns, "path");
    p.setAttribute("d", path);
    p.setAttribute("stroke", "currentColor");
    p.setAttribute("stroke-width", "1.5");
    p.setAttribute("stroke-linecap", "round");
    p.setAttribute("stroke-linejoin", "round");
    svg.append(p);
    return svg;
  }

  const collator = new Intl.Collator("pl", { sensitivity: "base", numeric: true });
  const compare = (a, b) => collator.compare((a || "").trim(), (b || "").trim());

  function plural(count, singular, pluralForm) {
    return count === 1 ? singular : pluralForm;
  }

  function sum(list, key) {
    return list.reduce((total, item) => total + (Number(item[key]) || 0), 0);
  }

  function fullName(info) {
    const value = [info && info.name, info && info.surname]
      .map((part) => (part || "").trim())
      .filter(Boolean)
      .join(" ");
    return value || "Unnamed record";
  }

  function initials(name, surname) {
    const given = (name || "").trim();
    const family = (surname || "").trim();
    if (given && family) return (given[0] + family[0]).toUpperCase();
    const single = given || family;
    return (single ? single.slice(0, 2) : "?").toUpperCase();
  }

  function hueFor(seed) {
    let hash = 0;
    const text = String(seed || "");
    for (let i = 0; i < text.length; i += 1) {
      hash = (hash * 31 + text.charCodeAt(i)) % 360;
    }
    return hash;
  }

  function shortId(id) {
    return String(id || "").slice(0, 8);
  }

  function formatDate(value) {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value || ""));
    if (!match) return String(value || "");
    const date = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])));
    return date.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
  }

  function addressText(value) {
    if (value === null || value === undefined || value === "") return "";
    if (typeof value === "string") return value;
    if (Array.isArray(value)) return value.map(addressText).filter(Boolean).join(", ");
    if (typeof value === "object") {
      return ["street", "line1", "line2", "city", "postal_code", "zip", "state", "country"]
        .map((key) => value[key])
        .filter(Boolean)
        .join(", ");
    }
    return String(value);
  }

  function asArray(value) {
    if (value === null || value === undefined) return [];
    return Array.isArray(value) ? value : [value];
  }

  function phoneText(item) {
    if (item === null || item === undefined) return "";
    if (typeof item === "string") return item;
    return String(item.number || item.phone || item.value || "");
  }

  function emailText(item) {
    if (item === null || item === undefined) return "";
    if (typeof item === "string") return item;
    return String(item.address || item.email || item.value || "");
  }

  function displayValue(value) {
    if (value === null || value === undefined || value === "") return "";
    if (typeof value === "string") return value;
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    if (Array.isArray(value)) return value.map(displayValue).filter(Boolean).join(", ");
    return Object.entries(value)
      .filter(([, v]) => v !== null && v !== undefined && v !== "")
      .map(([k, v]) => `${k}: ${displayValue(v)}`)
      .join(", ");
  }

  function isUrl(value) {
    return /^https?:\/\//i.test(String(value || ""));
  }

  function humanKey(key) {
    return String(key)
      .replace(/[_-]+/g, " ")
      .replace(/^\w/, (c) => c.toUpperCase());
  }

  // ------------------------------------------------------------------ state
  const dom = {
    tabs: $$(".tab"),
    views: $$(".view"),
    navCount: $("#navCount"),
    search: $("#search"),
    searchClear: $("#searchClear"),
    searchTrigger: $("#searchTrigger"),
    sort: $("#sort"),
    refresh: $("#refresh"),
    resultMeta: $("#resultMeta"),
    resultHint: $("#resultHint"),
    grid: $("#grid"),
    statPeople: $("#statPeople"),
    statPrints: $("#statPrints"),
    statLinked: $("#statLinked"),
    themeToggle: $("#themeToggle"),
    drawer: $("#drawer"),
    drawerBody: $("#drawerBody"),
    drawerClose: $("#drawerClose"),
    drawerKicker: $("#drawerKicker"),
    scrim: $("#scrim"),
    toast: $("#toast"),
    dropZone: $("#dropZone"),
    fileInput: $("#fileInput"),
    preview: $("#preview"),
    scanStage: $("#scanStage"),
    scanImage: $("#scanImage"),
    scanCanvas: $("#scanCanvas"),
    scanStatus: $("#scanStatus"),
    scanStatusText: $("#scanStatusText"),
    scanFile: $("#scanFile"),
    scanCount: $("#scanCount"),
    scanResults: $("#scanResults"),
    scanReset: $("#scanReset"),
  };

  const state = {
    persons: [],
    query: "",
    sort: "surname",
    view: "index",
    loading: true,
    error: null,
    selectedId: null,
    lastFocused: null,
    scan: null,
    scanning: false,
  };

  const detailCache = new Map();
  let toastTimer = null;

  // ------------------------------------------------------------------ toast
  function toast(message) {
    if (!dom.toast) return;
    dom.toast.textContent = message;
    dom.toast.classList.add("is-visible");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => dom.toast.classList.remove("is-visible"), 3600);
  }

  // ------------------------------------------------------------------ theme
  function activeTheme() {
    return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  }

  function applyTheme(theme, persist) {
    document.documentElement.setAttribute("data-theme", theme);
    if (persist) {
      try { localStorage.setItem("fs-theme", theme); } catch (error) { /* storage may be blocked */ }
    }
    dom.themeToggle.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
    drawScanBoxes();
  }

  dom.themeToggle.addEventListener("click", () => {
    applyTheme(activeTheme() === "dark" ? "light" : "dark", true);
  });

  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");
  const onSchemeChange = (event) => {
    let stored = null;
    try { stored = localStorage.getItem("fs-theme"); } catch (error) { /* ignore */ }
    if (!stored) applyTheme(event.matches ? "dark" : "light", false);
  };
  if (typeof prefersDark.addEventListener === "function") prefersDark.addEventListener("change", onSchemeChange);
  else if (typeof prefersDark.addListener === "function") prefersDark.addListener(onSchemeChange);
  applyTheme(activeTheme(), false);

  // ------------------------------------------------------------------ views
  function setView(name, options) {
    if (!["index", "scan", "about"].includes(name)) name = "index";
    state.view = name;
    for (const tab of dom.tabs) {
      const active = tab.dataset.view === name;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    }
    for (const view of dom.views) {
      const active = view.id === `view-${name}`;
      view.hidden = !active;
      view.classList.toggle("is-active", active);
    }
    window.scrollTo(0, 0);
    if (name === "index" && options && options.focusSearch) focusSearch();
  }

  dom.tabs.forEach((tab) => tab.addEventListener("click", () => setView(tab.dataset.view)));
  $$("[data-nav]").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      setView(link.dataset.nav);
    });
  });

  // ----------------------------------------------------------------- search
  function focusSearch() {
    setView("index");
    dom.search.focus({ preventScroll: false });
    dom.search.select();
  }

  function syncSearchClear() {
    dom.searchClear.hidden = dom.search.value.length === 0;
  }

  dom.search.addEventListener("input", () => {
    state.query = dom.search.value.trim();
    syncSearchClear();
    renderPeople();
  });

  dom.search.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.stopPropagation();
      dom.search.value = "";
      state.query = "";
      syncSearchClear();
      renderPeople();
      dom.search.blur();
    }
  });

  dom.searchClear.addEventListener("click", () => {
    dom.search.value = "";
    state.query = "";
    syncSearchClear();
    renderPeople();
    dom.search.focus();
  });

  dom.searchTrigger.addEventListener("click", focusSearch);

  dom.sort.addEventListener("change", () => {
    state.sort = dom.sort.value;
    renderPeople();
  });

  dom.refresh.addEventListener("click", () => {
    loadPersons({ force: true, announce: true });
  });

  // ------------------------------------------------------------ data loading
  async function loadPersons(options) {
    const settings = options || {};
    if (settings.force) detailCache.clear();
    if (!state.persons.length) {
      state.loading = true;
      state.error = null;
      renderPeople();
    }
    dom.refresh.classList.add("is-spinning");
    try {
      const response = await fetch(API.persons, { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`The service replied with ${response.status}.`);
      const data = await response.json();
      state.persons = Array.isArray(data) ? data : [];
      state.loading = false;
      state.error = null;
      updateStats();
      renderPeople();
      if (settings.announce) toast(`Register reloaded — ${state.persons.length} ${plural(state.persons.length, "record", "records")}.`);
    } catch (error) {
      state.loading = false;
      state.error = error instanceof Error ? error.message : "Unknown error";
      if (!state.persons.length) renderPeople();
      toast("Could not reach the local service.");
    } finally {
      dom.refresh.classList.remove("is-spinning");
    }
  }

  function updateStats() {
    const people = state.persons.length;
    const prints = sum(state.persons, "fingerprint_count");
    const linked = sum(state.persons, "imported_count");
    dom.statPeople.textContent = people.toLocaleString();
    dom.statPrints.textContent = prints.toLocaleString();
    dom.statLinked.textContent = linked.toLocaleString();
    dom.navCount.textContent = people.toLocaleString();
  }

  // ------------------------------------------------------------------ grid
  const SORT_LABELS = {
    surname: "surname A–Z",
    name: "given name A–Z",
    prints: "most faceprints",
    linked: "most linked records",
  };

  const SORTERS = {
    surname: (a, b) => compare(a.surname, b.surname) || compare(a.name, b.name),
    name: (a, b) => compare(a.name, b.name) || compare(a.surname, b.surname),
    prints: (a, b) => (Number(b.fingerprint_count) || 0) - (Number(a.fingerprint_count) || 0) || compare(a.surname, b.surname),
    linked: (a, b) => (Number(b.imported_count) || 0) - (Number(a.imported_count) || 0) || compare(a.surname, b.surname),
  };

  function visiblePeople() {
    const tokens = state.query.toLowerCase().split(/\s+/).filter(Boolean);
    const filtered = state.persons.filter((person) => {
      if (!tokens.length) return true;
      const haystack = `${person.name || ""} ${person.surname || ""} ${person.secondname || ""}`.toLowerCase();
      return tokens.every((token) => haystack.includes(token));
    });
    const sorter = SORTERS[state.sort] || SORTERS.surname;
    return filtered.slice().sort(sorter);
  }

  function personCard(person, index) {
    const prints = Number(person.fingerprint_count) || 0;
    const linked = Number(person.imported_count) || 0;
    const name = (person.name || "").trim();
    const surname = (person.surname || "").trim();
    const hue = hueFor(person.person_id);

    const media = h("div", { class: "card-media" },
      h("span", { class: "card-mono", style: `--h:${hue}` }, initials(name, surname)),
      person.person_id
        ? h("img", {
            class: "card-photo",
            src: person.photo_url || API.photo(person.person_id),
            alt: "",
            loading: "lazy",
            decoding: "async",
            onload: function () { this.classList.add("is-loaded"); },
            onerror: function () { this.remove(); },
          })
        : null,
      h("span", { class: "media-no", text: String(index + 1).padStart(2, "0") }),
      prints ? h("span", { class: "media-badge", text: `${prints} ${plural(prints, "fp", "fp")}` }) : null
    );

    const metaBits = [];
    metaBits.push(h("span", { class: "id", text: shortId(person.person_id) }));
    metaBits.push(h("span", { class: "dot" }));
    metaBits.push(h("span", { text: `${prints} ${plural(prints, "print", "prints")}` }));
    if (linked) {
      metaBits.push(h("span", { class: "dot" }));
      metaBits.push(h("span", { text: `${linked} ${plural(linked, "link", "links")}` }));
    }

    const card = h("a", {
      class: "card",
      href: `#/person/${person.person_id}`,
      style: `--i:${Math.min(index, 20)}`,
      "aria-label": `Open record: ${fullName(person)}`,
      onclick: (event) => {
        event.preventDefault();
        openPerson(person.person_id);
      },
      onkeydown: (event) => {
        if (event.key === " " || event.key === "Enter") {
          event.preventDefault();
          openPerson(person.person_id);
        }
      },
      onpointerenter: function () {
        prefetchPerson(person.person_id);
      },
    },
      media,
      h("div", { class: "card-body" },
        h("span", { class: "card-name" },
          name ? h("span", { class: "given" }, name) : null,
          name && surname ? " " : null,
          surname ? h("span", { class: "family" }, surname) : (!name ? "Unnamed record" : null)
        ),
        h("p", { class: "card-meta" }, metaBits)
      )
    );

    return card;
  }

  function skeletonCard(index) {
    return h("div", { class: "card is-skeleton", style: `--i:${Math.min(index, 12)}` },
      h("div", { class: "card-media" }),
      h("div", { class: "card-body" },
        h("span", { class: "sk sk-name" }),
        h("span", { class: "sk sk-meta" })
      )
    );
  }

  function emptyState(title, text, action) {
    return h("div", { class: "empty" },
      h("p", { class: "empty-title", text: title }),
      h("p", { text }),
      action || null
    );
  }

  function renderPeople() {
    const list = visiblePeople();
    dom.grid.setAttribute("aria-busy", state.loading ? "true" : "false");

    if (state.loading && !state.persons.length) {
      dom.grid.replaceChildren(...Array.from({ length: 10 }, (_, i) => skeletonCard(i)));
      setResultMeta(null);
      return;
    }

    if (state.error && !state.persons.length) {
      dom.grid.replaceChildren(
        emptyState("The register is out of reach", state.error,
          h("button", {
            class: "btn primary",
            type: "button",
            onclick: () => loadPersons({ force: true }),
          }, "Try again")
        )
      );
      dom.resultMeta.textContent = "The local service did not answer.";
      dom.resultHint.textContent = "";
      return;
    }

    if (!state.persons.length) {
      dom.grid.replaceChildren(
        emptyState("The index is empty", "No person records were found on this machine yet. Add people with the recognition tools, then reload this page.")
      );
      setResultMeta(0);
      return;
    }

    if (!list.length) {
      dom.grid.replaceChildren(
        emptyState("Nothing matches", `No record contains “${state.query}”. Try a shorter fragment of a name.`,
          h("button", {
            class: "btn ghost",
            type: "button",
            onclick: () => { dom.search.value = ""; state.query = ""; syncSearchClear(); renderPeople(); dom.search.focus(); },
          }, "Clear filter")
        )
      );
      setResultMeta(0);
      return;
    }

    dom.grid.replaceChildren(...list.map((person, index) => personCard(person, index)));
    setResultMeta(list.length);
  }

  function setResultMeta(shown) {
    if (shown === null) {
      dom.resultMeta.textContent = "Opening the register…";
      dom.resultHint.textContent = "";
      return;
    }
    const total = state.persons.length;
    if (state.query) {
      dom.resultMeta.replaceChildren(
        h("strong", { text: String(shown) }),
        h("span", { text: ` ${plural(shown, "match", "matches")} of ${total} ${plural(total, "record", "records")}` })
      );
      dom.resultHint.textContent = `Filtering “${state.query}”`;
    } else {
      dom.resultMeta.replaceChildren(
        h("strong", { text: String(total) }),
        h("span", { text: ` ${plural(total, "record", "records")} in the index` })
      );
      dom.resultHint.textContent = `Sorted by ${SORT_LABELS[state.sort] || SORT_LABELS.surname}`;
    }
  }

  // ---------------------------------------------------------------- drawer
  function prefetchPerson(id) {
    if (!id || detailCache.has(id)) return;
    fetch(API.person(id), { headers: { Accept: "application/json" } })
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => { if (data) detailCache.set(id, data); })
      .catch(() => { /* prefetch is best effort */ });
  }

  async function getPerson(id) {
    if (detailCache.has(id)) return detailCache.get(id);
    const response = await fetch(API.person(id), { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(response.status === 404 ? "This record no longer exists." : `The service replied with ${response.status}.`);
    const data = await response.json();
    detailCache.set(id, data);
    return data;
  }

  function drawerLoading() {
    return h("div", { class: "drawer-loading" },
      h("div", { class: "dl-media sk" }),
      h("div", { class: "dl-line lg sk" }),
      h("div", { class: "dl-line sk" }),
      h("div", { class: "dl-line sm sk" }),
      h("div", { class: "dl-grid sk" })
    );
  }

  function openDrawer() {
    dom.drawer.classList.add("is-open");
    dom.scrim.classList.add("is-open");
    dom.drawer.setAttribute("aria-hidden", "false");
    dom.scrim.setAttribute("aria-hidden", "false");
    document.body.classList.add("is-locked");
    window.setTimeout(() => dom.drawerClose.focus({ preventScroll: true }), 60);
  }

  function closeDrawer(options) {
    if (!state.selectedId) return;
    state.selectedId = null;
    dom.drawer.classList.remove("is-open");
    dom.scrim.classList.remove("is-open");
    dom.drawer.setAttribute("aria-hidden", "true");
    dom.scrim.setAttribute("aria-hidden", "true");
    document.body.classList.remove("is-locked");
    if (!options || options.history !== false) {
      history.replaceState(null, "", window.location.pathname + window.location.search);
    }
    if ((!options || options.restore !== false) && state.lastFocused && state.lastFocused.isConnected) {
      state.lastFocused.focus({ preventScroll: true });
    }
    window.setTimeout(() => {
      if (!state.selectedId) dom.drawerBody.replaceChildren();
    }, 500);
  }

  dom.drawerClose.addEventListener("click", () => closeDrawer());
  dom.scrim.addEventListener("click", () => closeDrawer());

  async function openPerson(id) {
    if (!id) return;
    const same = state.selectedId === id;
    state.selectedId = id;
    state.lastFocused = document.activeElement;
    dom.drawerKicker.textContent = `Record ${shortId(id)}`;
    openDrawer();
    if (!same) {
      dom.drawerBody.replaceChildren(drawerLoading());
      dom.drawerBody.scrollTop = 0;
      history.replaceState(null, "", `#/person/${id}`);
    }
    try {
      const data = await getPerson(id);
      if (state.selectedId !== id) return;
      renderDetail(data);
    } catch (error) {
      if (state.selectedId !== id) return;
      dom.drawerBody.replaceChildren(
        h("div", { class: "drawer-error" },
          h("p", { text: error instanceof Error ? error.message : "Could not load this record." }),
          h("button", { class: "btn ghost small", type: "button", onclick: () => { detailCache.delete(id); openPerson(id); } }, "Retry")
        )
      );
    }
  }

  function mediaBlock(photoUrl, info) {
    const given = (info.name || "").trim();
    const family = (info.surname || "").trim();
    return h("div", { class: "detail-media" },
      h("span", { class: "card-mono", style: `--h:${hueFor(info.person_id || fullName(info))}` }, initials(given, family)),
      photoUrl
        ? h("img", {
            src: photoUrl,
            alt: `Portrait of ${fullName(info)}`,
            decoding: "async",
            onload: function () { this.classList.add("is-loaded"); },
            onerror: function () { this.remove(); },
          })
        : null
    );
  }

  function collectContacts(records) {
    const phones = new Set();
    const emails = new Set();
    const birthdays = new Set();
    const addresses = new Set();
    const aliases = new Set();
    const profiles = new Map();

    for (const record of records) {
      if (!record || typeof record !== "object") continue;
      if (record.alias) aliases.add(String(record.alias));
      if (record.birthday) birthdays.add(String(record.birthday));
      const address = addressText(record.address);
      if (address) addresses.add(address);
      for (const phone of asArray(record.phones)) {
        const value = phoneText(phone);
        if (value) phones.add(value);
      }
      for (const email of asArray(record.emails)) {
        const value = emailText(email);
        if (value) emails.add(value);
      }
      const custom = record.custom && typeof record.custom === "object" ? record.custom : {};
      for (const [key, value] of Object.entries(custom)) {
        const text = displayValue(value);
        if (!text) continue;
        if (isUrl(text)) profiles.set(humanKey(key), text);
      }
    }

    return {
      phones: Array.from(phones),
      emails: Array.from(emails),
      birthdays: Array.from(birthdays),
      addresses: Array.from(addresses),
      aliases: Array.from(aliases),
      profiles: Array.from(profiles.entries()),
    };
  }

  function fact(label, value, options) {
    if (value === null || value === undefined || value === "") return null;
    const opts = options || {};
    return h("div", { class: `fact${opts.full ? " full" : ""}` },
      h("dt", { text: label }),
      h("dd", {}, value)
    );
  }

  function linkList(values, hrefFor, labelFor) {
    return h("span", { class: "tag-row" },
      values.map((value) => h("a", {
        class: "tag",
        href: hrefFor(value),
        text: labelFor ? labelFor(value) : value,
      }))
    );
  }

  function recordBody(record) {
    const rows = [];
    const skip = new Set(["phones", "emails", "custom", "photo", "address", "birthday"]);

    const order = ["id", "name", "full_name", "alias"];
    for (const key of order) {
      const value = displayValue(record[key]);
      if (!value) continue;
      rows.push(h("dt", { text: humanKey(key) }), h("dd", { text: value }));
    }

    if (record.birthday) {
      rows.push(h("dt", { text: "Birthday" }), h("dd", { text: formatDate(record.birthday) }));
    }

    const address = addressText(record.address);
    if (address) rows.push(h("dt", { text: "Address" }), h("dd", { text: address }));

    const phones = asArray(record.phones).map(phoneText).filter(Boolean);
    if (phones.length) {
      rows.push(h("dt", { text: "Phones" }), h("dd", {}, linkList(phones, (v) => `tel:${v.replace(/\s+/g, "")}`)));
    }

    const emails = asArray(record.emails).map(emailText).filter(Boolean);
    if (emails.length) {
      rows.push(h("dt", { text: "Emails" }), h("dd", {}, linkList(emails, (v) => `mailto:${v}`)));
    }

    const custom = record.custom && typeof record.custom === "object" ? record.custom : {};
    for (const [key, value] of Object.entries(custom)) {
      const text = displayValue(value);
      if (!text) continue;
      rows.push(
        h("dt", { text: humanKey(key) }),
        h("dd", {}, isUrl(text) ? h("a", { href: text, target: "_blank", rel: "noopener noreferrer", text }) : text)
      );
    }

    for (const [key, value] of Object.entries(record)) {
      if (skip.has(key) || order.includes(key)) continue;
      if (key === "birthday" || key === "address") continue;
      const text = displayValue(value);
      if (!text) continue;
      rows.push(h("dt", { text: humanKey(key) }), h("dd", { text }));
    }

    if (!rows.length) {
      return h("div", { class: "record-body" }, h("p", { class: "panel-note", text: "This source record is empty." }));
    }
    return h("div", { class: "record-body" }, h("dl", { class: "kv" }, rows));
  }

  function linkedRecords(records) {
    if (!records.length) {
      return h("p", { class: "panel-note", text: "No imported records are linked to this person yet." });
    }
    return h("div", {},
      records.map((record, index) => {
        const source = record && record.id !== undefined ? `#${record.id}` : `#${index + 1}`;
        const name = displayValue(record.full_name || record.name) || "Unnamed record";
        const details = h("details", { class: "record" },
          h("summary", {},
            h("span", { class: "source", text: source }),
            h("span", { class: "summary-name", text: name }),
            svgIcon("m3.5 5.5 3.5 3.5 3.5-3.5", "0 0 14 14")
          ),
          recordBody(record || {})
        );
        if (records.length === 1) details.open = true;
        return details;
      })
    );
  }

  function renderDetail(data) {
    const info = data && data.info && typeof data.info === "object" ? data.info : {};
    const imported = Array.isArray(info.imported_from_input) ? info.imported_from_input : [];
    const fingerprints = Array.isArray(data.fingerprints) ? data.fingerprints : [];
    const contacts = collectContacts(imported);
    const given = (info.name || "").trim();
    const family = (info.surname || "").trim();
    const alias = (info.secondname || "").trim() || contacts.aliases[0] || "";

    dom.drawerKicker.textContent = `Record ${shortId(data.person_id)}`;

    const chips = [
      h("span", { class: `chip${fingerprints.length ? " is-accent" : ""}` },
        h("b", { text: String(fingerprints.length) }),
        ` ${plural(fingerprints.length, "faceprint", "faceprints")}`
      ),
    ];
    if (imported.length) {
      chips.push(h("span", { class: "chip" },
        h("b", { text: String(imported.length) }),
        ` linked ${plural(imported.length, "record", "records")}`
      ));
    }
    if (contacts.birthdays.length) {
      chips.push(h("span", { class: "chip" }, `Born ${formatDate(contacts.birthdays[0])}`));
    }

    const facts = [
      fact("Record ID", h("span", { class: "mono", text: data.person_id }), { full: true }),
      fact("Birthday", contacts.birthdays.map(formatDate).join(" · ")),
      fact("Phone", contacts.phones.length
        ? linkList(contacts.phones, (v) => `tel:${v.replace(/\s+/g, "")}`)
        : ""),
      fact("Email", contacts.emails.length
        ? linkList(contacts.emails, (v) => `mailto:${v}`)
        : ""),
      fact("Address", contacts.addresses.join(" · "), { full: true }),
      fact("Alias", contacts.aliases.join(" · ")),
    ].filter(Boolean);

    if (fingerprints.length) {
      facts.push(fact("Fingerprint files",
        h("span", { class: "tag-row" }, fingerprints.map((fp) =>
          h("a", {
            class: "tag",
            href: fp.url,
            download: `${fp.index}.npy`,
            title: "Download embedding",
            text: `#${fp.index}`,
          })
        )),
        { full: true }
      ));
    }

    const content = h("div", { class: "drawer-content" },
      mediaBlock(data.photo_url || (data.person_id ? API.photo(data.person_id) : null), info),
      h("h2", { class: "detail-title", id: "drawerTitle" },
        given ? h("span", { class: "given" }, given) : null,
        given && family ? " " : null,
        family ? h("span", { class: "family" }, family) : null,
        !given && !family ? "Unnamed record" : null
      ),
      alias ? h("p", { class: "detail-alias" }, h("span", { class: "label", text: "Also known as" }), alias) : null,
      h("div", { class: "chips" }, chips),
      contacts.profiles.length
        ? h("div", { class: "chips" }, contacts.profiles.map(([label, url]) =>
            h("a", { class: "tag", href: url, target: "_blank", rel: "noopener noreferrer", text: label })
          ))
        : null,
      h("dl", { class: "facts" }, facts),
      h("section", { class: "section" },
        h("div", { class: "section-head" },
          h("h3", { text: "Linked records" }),
          h("span", { text: String(imported.length) })
        ),
        linkedRecords(imported)
      ),
      h("div", { class: "drawer-foot" },
        h("span", { class: "record-id", text: data.person_id }),
        h("button", {
          class: "btn ghost small",
          type: "button",
          onclick: async () => {
            const ok = await copyText(data.person_id);
            toast(ok ? "Record ID copied." : "Could not copy the ID.");
          },
        }, "Copy ID")
      )
    );

    dom.drawerBody.replaceChildren(content);
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (error) {
      try {
        const field = h("textarea", { style: "position:fixed;opacity:0;pointer-events:none" });
        field.value = text;
        document.body.append(field);
        field.select();
        const ok = document.execCommand("copy");
        field.remove();
        return ok;
      } catch (fallbackError) {
        return false;
      }
    }
  }

  // ------------------------------------------------------------------ scan
  function setScanState(kind, text) {
    dom.preview.classList.remove("is-scanning", "is-ready", "is-error");
    if (kind) dom.preview.classList.add(`is-${kind}`);
    dom.scanStatusText.textContent = text;
    dom.preview.classList.toggle("is-scanning", kind === "scanning");
  }

  function emptyMatches(title, text) {
    return h("div", { class: "panel-empty" },
      h("p", { class: "panel-empty-title", text: title }),
      h("p", { text })
    );
  }

  function resetScan() {
    if (state.scan && state.scan.url) URL.revokeObjectURL(state.scan.url);
    state.scan = null;
    state.scanning = false;
    dom.preview.hidden = true;
    dom.preview.classList.remove("is-scanning", "is-ready", "is-error");
    dom.dropZone.hidden = false;
    dom.scanImage.removeAttribute("src");
    dom.scanFile.textContent = "";
    const context = dom.scanCanvas.getContext("2d");
    context.clearRect(0, 0, dom.scanCanvas.width || 0, dom.scanCanvas.height || 0);
    dom.scanCanvas.width = 0;
    dom.scanCanvas.height = 0;
    dom.scanCount.textContent = "—";
    dom.scanCount.classList.remove("is-hot");
    dom.scanResults.replaceChildren(emptyMatches("Nothing scanned yet", "Results for every face in the image will appear here."));
    dom.fileInput.value = "";
    if (state.view === "scan") dom.dropZone.focus({ preventScroll: true });
  }

  function acceptFile(file) {
    if (!file) return;
    const looksLikeImage = file.type ? file.type.startsWith("image/") : /\.(jpe?g|png|webp|gif|bmp|avif)$/i.test(file.name || "");
    if (!looksLikeImage) {
      toast("That file is not an image.");
      return;
    }
    if (file.size > MAX_SCAN_BYTES) {
      toast("That image is larger than 25 MB.");
      return;
    }
    runRecognition(file);
  }

  async function runRecognition(file) {
    if (state.scanning) return;
    state.scanning = true;

    if (state.scan && state.scan.url) URL.revokeObjectURL(state.scan.url);
    const url = URL.createObjectURL(file);
    state.scan = { file, url, results: null };

    dom.dropZone.hidden = true;
    dom.preview.hidden = false;
    dom.preview.classList.remove("is-ready", "is-error");
    dom.scanImage.src = url;
    dom.scanImage.alt = `Photograph being scanned: ${file.name}`;
    dom.scanFile.textContent = file.name;
    dom.scanCanvas.width = 0;
    dom.scanCanvas.height = 0;
    dom.scanCount.textContent = "…";
    dom.scanCount.classList.remove("is-hot");
    dom.scanResults.replaceChildren(emptyMatches("Analyzing", "Looking for faces and comparing them with the index."));
    setScanState("scanning", "Analyzing the photograph…");

    const form = new FormData();
    form.append("image", file);

    try {
      const response = await fetch(API.recognize, { method: "POST", body: form });
      let data = null;
      try { data = await response.json(); } catch (error) { data = null; }
      if (!response.ok) {
        throw new Error((data && data.error) || `The service replied with ${response.status}.`);
      }
      if (state.selectedId && state.scan && state.scan.file !== file) return;
      if (!state.scan || state.scan.file !== file) {
        URL.revokeObjectURL(url);
        return;
      }
      const results = Array.isArray(data && data.results) ? data.results : [];
      state.scan.results = results;
      renderScanResults(results);
    } catch (error) {
      if (!state.scan || state.scan.file !== file) return;
      state.scan.results = [];
      const message = error instanceof Error ? error.message : "The scan failed.";
      setScanState("error", message);
      dom.scanResults.replaceChildren(emptyMatches("Scan failed", message));
      toast(message);
      dom.scanCount.textContent = "—";
    } finally {
      state.scanning = false;
    }
  }

  function renderScanResults(results) {
    const recognized = results.filter((item) => item.person).length;
    dom.scanCount.textContent = String(results.length);
    dom.scanCount.classList.toggle("is-hot", recognized > 0);

    if (!results.length) {
      setScanState("ready", "No faces were found in this image.");
    } else {
      setScanState(
        "ready",
        `${results.length} ${plural(results.length, "face", "faces")} found · ${recognized} ${plural(recognized, "match", "matches")}`
      );
    }

    const list = dom.scanResults;
    list.replaceChildren();

    if (!results.length) {
      list.append(emptyMatches("No faces detected", "Try a sharper, front-facing photograph."));
      drawScanBoxes();
      return;
    }

    const sorted = results.slice().sort((a, b) => {
      const known = (b.person ? 1 : 0) - (a.person ? 1 : 0);
      if (known) return known;
      return (Number(b.similarity) || 0) - (Number(a.similarity) || 0);
    });

    for (const result of sorted) {
      const person = result.person;
      const similarity = result.similarity === null || result.similarity === undefined
        ? null
        : Math.max(0, Math.min(1, Number(result.similarity)));
      const percent = similarity === null ? "—" : `${Math.round(similarity * 100)}%`;

      const avatar = h("span", { class: "match-avatar" },
        person ? initials(person.name, person.surname) : "?",
        person
          ? h("img", {
              src: API.photo(person.person_id),
              alt: "",
              loading: "lazy",
              onerror: function () { this.remove(); },
            })
          : null
      );

      const name = person
        ? fullName(person)
        : "Unknown face";

      let sub;
      if (person) sub = "Matched record";
      else if (similarity === null) sub = "No database entries";
      else sub = similarity >= MATCH_THRESHOLD ? "Matched record" : "Below match threshold";

      const score = h("span", { class: "match-score" },
        h("span", { class: "match-percent", text: percent }),
        h("span", { class: "match-bar" },
          h("i", { style: `width:${similarity === null ? 0 : Math.max(4, similarity * 100)}%` })
        )
      );

      if (person) {
        list.append(h("button", {
          class: "match",
          type: "button",
          title: `${name} — ${percent} similarity`,
          onclick: () => openPerson(person.person_id),
        },
          avatar,
          h("span", { class: "match-main" },
            h("span", { class: "match-name", text: name }),
            h("span", { class: "match-sub", text: sub })
          ),
          score
        ));
      } else {
        list.append(h("div", { class: "match is-unknown" },
          avatar,
          h("span", { class: "match-main" },
            h("span", { class: "match-name", text: name }),
            h("span", { class: "match-sub", text: sub })
          ),
          score
        ));
      }
    }

    drawScanBoxes();
  }

  function roundedRect(context, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    context.beginPath();
    context.moveTo(x + r, y);
    context.lineTo(x + width - r, y);
    context.quadraticCurveTo(x + width, y, x + width, y + r);
    context.lineTo(x + width, y + height - r);
    context.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    context.lineTo(x + r, y + height);
    context.quadraticCurveTo(x, y + height, x, y + height - r);
    context.lineTo(x, y + r);
    context.quadraticCurveTo(x, y, x + r, y);
    context.closePath();
  }

  function drawScanBoxes() {
    const image = dom.scanImage;
    const canvas = dom.scanCanvas;
    const results = state.scan && Array.isArray(state.scan.results) ? state.scan.results : null;
    if (!results || !results.length || !image.complete || !image.naturalWidth) {
      if (canvas.width) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const width = image.naturalWidth;
    const height = image.naturalHeight;
    canvas.width = width;
    canvas.height = height;

    const context = canvas.getContext("2d");
    context.clearRect(0, 0, width, height);

    const styles = getComputedStyle(document.documentElement);
    const accent = styles.getPropertyValue("--accent").trim() || "#b04a22";
    const muted = styles.getPropertyValue("--muted").trim() || "#7a746a";
    const labelInk = "#fbf7ef";

    const lineWidth = Math.max(2, width / 420);
    const fontSize = Math.max(14, Math.round(width * 0.026));
    const font = `600 ${fontSize}px ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;

    for (const result of results) {
      const box = Array.isArray(result.bbox) ? result.bbox : null;
      if (!box || box.length < 4) continue;
      const x1 = Number(box[0]);
      const y1 = Number(box[1]);
      const x2 = Number(box[2]);
      const y2 = Number(box[3]);
      if (![x1, y1, x2, y2].every(Number.isFinite)) continue;

      const person = result.person;
      const color = person ? accent : muted;
      const boxWidth = Math.max(1, x2 - x1);
      const boxHeight = Math.max(1, y2 - y1);
      const corner = Math.min(boxWidth, boxHeight) * 0.18;

      context.strokeStyle = color;
      context.lineWidth = lineWidth;
      context.lineCap = "round";
      context.beginPath();
      // top-left
      context.moveTo(x1, y1 + corner);
      context.lineTo(x1, y1);
      context.lineTo(x1 + corner, y1);
      // top-right
      context.moveTo(x2 - corner, y1);
      context.lineTo(x2, y1);
      context.lineTo(x2, y1 + corner);
      // bottom-right
      context.moveTo(x2, y2 - corner);
      context.lineTo(x2, y2);
      context.lineTo(x2 - corner, y2);
      // bottom-left
      context.moveTo(x1 + corner, y2);
      context.lineTo(x1, y2);
      context.lineTo(x1, y2 - corner);
      context.stroke();

      const similarity = result.similarity === null || result.similarity === undefined
        ? null
        : Math.max(0, Math.min(1, Number(result.similarity)));
      const labelParts = [];
      labelParts.push(person ? fullName(person) : "Unknown");
      if (similarity !== null) labelParts.push(`${Math.round(similarity * 100)}%`);
      const label = labelParts.join(" · ");

      context.font = font;
      const textWidth = context.measureText(label).width;
      const padX = fontSize * 0.55;
      const padY = fontSize * 0.4;
      const chipHeight = fontSize + padY * 2;
      const chipWidth = textWidth + padX * 2;

      let chipX = x1;
      if (chipX + chipWidth > width) chipX = Math.max(0, width - chipWidth);
      let chipY = y1 - chipHeight - lineWidth * 2;
      if (chipY < 0) chipY = y1 + lineWidth * 2;

      context.fillStyle = color;
      roundedRect(context, chipX, chipY, chipWidth, chipHeight, Math.min(10, chipHeight / 3));
      context.fill();

      context.fillStyle = labelInk;
      context.textBaseline = "middle";
      context.fillText(label, chipX + padX, chipY + chipHeight / 2 + fontSize * 0.03);
    }
  }

  dom.scanImage.addEventListener("load", () => {
    dom.scanCanvas.width = 0;
    dom.scanCanvas.height = 0;
    drawScanBoxes();
  });

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(drawScanBoxes, 140);
  });

  // drop zone + file input
  dom.dropZone.addEventListener("click", () => dom.fileInput.click());
  dom.dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      dom.fileInput.click();
    }
  });
  dom.fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) acceptFile(file);
  });

  // the whole stage accepts drops (including while a preview is shown)
  let dragDepth = 0;
  dom.scanStage.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dragDepth += 1;
    dom.dropZone.classList.add("is-dragover");
  });
  dom.scanStage.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  });
  dom.scanStage.addEventListener("dragleave", (event) => {
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) dom.dropZone.classList.remove("is-dragover");
  });
  dom.scanStage.addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    dom.dropZone.classList.remove("is-dragover");
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) acceptFile(file);
  });

  // never let a stray drop navigate the page away
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  document.addEventListener("paste", (event) => {
    if (state.view !== "scan") return;
    const items = event.clipboardData ? Array.from(event.clipboardData.items) : [];
    const imageItem = items.find((item) => item.kind === "file" && item.type.startsWith("image/"));
    if (!imageItem) return;
    const file = imageItem.getAsFile();
    if (file) {
      event.preventDefault();
      acceptFile(file);
    }
  });

  dom.scanReset.addEventListener("click", resetScan);

  // -------------------------------------------------------------- shortcuts
  document.addEventListener("keydown", (event) => {
    const target = event.target;
    const tag = target && target.tagName ? target.tagName.toLowerCase() : "";
    const typing = tag === "input" || tag === "textarea" || tag === "select" || (target && target.isContentEditable);

    if (event.key === "Escape") {
      if (state.selectedId) {
        event.preventDefault();
        closeDrawer();
        return;
      }
      if (state.view === "index" && (state.query || document.activeElement === dom.search)) {
        if (state.query) {
          dom.search.value = "";
          state.query = "";
          syncSearchClear();
          renderPeople();
        }
        dom.search.blur();
      }
      return;
    }

    if ((event.metaKey || event.ctrlKey) && !event.altKey && String(event.key).toLowerCase() === "k") {
      event.preventDefault();
      focusSearch();
      return;
    }

    if (event.key === "/" && !typing && !event.metaKey && !event.ctrlKey) {
      event.preventDefault();
      focusSearch();
    }
  });

  // open a deep-linked record (e.g. after a reload)
  function openFromHash() {
    const match = /^#\/person\/([A-Za-z0-9_-]{6,64})$/.exec(window.location.hash || "");
    if (match) openPerson(match[1]);
  }
  window.addEventListener("hashchange", () => {
    const match = /^#\/person\/([A-Za-z0-9_-]{6,64})$/.exec(window.location.hash || "");
    if (!match && state.selectedId) closeDrawer({ history: false });
  });

  // ------------------------------------------------------------------- boot
  syncSearchClear();
  renderPeople();
  loadPersons();
  openFromHash();
})();
