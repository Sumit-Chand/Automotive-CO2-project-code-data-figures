// Shared helpers for building the four Word reports with docx-js
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, Header, Footer,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, LevelFormat, PageNumber, PageBreak, TabStopType,
} = require("docx");

const NAVY = "1F4E79", BLUE = "2E86AB", LIGHT = "EAF1F8", GREY = "595959", ZEBRA = "F5F8FB", CODEBG = "F3F4F6";
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1304;           // A4, ~0.9 inch margins
const CONTENT_W = PAGE_W - 2 * MARGIN;                           // DXA
const FONT = "Calibri", MONO = "Consolas";

let figCounter = 0, tabCounter = 0, numCounter = 0;
const numberingConfigs = [
  { reference: "bullets", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } } ] },
];

// ---------- inline markup: **bold**, *italic*, `code` ----------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, ...base }));
    else if (tok.startsWith("`")) out.push(new TextRun({ text: tok.slice(1, -1), font: MONO, size: (base.size || 22) - 2, color: "7A1F1F", ...base, font: MONO }));
    else out.push(new TextRun({ text: tok.slice(1, -1), italics: true, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const p = (text, o = {}) => new Paragraph({
  children: runs(text, { size: o.size || 22, color: o.color }),
  spacing: { after: o.after ?? 120, before: o.before ?? 0, line: 276 },
  alignment: o.align || AlignmentType.LEFT, keepNext: o.keepNext,
});
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t })], keepNext: true });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t })], keepNext: true });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t })], keepNext: true });

const bullets = (items, level = 0) => items.map((t) => new Paragraph({
  numbering: { reference: "bullets", level }, children: runs(t, { size: 22 }), spacing: { after: 60, line: 270 } }));

function numbered(items) {
  const ref = `num${++numCounter}`;
  numberingConfigs.push({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 320 } } } }] });
  return items.map((t) => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t, { size: 22 }), spacing: { after: 70, line: 270 } }));
}

// ---------- code block ----------
function code(src) {
  const lines = src.replace(/\n$/, "").split("\n");
  return lines.map((ln, i) => new Paragraph({
    children: [new TextRun({ text: ln === "" ? " " : ln, font: MONO, size: 16 })],
    shading: { type: ShadingType.CLEAR, fill: CODEBG, color: "auto" },
    spacing: { after: i === lines.length - 1 ? 160 : 0, line: 240 },
    indent: { left: 120, right: 120 },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: BLUE, space: 6 } },
    keepLines: true,
  }));
}

// ---------- tables ----------
const thin = { style: BorderStyle.SINGLE, size: 4, color: "C9D3DD" };
const borders = { top: thin, bottom: thin, left: thin, right: thin };

function cell(text, w, o = {}) {
  const parts = Array.isArray(text) ? text : [text];
  return new TableCell({
    width: { size: w, type: WidthType.DXA }, borders,
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    verticalAlign: "center",
    children: parts.map((t) => new Paragraph({
      alignment: o.align || AlignmentType.LEFT, spacing: { after: 0, line: 252 },
      children: runs(String(t), { size: o.size || 18, bold: o.bold, color: o.color }) })),
  });
}

function table(headers, rows, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const w = widths.map((x) => Math.round(x * scale));
  w[w.length - 1] += CONTENT_W - w.reduce((a, b) => a + b, 0);
  const aligns = o.aligns || headers.map(() => AlignmentType.LEFT);
  const trs = [new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((hd, i) => cell(hd, w[i], { fill: NAVY, color: "FFFFFF", bold: true, align: aligns[i] === AlignmentType.RIGHT ? AlignmentType.RIGHT : AlignmentType.LEFT })) })];
  rows.forEach((r, ri) => {
    const hl = o.highlightRow !== undefined && ri === o.highlightRow;
    trs.push(new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, w[i], { fill: hl ? "FFF2CC" : (ri % 2 ? ZEBRA : undefined), align: aligns[i], size: o.size, bold: hl })) }));
  });
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: w, rows: trs });
}
const caption = (t, kind = "Table") => new Paragraph({
  children: [new TextRun({ text: `${kind} ${kind === "Table" ? ++tabCounter : figCounter}. `, bold: true, size: 18, color: NAVY }), ...runs(t, { size: 18, color: GREY })],
  spacing: { before: kind === "Table" ? 160 : 60, after: kind === "Table" ? 60 : 200 }, keepNext: kind === "Table",
  alignment: kind === "Table" ? AlignmentType.LEFT : AlignmentType.CENTER,
});
const tcap = (t) => caption(t, "Table");
const spacer = (n = 120) => new Paragraph({ children: [], spacing: { after: n } });

// ---------- figures ----------
function pngSize(path) {
  const b = fs.readFileSync(path);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(name, text, widthPx = 600, dir = "../figures") {
  figCounter++;
  const path = `${dir}/${name}.png`;
  const { w, h } = pngSize(path);
  const W = Math.min(widthPx, 610), H = Math.round(W * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 20 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: W, height: H }, altText: { title: name, description: text, name } })] }),
    new Paragraph({ children: [new TextRun({ text: `Figure ${figCounter}. `, bold: true, size: 18, color: NAVY }), ...runs(text, { size: 18, color: GREY })],
      alignment: AlignmentType.CENTER, spacing: { after: 200 } }),
  ];
}

// ---------- callout ----------
function callout(title, lines, fill = LIGHT, bar = BLUE) {
  const arr = Array.isArray(lines) ? lines : [lines];
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W],
    rows: [new TableRow({ cantSplit: true, children: [new TableCell({
      width: { size: CONTENT_W, type: WidthType.DXA },
      borders: { top: none, bottom: none, right: none, left: { style: BorderStyle.SINGLE, size: 24, color: bar } },
      shading: { type: ShadingType.CLEAR, fill, color: "auto" }, margins: { top: 100, bottom: 100, left: 180, right: 160 },
      children: [new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: title, bold: true, size: 22, color: NAVY })] }),
        ...arr.map((t) => new Paragraph({ spacing: { after: 50, line: 264 }, children: runs(t, { size: 21 }) }))],
    })] })],
  });
}

const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

// ---------- document ----------
function titleBlock(title, subtitle, meta) {
  return [
    new Paragraph({ spacing: { before: 0, after: 60 }, children: [new TextRun({ text: meta.toUpperCase(), size: 18, bold: true, color: BLUE, characterSpacing: 40 })] }),
    new Paragraph({ spacing: { before: 120, after: 100 }, children: [new TextRun({ text: title, size: 40, bold: true, color: NAVY })] }),
    new Paragraph({ spacing: { after: 200 }, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: BLUE, space: 8 } },
      children: [new TextRun({ text: subtitle, size: 24, color: GREY })] }),
  ];
}

function build(file, { headerText, children }) {
  const doc = new Document({
    creator: "Automotive Data Science Project", title: headerText,
    styles: {
      default: { document: { run: { font: FONT, size: 22 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 0,
            border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "C9D3DD", space: 3 } } } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 26, bold: true, font: FONT, color: BLUE }, paragraph: { spacing: { before: 260, after: 100 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 23, bold: true, font: FONT, color: "333333" }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
      ],
    },
    numbering: { config: numberingConfigs },
    sections: [{
      properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1300, bottom: 1200, left: MARGIN, right: MARGIN } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "C9D3DD", space: 4 } },
        children: [new TextRun({ text: headerText, size: 16, color: GREY })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Page ", size: 16, color: GREY }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: GREY }),
          new TextRun({ text: " of ", size: 16, color: GREY }), new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: GREY })] })] }) },
      children,
    }],
  });
  return Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(file, buf); console.log("wrote", file); });
}

const R = AlignmentType.RIGHT, L = AlignmentType.LEFT, C = AlignmentType.CENTER;
const f1 = (x) => Number(x).toFixed(1), f2 = (x) => Number(x).toFixed(2), f3 = (x) => Number(x).toFixed(3), f0 = (x) => Math.round(Number(x)).toString();
const load = (name) => JSON.parse(fs.readFileSync(`../results/${name}`, "utf8"));
const resetCounters = () => { figCounter = 0; tabCounter = 0; };

module.exports = { p, h1, h2, h3, bullets, numbered, code, table, tcap, caption, spacer, figure, callout, pageBreak, titleBlock, build,
  R, L, C, f0, f1, f2, f3, load, resetCounters, CONTENT_W, runs };
