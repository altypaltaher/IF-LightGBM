const fs = require("fs");
const d = require("docx");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  BorderStyle, ShadingType, TabStopType, LevelFormat, Footer, PageNumber, HeadingLevel,
  Math: M, MathRun: R, MathFraction: Fr, MathSubScript: Sub, MathSuperScript: Sup, MathSubSuperScript: SubSup,
  MathSum: Sum, MathRadical: Rad, MathRoundBrackets: RB, MathSquareBrackets: SB, MathCurlyBrackets: CB } = d;
const REFS = require("./refs20.js");
const B = require("./content20_filled.js");
const FONT = "Palatino Linotype", SZ = 20; // 10 pt

// ---------- citation numbering ----------
const order = [];
const cite = keys => {
  const nums = keys.map(k => { k = k.trim().replace(/^@/, ""); if (!REFS[k]) throw new Error("missing ref " + k);
    if (!order.includes(k)) order.push(k); return order.indexOf(k) + 1; }).sort((a, b) => a - b);
  // compress ranges
  const out = []; let i = 0;
  while (i < nums.length) { let j = i; while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
    out.push(j - i >= 2 ? `${nums[i]}–${nums[j]}` : j > i ? `${nums[i]},${nums[j]}` : `${nums[i]}`); i = j + 1; }
  return "[" + out.join(",") + "]";
};
const resolve = t => t.replace(/\[(@[^\]]+)\]/g, (_, g) => cite(g.split(",")));

// ---------- inline markup -> runs ----------
function runs(text, base = {}) {
  text = resolve(text);
  const out = [];
  const re = /(⟦[\s\S]*?⟧|\*\*[\s\S]*?\*\*|«[\s\S]*?»|\^\{[^}]*\}|_\{[^}]*\})/g;
  let last = 0, m;
  const push = (t, o = {}) => { if (t) out.push(new TextRun(Object.assign({ text: t, font: FONT, size: SZ }, base, o))); };
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index));
    const s = m[0];
    if (s.startsWith("⟦")) runs(s.slice(1, -1), Object.assign({}, base, { highlight: "yellow" })).forEach(r => out.push(r));
    else if (s.startsWith("**")) runs(s.slice(2, -2), Object.assign({}, base, { bold: true })).forEach(r => out.push(r));
    else if (s.startsWith("«")) runs(s.slice(1, -1), Object.assign({}, base, { italics: true })).forEach(r => out.push(r));
    else if (s.startsWith("^{")) push(s.slice(2, -1), { superScript: true });
    else push(s.slice(2, -1), { subScript: true });
    last = m.index + s.length;
  }
  push(text.slice(last));
  return out;
}
const para = (text, o = {}) => new Paragraph(Object.assign({ children: runs(text, o.run || {}), alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 80, line: 276 } }, o.p || {}));

// ---------- equations ----------
const r = t => new R(t);
const sub = (a, b) => new Sub({ children: [r(a)], subScript: [r(b)] });
const sup = (a, b) => new Sup({ children: [r(a)], superScript: [r(b)] });
const fr = (n, dd) => new Fr({ numerator: n, denominator: dd });
// robust n-ary sum (renders in Word and LibreOffice): ∑ with limits as sub/superscripts, followed by the body
const Sum2 = o => { const head = o.superScript ? new SubSup({ children: [r("∑")], subScript: o.subScript, superScript: o.superScript })
  : new Sub({ children: [r("∑")], subScript: o.subScript }); return [head, ...o.children]; };
const EQ = {
  aug3: [sub("z", "i"), r("="), new SB({ children: [sub("x", "i"), r(", "), sub("ν", "i"), r(", "), sub("π", "i")] }), r("∈"), sup("ℝ", "3d")],
  ct: [r("t="), fr([r("d̄")], [new Rad({ children: [new RB({ children: [fr([r("1")], [r("J")]), r("+"), fr([sub("n", "te")], [sub("n", "tr")])] }), sup("σ̂", "2")] })]), r(",   df=J−1")],
  ifs: [r("A={⟨x, "), sub("μ", "A"), r("(x), "), sub("ν", "A"), r("(x)⟩ ∣ x∈X},  0≤"), sub("μ", "A"), r("(x)+"), sub("ν", "A"), r("(x)≤1")],
  pi: [sub("π", "A"), r("(x)=1−"), sub("μ", "A"), r("(x)−"), sub("ν", "A"), r("(x)")],
  mu: [sub("μ", "ij"), r("=min"), new RB({ children: [r("1, max"), new RB({ children: [r("0, "), fr([sub("x", "ij"), r("−"), sub("m", "j")], [sub("M", "j"), r("−"), sub("m", "j")])] })] })],
  nu: [sub("ν", "ij"), r("="), new Sup({ children: [new RB({ children: [r("1−"), sub("μ", "ij")] })], superScript: [r("2")] })],
  pij: [sub("π", "ij"), r("=1−"), sub("μ", "ij"), r("−"), sub("ν", "ij"), r("="), sub("μ", "ij"), new RB({ children: [r("1−"), sub("μ", "ij")] }), r(",   0≤"), sub("π", "ij"), r("≤0.25")],
  aug: [sub("z", "i"), r("="), new SB({ children: [sub("x", "i"), r(", "), sub("x̃", "i"), r(", "), sub("μ", "i"), r(", "), sub("ν", "i"), r(", "), sub("π", "i")] }), r("∈"), sup("ℝ", "5d")],
  obj: [new Sup({ children: [r("ℒ")], superScript: [r("(t)")] }), r("="), ...Sum2({ children: [r("ℓ"), new RB({ children: [sub("y", "i"), r(", "), sub("p", "i")] })], subScript: [r("i")] }),
    r("+γL+"), fr([r("1")], [r("2")]), r("λ"), ...Sum2({ children: [new SubSup({ children: [r("w")], subScript: [r("l")], superScript: [r("2")] })], subScript: [r("l=1")], superScript: [r("L")] }),
    r(",   ℓ=−"), new SB({ children: [sub("y", "i"), r("log "), sub("p", "i"), r("+"), new RB({ children: [r("1−"), sub("y", "i")] }), r("log"), new RB({ children: [r("1−"), sub("p", "i")] })] })],
  gain: [new SubSup({ children: [r("w")], subScript: [r("l")], superScript: [r("∗")] }), r("=−"), fr([sub("G", "l")], [sub("H", "l"), r("+λ")]), r(",   Gain="), fr([r("1")], [r("2")]),
    new SB({ children: [fr([new SubSup({ children: [r("G")], subScript: [r("L")], superScript: [r("2")] })], [sub("H", "L"), r("+λ")]), r("+"),
      fr([new SubSup({ children: [r("G")], subScript: [r("R")], superScript: [r("2")] })], [sub("H", "R"), r("+λ")]), r("−"),
      fr([new Sup({ children: [new RB({ children: [sub("G", "L"), r("+"), sub("G", "R")] })], superScript: [r("2")] })], [sub("H", "L"), r("+"), sub("H", "R"), r("+λ")])] }), r("−γ")],
  accf1: [r("Accuracy="), fr([r("TP+TN")], [r("TP+TN+FP+FN")]), r(",   F1="), fr([r("2TP")], [r("2TP+FP+FN")])],
  metrics1: [r("Accuracy="), fr([r("TP+TN")], [r("TP+TN+FP+FN")]), r(",   Precision="), fr([r("TP")], [r("TP+FP")]), r(",   Recall="), fr([r("TP")], [r("TP+FN")])],
  metrics2: [r("F1="), fr([r("2·Precision·Recall")], [r("Precision+Recall")]), r(",  FPR="), fr([r("FP")], [r("FP+TN")])],
  mcc: [r("MCC="),
    fr([r("TP·TN−FP·FN")], [new Rad({ children: [r("(TP+FP)(TP+FN)(TN+FP)(TN+FN)")] })])],
  ap: [r("AP="), ...Sum2({ children: [new RB({ children: [sub("R", "n"), r("−"), sub("R", "n−1")] }), sub("P", "n")], subScript: [r("n")] })],
  mcnemar: [r("p=min"), new RB({ children: [r("1, 2"), ...Sum2({ children: [r("C"), new RB({ children: [r("b+c, k")] }), sup("0.5", "b+c")], subScript: [r("k=0")], superScript: [r("min(b,c)")] })] })],
  shap: [sub("φ", "j"), r("="), ...Sum2({ children: [fr([r("∣S∣!"), new RB({ children: [r("∣F∣−∣S∣−1")] }), r("!")], [r("∣F∣!")]),
    new SB({ children: [sub("f", "x"), new RB({ children: [r("S∪{j}")] }), r("−"), sub("f", "x"), new RB({ children: [r("S")] })] })], subScript: [r("S⊆F∖{j}")] })],
};
const eqPara = (id, n) => new Paragraph({ tabStops: [{ type: TabStopType.CENTER, position: 4680 }, { type: TabStopType.RIGHT, position: 9360 }],
  spacing: { before: 100, after: 100 },
  children: [new TextRun({ text: "\t", font: FONT, size: SZ }), new M({ children: EQ[id] }), new TextRun({ text: `\t(${n})`, font: FONT, size: SZ })] });

// ---------- tables ----------
const thin = { style: BorderStyle.SINGLE, size: 6, color: "000000" }, none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(o) {
  let widths = o.widths.slice(), header = o.header.slice(), rows = o.rows;
  if (o.drop !== undefined) { widths.splice(o.drop, 1); header.splice(o.drop, 1); }
  const total = widths.reduce((a, b) => a + b, 0);
  const fs = o.small ? 16 : 17;
  const mk = (txt, i, isH, isLast) => new TableCell({ width: { size: widths[i], type: WidthType.DXA },
    borders: { top: isH ? thin : none, bottom: (isH || isLast) ? thin : none, left: none, right: none },
    shading: isH ? { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    children: [new Paragraph({ alignment: i === 0 || (!o.small && i <= 2 && !/^[\d.]/.test(txt)) ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(txt, Object.assign({ size: fs }, isH ? { bold: true } : {}, o.hl && !isH ? { highlight: "yellow" } : {})) })] });
  const trs = [new TableRow({ tableHeader: true, children: header.map((h, i) => mk(h, i, true, false)) })];
  rows.forEach((row, ri) => trs.push(new TableRow({ cantSplit: true, children: row.map((c, i) => mk(c, i, false, ri === rows.length - 1)) })));
  return [new Paragraph({ children: runs(o.caption, { size: 18 }), spacing: { before: 160, after: 60 }, keepNext: true }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER, rows: trs }),
    new Paragraph({ children: [], spacing: { after: 120 } })];
}
// ---------- figures ----------
function figure(o) {
  const buf = fs.readFileSync(o.file);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const W = o.w || 600, H = Math.round(W * h / w);
  return [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: W, height: H } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 }, children: runs(o.caption, { size: 18 }) })];
}

// ---------- assemble ----------
const kids = [];
for (const b of B) {
  switch (b.t) {
    case "type": kids.push(para(b.text, { run: { italics: true, size: 18 }, p: { alignment: AlignmentType.LEFT } })); break;
    case "title": kids.push(new Paragraph({ heading: HeadingLevel.TITLE, spacing: { before: 120, after: 160 }, children: runs(b.text, { bold: true, size: 36 }) })); break;
    case "authors": kids.push(para(b.text, { run: { bold: true, size: 20 }, p: { alignment: AlignmentType.LEFT, spacing: { after: 120 } } })); break;
    case "affil": b.lines.forEach(l => kids.push(para(l, { run: { size: 16 }, p: { alignment: AlignmentType.LEFT, spacing: { after: 20 } } }))); kids.push(new Paragraph({ children: [] })); break;
    case "abstract": kids.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100, line: 260 },
      children: [new TextRun({ text: "Abstract: ", bold: true, font: FONT, size: 18 }), ...runs(b.text, { size: 18 })] })); break;
    case "keywords": kids.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 240 },
      children: [new TextRun({ text: "Keywords: ", bold: true, font: FONT, size: 18 }), ...runs(b.text, { size: 18 })] })); break;
    case "h1": kids.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 100 }, keepNext: true, children: runs(b.text, { bold: true, size: 22 }) })); break;
    case "h2": kids.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 160, after: 80 }, keepNext: true, children: runs(b.text, { italics: true, size: 20 }) })); break;
    case "p": kids.push(para(b.text, { p: { indent: { firstLine: 340 } } })); break;
    case "pn": kids.push(para(b.text)); break;
    case "bullet": kids.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 276 }, children: runs(b.text) })); break;
    case "eq": kids.push(eqPara(b.id, b.n)); break;
    case "table": kids.push(...table(b)); break;
    case "fig": kids.push(...figure(b)); break;
    case "algo": {
      const rows = [b.title, ...b.lines].map((l, i) => new TableRow({ children: [new TableCell({ width: { size: 9000, type: WidthType.DXA },
        borders: { top: i <= 1 ? thin : none, bottom: (i === 0 || i === b.lines.length) ? thin : none, left: none, right: none },
        margins: { top: 30, bottom: 30, left: 100, right: 100 },
        children: [new Paragraph({ children: runs(l, { size: 18 }) })] })] }));
      kids.push(new Paragraph({ children: [], spacing: { before: 80 } }), new Table({ width: { size: 9000, type: WidthType.DXA }, columnWidths: [9000], alignment: AlignmentType.CENTER, rows }), new Paragraph({ children: [], spacing: { after: 120 } }));
      break; }
    case "back": b.items.forEach(([k, v]) => kids.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 80 },
      children: [new TextRun({ text: k + " ", bold: true, font: FONT, size: 18 }), ...runs(v, { size: 18 })] }))); break;
    case "abbr": {
      kids.push(new Paragraph({ spacing: { before: 160, after: 60 }, children: [new TextRun({ text: "Abbreviations", bold: true, font: FONT, size: 18 })] }));
      kids.push(new Table({ width: { size: 7000, type: WidthType.DXA }, columnWidths: [1600, 5400], rows: b.rows.map(([a, x]) => new TableRow({ children: [a, x].map((t, i) =>
        new TableCell({ width: { size: i ? 5400 : 1600, type: WidthType.DXA }, borders: { top: none, bottom: none, left: none, right: none }, children: [new Paragraph({ children: runs(t, { size: 18 }) })] })) })) }));
      break; }
    case "refs": {
      kids.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 100 }, children: [new TextRun({ text: "References", bold: true, font: FONT, size: 22 })] }));
      order.forEach((k, i) => kids.push(new Paragraph({ alignment: AlignmentType.LEFT, indent: { left: 400, hanging: 400 }, spacing: { after: 40 },
        children: [new TextRun({ text: `${i + 1}.\t`, font: FONT, size: 18 }), new TextRun({ text: REFS[k], font: FONT, size: 18 })],
        tabStops: [{ type: TabStopType.LEFT, position: 400 }] })));
      const unused = Object.keys(REFS).filter(k => !order.includes(k));
      if (unused.length) console.log("UNCITED:", unused.join(", "));
      console.log("references:", order.length);
      break; }
  }
}
const doc = new Document({
  creator: "Altyeb Altaher Taha", title: "IF feature augmentation for GBDT malware detection",
  styles: { default: { document: { run: { font: FONT, size: SZ } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 36, bold: true, color: "000000" } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 22, bold: true, color: "000000" }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 20, italics: true, color: "000000" }, paragraph: { outlineLevel: 1 } },
    ] },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 560, hanging: 280 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1270, right: 1270 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] }) },
    children: kids }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(process.argv[2] || "out.docx", buf); console.log("written"); });
