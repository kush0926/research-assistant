"use client";

import { useState } from "react";

type Coverage = { rows: string[]; cols: string[]; cells: number[][] };
type CombinationMatrix = {
  problems: string[];
  datasets: string[];
  methods: string[];
  counts: { problem: string; dataset: string; method: string; count: number }[];
};
type Matrix = {
  problem_dataset_method?: CombinationMatrix;
  problem_dataset?: Coverage;
  problem_method?: Coverage;
};
type View = "problem_dataset_method" | "problem_dataset" | "problem_method";
const VIEWS: { value: View; label: string }[] = [
  { value: "problem_dataset_method", label: "Dataset × method" },
  { value: "problem_dataset", label: "Problem × dataset" },
  { value: "problem_method", label: "Problem × method" },
];
const ROWS_PER_PAGE = 12;
const COLS_PER_PAGE = 8;

export default function SparsityMatrix({ matrix, corpusSize }: { matrix: Matrix; corpusSize: number }) {
  const combinations = matrix.problem_dataset_method;
  const [view, setView] = useState<View>(combinations ? "problem_dataset_method" : "problem_dataset");
  const [problem, setProblem] = useState(combinations?.problems[0] || "");
  const [rowPage, setRowPage] = useState(0);
  const [colPage, setColPage] = useState(0);
  const isCombination = view === "problem_dataset_method";
  const problemCounts = (combinations?.counts || []).filter((cell) => cell.problem === problem);
  const counts = new Map(
    problemCounts.map((cell) => [JSON.stringify([cell.dataset, cell.method]), cell.count]),
  );
  const coverage = isCombination ? undefined : matrix[view];
  const rows = (isCombination ? combinations?.datasets : coverage?.rows) || [];
  const cols = (isCombination ? combinations?.methods : coverage?.cols) || [];
  const cells = coverage?.cells || [];
  const hasCells = rows.length > 0 && cols.length > 0 && (!isCombination || Boolean(problem));
  const values = hasCells ? (isCombination ? problemCounts.map((cell) => cell.count) : cells.flat()) : [];
  const total = hasCells ? rows.length * cols.length : 0;
  const empty = total - values.filter((count) => count > 0).length;
  const maximum = values.reduce((max, count) => Math.max(max, count), 1);
  const rowLabel = isCombination ? "Dataset" : "Research problem";
  const columnLabel = view === "problem_dataset" ? "dataset" : "method";
  const rowStart = rowPage * ROWS_PER_PAGE;
  const colStart = colPage * COLS_PER_PAGE;
  const visibleRows = rows.slice(rowStart, rowStart + ROWS_PER_PAGE);
  const visibleCols = cols.slice(colStart, colStart + COLS_PER_PAGE);
  function changeView(nextView: View) { setView(nextView); setRowPage(0); setColPage(0); }

  return <section className="analysis-table-panel gap-matrix-panel" aria-labelledby="sparsity-heading">
    <div className="analysis-section-header">
      <div><h2 id="sparsity-heading">Sparsity matrix</h2><p className="muted">{isCombination
        ? "The problem × dataset × method paper counts used to find candidate gaps. Choose a research problem to inspect its combinations."
        : "Pairwise corpus coverage for context. A populated pair can still contain an empty dataset–method combination."}</p></div>
      <span className="analysis-count">{corpusSize} papers</span>
    </div>
    <div className="gap-matrix-controls">
      <div className="analysis-segmented gap-matrix-views" role="group" aria-label="Matrix view">
        {VIEWS.filter((option) => matrix[option.value]).map((option) => <button type="button" key={option.value}
          aria-pressed={view === option.value} onClick={() => changeView(option.value)}>{option.label}</button>)}
      </div>
      {isCombination && <label className="gap-matrix-problem" htmlFor="matrix-problem">Research problem
        <select id="matrix-problem" value={problem} onChange={(event) => { setProblem(event.target.value); setRowPage(0); setColPage(0); }} disabled={!combinations?.problems.length}>
          {combinations?.problems.length ? combinations.problems.map((name) => <option key={name} value={name}>{name}</option>) : <option value="">No problems extracted</option>}
        </select>
      </label>}
    </div>
    {hasCells ? <>
      <div className="gap-matrix-summary">
        <p role="status"><strong>{empty.toLocaleString()}</strong> of {total.toLocaleString()} {isCombination ? "combinations" : "pairs"} empty <span>({Math.round(empty / total * 100)}% sparsity in this view)</span></p>
        <div className="gap-matrix-legend" aria-label="Paper count legend">
          <span><i className="gap-matrix-empty" />0 papers</span><span><i className="gap-matrix-low" />Fewer</span><span><i className="gap-matrix-high" />More papers</span>
        </div>
      </div>
      <div className="analysis-table-scroll gap-matrix-scroll" tabIndex={0} role="region" aria-label="Scrollable sparsity matrix" aria-describedby="sparsity-note">
        <table className="analysis-data-table gap-heatmap gap-matrix-table">
          <caption className="sr-only">{isCombination ? `Dataset by method paper counts for ${problem}.` : `Research problem by ${columnLabel} paper counts.`} Zero means no papers in this analysis have this extracted combination.</caption>
          <thead><tr><th scope="col">{rowLabel} <span className="gap-matrix-axis">↓ / {columnLabel} →</span></th>{visibleCols.map((name) => <th scope="col" key={name}>{name}</th>)}</tr></thead>
          <tbody>{visibleRows.map((name, row) => <tr key={name}><th scope="row">{name}</th>{visibleCols.map((column, col) => {
            const count = isCombination ? counts.get(JSON.stringify([name, column])) || 0 : cells[rowStart + row]?.[colStart + col] || 0;
            const level = count === 0 ? "empty" : count / maximum > 0.66 ? "high" : count / maximum > 0.33 ? "medium" : "low";
            return <td key={column}><span className={`gap-heat-value gap-matrix-${level}`} title={`${isCombination ? `${problem} · ` : ""}${name} × ${column}: ${count} paper${count === 1 ? "" : "s"}`}>{count}</span></td>;
          })}</tr>)}</tbody>
        </table>
      </div>
      {(rows.length > ROWS_PER_PAGE || cols.length > COLS_PER_PAGE) && <div className="gap-matrix-pagination">
        <div><span>{rowLabel}s {rowStart + 1}–{rowStart + visibleRows.length} of {rows.length}</span>
          {rows.length > ROWS_PER_PAGE && <><button className="btn ghost small" type="button" aria-label="Previous matrix rows" disabled={rowPage === 0} onClick={() => setRowPage(rowPage - 1)}>Previous</button><button className="btn ghost small" type="button" aria-label="Next matrix rows" disabled={rowStart + ROWS_PER_PAGE >= rows.length} onClick={() => setRowPage(rowPage + 1)}>Next</button></>}
        </div>
        <div><span>{columnLabel === "dataset" ? "Datasets" : "Methods"} {colStart + 1}–{colStart + visibleCols.length} of {cols.length}</span>
          {cols.length > COLS_PER_PAGE && <><button className="btn ghost small" type="button" aria-label="Previous matrix columns" disabled={colPage === 0} onClick={() => setColPage(colPage - 1)}>Previous</button><button className="btn ghost small" type="button" aria-label="Next matrix columns" disabled={colStart + COLS_PER_PAGE >= cols.length} onClick={() => setColPage(colPage + 1)}>Next</button></>}
        </div>
      </div>}
    </> : <div className="analysis-inline-empty">{isCombination
      ? "Not enough extracted problems, datasets, and methods to build this matrix. Include papers with these details and run the analysis again."
      : `No ${columnLabel} coverage was extracted for this scope.`}</div>}
    <p id="sparsity-note" className="gap-matrix-note">{isCombination
      ? "0 means no extracted papers for that combination in this corpus. Empty cells are structural signals; supporting evidence and populated neighbours determine candidate gaps."
      : "This overview shows up to 6 research problems and the 8 most frequent datasets or methods. Use Dataset × method to inspect the full combinations."}</p>
  </section>;
}
