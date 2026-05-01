# Copilot System Instructions: Credit Portfolio Management Research

## 1. General Interaction
- **No Emojis**: Do not use smileys or emojis in any responses.
- **Language**: Respond in the same language as the user (Russian), but all code artifacts (docstrings, comments) must be in English.
- **Dialog closure**: Before finishing any assistant response, ALWAYS call the MCP server's ask_user_text method from the VS Code extension (mcp.ask_user_text) to ask: "Могу ли я закончить диалог?" Only finish the response after receiving the user's answer. Dialog can ONLY be closed through MCP with explicit "да" (yes) confirmation — never close directly or proceed without explicit user confirmation. If the user answers anything other than "да", the dialog must continue.
- **MCP Question Rule**: Every response MUST end with the MCP question "Могу ли я закончить диалог?" without exception. This is the ONLY way to close a dialog.

## 2. Python Coding Standards
- **Style Guide**: Follow PEP 8 strictly.
- **Docstrings**: Use Google-style docstrings for all functions.
- **Typing**: Use static typing (type hints) for all function arguments and returns.
- **Formatting**: Adhere to `black` (120 chars) and `flake8` rules.
- **Forbidden**: NEVER use `try-except` blocks.

## 3. Documentation & Comments
- **Minimalism**: Write comments ONLY when code logic is extremely complex.
- **Language**: All comments and docstrings must be in **English**.

## 4. Git Commit Standards
Follow the **Conventional Commits 1.0.0** specification:
- **Format**: `<type>(<scope>): <short description>;`
- **Body**: Use a bulleted list for detailed changes if the commit contains multiple modifications.
- **Rule**: Description and body must be in English, lowercase, and the short description must end with `;`.
- **Types**:
  - `feat`: New features
  - `fix`: Bug fixes
  - `docs`: Documentation changes
  - `style`: Formatting, missing semi-colons, etc. (no code changes)
  - `refactor`: Code changes that neither fix a bug nor add a feature
  - `perf`: Code changes that improve performance
  - `test`: Adding missing tests
  - `chore`: Changes to the build process or auxiliary tools/libraries
- **Example**:
  ```text
  feat(portfolio): implement risk-based optimization and backtesting;

  - add `optimize_portfolio` to minimize volatility and expected loss.
  - implement `backtest_portfolio_strategies` for active vs passive comparison.
  - fix `Backtest Start` marker logic in strategy plots;
  ```

## 5. PDF Reading Rule
- For any request that involves reading, summarizing, translating, extracting text from, or analyzing a PDF, ALWAYS use the `markitdown` MCP server first.
- Treat PDF files as requiring tool-based extraction, not visual guessing.
- First convert the PDF to Markdown with the `markitdown` MCP tool, then work only from the extracted content.
- If a PDF path is available, use `#markitdown` explicitly before answering.
- If the PDF cannot be read through `markitdown`, clearly say that the tool failed and ask for another file or path.

## 6. LaTeX Coding Standards
- **Line length**: Maximum 100 characters per source line. Wrap long text at natural phrase boundaries.
- **Display math**: Use `\[...\]` (LaTeX2e). NEVER use `$$...$$` (plain TeX).
- **Indentation**: 4 spaces inside all environments (`\begin{...}...\end{...}`), including `align*`, `pmatrix`, `document`.
- **Operators**: Define custom math operators with `\DeclareMathOperator` in the preamble instead of repeating `\mathrm{...}` inline.
- **Paragraph spacing**: Prefer global `\setlength{\parindent}{0pt}` + `\setlength{\parskip}{...}` instead of scattering `\noindent` manually.
- **Percent sign**: Use `\,\%` (thin space before `\%`) in both text and math mode for typographic correctness.
- **Non-breaking spaces**: Use `~` before dashes (`~---`), references, units, and short words to prevent bad line breaks.
- **Decimal comma**: Use `{,}` for Russian-locale decimal separators inside math mode (e.g., `0{,}50`).
- **Packages**: Only include packages actually used in the document. Remove unused imports.
- **Semantic markup**: Use `\emph{}` for emphasis, `\textbf{}` for structural headers. Do not use font commands (`\bf`, `\it`).
- **Structure style** (for homework/exam solutions): Use `\textbf{Условие}` / `\textbf{Решение}` / `\textbf{Ответ}` as section-like headers without `\section{}`. Separate sections with `\bigskip`.
- **Formula layout**: Break long formulas across lines inside `\[...\]` using alignment (`=` on separate lines). Inside `align*`, break each equation row into readable parts with `&=` alignment.
- **Comments**: All LaTeX comments must be in English (same as code comments rule).
