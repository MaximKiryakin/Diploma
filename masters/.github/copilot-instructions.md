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
