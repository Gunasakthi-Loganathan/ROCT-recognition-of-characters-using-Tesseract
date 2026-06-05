# Contributing

## Development workflow

1. Create a branch for your change.
2. Do not commit `.env` files, datasets with private data, model checkpoints, API keys, passwords, or tokens.
3. Run the verification commands before opening a pull request:

```bash
npm ci
npm run lint
npm run test:all
npm run smoke
npm run build
npm run security:secrets
```

4. If you change dataset, training, evaluation, or inference behavior, update the reports in `reports/` and document whether Tesseract remains the default engine.

## Dataset contributions

Small synthetic fixtures are welcome for tests. Large datasets should not be committed; document how to obtain them and keep them under ignored paths such as `data/raw/` or `data/processed/`.

## Security

Report suspected secret exposure or unsafe upload/model-loading behavior immediately. Tests and examples must use placeholders only.
