# Dashboard (React + TypeScript + Vite)

Five pages — Overview, Forecasts, Inventory, What-if, Models — backed by the FastAPI
service in `src/demand_engine/api`.

```bash
npm ci
npm run dev        # http://localhost:5173, proxies /api to http://127.0.0.1:8000
npm run lint       # oxlint (warnings fail)
npm run typecheck  # tsc, strict mode
npm test           # vitest + Testing Library
npm run build      # dist/ — served by nginx (Docker) or by `demand-engine demo`
```

Structure: `src/api` (typed client mirroring the API schemas), `src/hooks` (data
loading), `src/components` (UI primitives, charts, layout), `src/pages`, `src/lib`
(formatting, filter state), `src/test` (fixtures and setup).
