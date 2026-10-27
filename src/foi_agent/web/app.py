"""The demo UI: FastAPI routes serving HTMX pages over `foi_agent.web.cases`.

python -m uvicorn foi_agent.web.app:app --reload

Triage runs a request and streams the path it takes as server-sent events. Review is the FOI
officer's queue of cases paused at the human_review interrupt.
"""

from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from foi_agent.web.cases import Cases

HERE = Path(__file__).parent


def sse(event: str, data: str) -> str:
    """One server-sent event. Every line of the data needs its own `data:` field."""
    lines = "\n".join(f"data: {line}" for line in data.splitlines() or [""])
    return f"event: {event}\n{lines}\n\n"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Loaded when the server starts rather than on import, so importing the app (as the tests do)
    # never picks up the keys and LANGSMITH_TRACING from .env.
    load_dotenv()
    yield


def create_app(graph=None, data_dir: str | Path | None = None) -> FastAPI:
    """Build the app. Tests pass a graph or monkeypatch the nodes, and a temporary data_dir."""
    cases = Cases(graph=graph, data_dir=data_dir)
    templates = Jinja2Templates(directory=HERE / "templates")
    app = FastAPI(title="Northbrook FOI triage", lifespan=_lifespan)
    app.state.cases = cases
    app.mount("/static", StaticFiles(directory=HERE / "static", check_dir=False), name="static")

    def page(request: Request, name: str, **context) -> HTMLResponse:
        context["pending_count"] = len(cases.pending())
        return templates.TemplateResponse(request, name, context)

    def partial(request: Request, name: str, **context) -> HTMLResponse:
        return templates.TemplateResponse(request, name, context)

    def render(name: str, **context) -> str:
        return templates.get_template(name).render(context)

    def example_or_404(line: int) -> dict:
        example = cases.example(line)
        if example is None:
            raise HTTPException(404, f"No example on line {line}.")
        return example

    @app.get("/")
    def triage(request: Request):
        form = {"request": "", "received": date.today().isoformat(), "example": None}
        return page(request, "triage.html", active="triage", examples=cases.examples(), form=form)

    @app.get("/examples/{line}")
    def example_form(request: Request, line: int):
        example = example_or_404(line)
        form = {
            "request": example["request"],
            "received": example["received"] or date.today().isoformat(),
            "example": example,
        }
        return partial(request, "partials/request_form.html", form=form)

    @app.post("/triage")
    def triage_request(
        request: Request,
        text: Annotated[str, Form(alias="request")],
        received: Annotated[str, Form()] = "",
    ):
        if not text.strip():
            raise HTTPException(400, "Enter the request text.")
        if received:
            try:
                date.fromisoformat(received)
            except ValueError:
                raise HTTPException(400, "The received date must be YYYY-MM-DD.") from None
        case_id = cases.create(text, received or None)
        return partial(request, "partials/run.html", live=True, **cases.new_run(case_id))

    @app.get("/cases/{case_id}/stream")
    def stream(case_id: str):
        if not cases.exists(case_id):
            raise HTTPException(404, "No such case.")

        # A sync generator, so Starlette runs the graph in its threadpool.
        def events():
            for event, context in cases.run(case_id):
                name = "partials/pipeline.html" if event == "pipeline" else "partials/result.html"
                yield sse(event, render(name, **context))

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/examples/{line}/replay")
    def replay(request: Request, line: int):
        example_or_404(line)
        replayed = cases.replay(line)
        if replayed is None:
            raise HTTPException(404, f"Example {line} has not been run yet.")
        return partial(request, "partials/run.html", live=False, **replayed)

    @app.get("/cases/{case_id}/trace")
    def trace(case_id: str):
        url = cases.trace_url(case_id)
        if url:
            return RedirectResponse(url, status_code=302)
        return HTMLResponse(
            "<p>No trace available. Tracing may be off (set LANGSMITH_TRACING=true), or "
            "LangSmith may not have finished recording this run yet.</p>"
        )

    @app.get("/review")
    def review_page(request: Request, case: str | None = None):
        selected = cases.case(case) if case else None
        return page(
            request,
            "review.html",
            active="review",
            pending=cases.pending(),
            decided=cases.decided(),
            selected=selected,
            selected_id=selected["id"] if selected else None,
        )

    @app.get("/review/{case_id}")
    def review_case(request: Request, case_id: str):
        case = cases.case(case_id)
        if case is None:
            raise HTTPException(404, "No such case.")
        return partial(request, "partials/review_case.html", case=case)

    @app.post("/review/{case_id}")
    def decide(
        case_id: str,
        action: Annotated[str, Form()],
        draft: Annotated[str, Form()] = "",
    ):
        try:
            case = cases.review(case_id, action, draft)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        if case is None:
            raise HTTPException(404, "No such case.")
        # The queue's root carries hx-swap-oob, so it replaces the queue alongside the case.
        queue = render(
            "partials/review_queue.html",
            pending=cases.pending(),
            decided=cases.decided(),
            selected_id=case_id,
        )
        return HTMLResponse(render("partials/review_case.html", case=case) + queue)

    return app


app = create_app()
