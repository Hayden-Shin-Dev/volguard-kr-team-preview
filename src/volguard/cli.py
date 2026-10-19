import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from .config import KST, Settings, credential_status
from .data import Lake
from .providers import ingest_krx, probe
from .store import Store


def parser():
    p = argparse.ArgumentParser(description="VolGuard KR — read-only local research and risk gauge")
    p.add_argument("--demo", action="store_true", help="Use isolated synthetic data directory")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("demo")
    test = sub.add_parser("probe")
    test.add_argument("--date", default=(datetime.now(KST).date() - timedelta(days=1)).isoformat())
    ws = sub.add_parser("probe-ws")
    ws.add_argument("--symbols", required=True, help="comma-separated KRX stock codes, max 30")
    ws.add_argument("--seconds", type=int, default=15)
    ingest = sub.add_parser("ingest-krx")
    ingest.add_argument("--start", required=True)
    ingest.add_argument("--end")
    for name in ("import-csv", "import-calendar", "import-market"):
        cmd = sub.add_parser(name)
        cmd.add_argument("path", type=Path)
    universe = sub.add_parser("freeze-universe")
    universe.add_argument("path", type=Path)
    universe.add_argument("--as-of", required=True)
    universe.add_argument("--source", required=True)
    watch = sub.add_parser("watch")
    watch.add_argument("symbols", help="comma-separated codes from frozen universe")
    for name in ("validate", "freeze-study", "final-test", "data-gate", "ops-report"):
        sub.add_parser(name)
    for name in ("improve", "activate", "freeze-shadow", "serve"):
        sub.add_parser(name)
    for name in (
        "backfill",
        "crosscheck-prices",
        "build-history",
        "build-calendar",
        "audit-development",
        "audit-holdout",
        "audit-portfolio",
    ):
        sub.add_parser(name)
    for name in ("publish", "daily"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--date", default=datetime.now(KST).date().isoformat())
    collector = sub.add_parser("collect")
    collector.add_argument("--seconds", type=int)
    replay = sub.add_parser("replay")
    replay.add_argument("--ticks", type=int, default=120)
    replay.add_argument("--interval", type=float, default=1.0)
    return p


def execute(args):
    """Run research after either production daily path, even if that path failed."""
    try:
        return _execute(args)
    finally:
        if args.command == "daily" and not args.demo:
            settings = Settings.load().prepare()
            now = datetime.now(KST)
            sessions = Lake(settings.data_dir).calendar()
            session = next((s for s in sessions if s["date"] == args.date), None)
            if (
                args.date == now.date().isoformat()
                and session
                and now.strftime("%H:%M:%S") >= session["close_time"]
            ):
                try:
                    from .direction_forward import run

                    run(settings, args.date)
                except Exception as exc:
                    Store(settings.db_path).put(
                        "meta",
                        "direction_batch_error",
                        {
                            "at": now.isoformat(),
                            "error": type(exc).__name__,
                        },
                    )
                try:
                    from .probability_quality import prospective

                    prospective(settings)
                except Exception as exc:
                    Store(settings.db_path).put("meta", "probability_quality_error", {
                        "at": now.isoformat(), "error": type(exc).__name__
                    })
                try:
                    from .daily_sequence_forward import run as run_daily_sequence

                    run_daily_sequence(settings, args.date)
                except Exception as exc:
                    Store(settings.db_path).put("meta", "daily_sequence_research_error", {
                        "at": now.isoformat(), "error": type(exc).__name__
                    })
                try:
                    from .flow_forward import run as run_flow_research

                    run_flow_research(settings, args.date)
                except Exception as exc:
                    Store(settings.db_path).put("meta", "flow_research_error", {
                        "at": now.isoformat(), "error": type(exc).__name__
                    })
                try:
                    from .oof_signal_forward import run as run_oof_research

                    run_oof_research(settings, args.date)
                except Exception as exc:
                    Store(settings.db_path).put("meta", "oof_research_error", {
                        "at": now.isoformat(), "error": type(exc).__name__
                    })
                try:
                    from .oof_orthogonal_forward import run as run_orthogonal_research

                    run_orthogonal_research(settings, args.date)
                except Exception as exc:
                    Store(settings.db_path).put("meta", "orthogonal_research_error", {
                        "at": now.isoformat(), "error": type(exc).__name__
                    })


def _execute(args):
    settings = Settings.load(demo=args.demo).prepare()
    store, lake = Store(settings.db_path), Lake(settings.data_dir)
    if args.command in {"improve", "activate", "freeze-shadow", "serve"}:
        if args.demo:
            raise ValueError("실운영 명령은 데모에서 사용할 수 없습니다")
        from .improvement import freeze_shadow, run
        from .service import activate_baseline

        if args.command == "serve":
            from .supervisor import serve

            return serve(settings)
        with store.run(args.command):
            return {"improve": run, "activate": activate_baseline, "freeze-shadow": freeze_shadow}[
                args.command
            ](settings)
    if args.command in {
        "backfill",
        "crosscheck-prices",
        "build-history",
        "build-calendar",
        "audit-development",
        "audit-holdout",
        "audit-portfolio",
    }:
        if args.demo:
            raise ValueError("실데이터 수집·회고 평가는 demo에서 실행할 수 없습니다")
        from .adjustments import download_auxiliary
        from .audit import portfolio_audit, run_development, run_holdout
        from .backfill import download_history
        from .calendar import build_schedule
        from .history import materialize

        function = {
            "backfill": download_history,
            "crosscheck-prices": download_auxiliary,
            "build-history": materialize,
            "build-calendar": build_schedule,
            "audit-development": run_development,
            "audit-holdout": run_holdout,
            "audit-portfolio": portfolio_audit,
        }[args.command]
        with store.run(args.command):
            return function(settings)
    if args.command == "doctor":
        return {
            "python": sys.version.split()[0],
            "directory": str(settings.data_dir),
            "credentials": credential_status(),
            "mode": "demo" if args.demo else "live",
        }
    if args.command == "probe":
        if args.demo:
            raise ValueError("API probe는 실데이터 모드에서 실행하세요")
        return probe(settings, args.date)
    if args.command == "probe-ws":
        from .realtime import websocket_probe

        return asyncio.run(websocket_probe(settings, args.symbols.split(","), args.seconds))
    if args.command == "demo":
        if not args.demo:
            raise ValueError("demo 명령에는 --demo 옵션이 필요합니다")
        from .demo import seed_demo

        return seed_demo(settings)
    if args.command == "import-csv":
        with store.run("import-csv") as run:
            report = lake.import_csv(args.path)
            run["rows"] = report["rows"]
            return report
    if args.command == "import-calendar":
        lake.set_calendar(args.path)
        return {"sessions": len(lake.calendar())}
    if args.command == "import-market":
        from .research import import_market

        import_market(lake, args.path)
        return {"status": "imported"}
    if args.command == "freeze-universe":
        lake.freeze_universe(args.path, args.as_of, args.source, args.demo)
        return {"members": len(lake.universe()["members"])}
    if args.command == "watch":
        members = {m["symbol"]: m for m in lake.universe()["members"]}
        symbols = args.symbols.split(",")
        if not set(symbols).issubset(members):
            raise ValueError("유니버스에 없는 종목입니다")
        store.set_watchlist([members[s] for s in symbols], set(members))
        return {"watchlist": store.watchlist()}
    if args.command == "ingest-krx":
        import time

        import pandas as pd

        if args.demo:
            raise ValueError("실제 API 데이터는 demo에 수집하지 않습니다")
        dates = pd.date_range(args.start, args.end or args.start)
        if len(dates) > 31:
            raise ValueError("한 번에 최대 31일: 호출 제한을 확인하며 월별로 실행하세요")
        reports = []
        for day in dates:
            with store.run("ingest-krx") as run:
                report = ingest_krx(lake, day.strftime("%Y-%m-%d"))
                run["rows"] = report["rows"]
                reports.append(report)
            time.sleep(0.5)
        return {"days": reports}
    if args.command in {"validate", "freeze-study", "final-test", "data-gate"}:
        from . import research

        if args.command == "data-gate":
            return research.data_gate(lake, args.demo)
        function = {
            "validate": research.validate,
            "freeze-study": research.freeze,
            "final-test": research.final_test,
        }[args.command]
        with store.run(args.command):
            return function(settings, args.demo)
    if args.command in {"publish", "daily"}:
        from .research import publish_daily

        if args.command == "daily" and not args.demo:
            if (settings.data_dir / "config/forecast-release.json").exists():
                from .live_forecast import run_batch

                if args.date != datetime.now(KST).date().isoformat():
                    raise ValueError("과거 날짜 예측의 소급 발행은 허용하지 않습니다")
                if args.date not in [s["date"] for s in lake.calendar()]:
                    return {"status": "non_trading_day"}
                with store.run("kiwoom-daily") as run:
                    result = run_batch(settings)
                    # KRX remains the historical/label source, but cannot block broker inference.
                    last_official = str(lake.bars().date.max().date())
                    label_ingest = {}
                    for day in [s["date"] for s in lake.calendar() if last_official < s["date"] <= args.date][
                        -5:
                    ]:
                        try:
                            ingest_krx(lake, day)
                            label_ingest[day] = "received"
                        except Exception as exc:
                            label_ingest[day] = type(exc).__name__
                    result["krx_label_ingest"] = label_ingest
                    from .forward import evaluate

                    evaluate(settings)
                    run.update(status=result["status"], rows=len(result["predictions"]))
                    store.put(
                        "meta",
                        f"batch:{args.date}",
                        {
                            **result,
                            "input_date": args.date,
                            "observed_at": datetime.now(KST).isoformat(),
                            "demo": False,
                        },
                    )
                    return result
            from .service import run_daily

            return run_daily(settings, args.date)

        with store.run(args.command) as run:
            if args.command == "daily":
                if args.date not in [s["date"] for s in lake.calendar()]:
                    run["status"] = "skipped"
                    return {"status": "non_trading_day"}
                if not args.demo:
                    ingest_krx(lake, args.date)
            batch = None
            if args.command == "daily":
                import pandas as pd

                current = lake.bars()
                current = current[current.date.eq(pd.Timestamp(args.date))]
                expected = {m["symbol"] for m in store.watchlist()}
                verified = current.valid_label & current.pit_verified
                batch = {
                    "input_date": args.date,
                    "demo": args.demo,
                    "status": "running",
                    "data_ready": expected.issubset(set(current.loc[verified, "symbol"])),
                    "observed_at": datetime.now(KST).isoformat(),
                }
                store.put("meta", f"batch:{args.date}", batch)
            try:
                result = publish_daily(settings, args.date, args.demo)
            except Exception:
                if batch:
                    store.put("meta", f"batch:{args.date}", {**batch, "status": "failed"})
                raise
            run["rows"] = result["published"]
            run["status"] = result["status"]
            if batch:
                store.put("meta", f"batch:{args.date}", {**batch, **result})
            return result
    if args.command == "collect":
        if args.demo:
            raise ValueError("합성 시세는 replay를 사용하세요")
        from .realtime import collect

        asyncio.run(collect(settings, args.seconds))
        return {"status": "stopped"}
    if args.command == "replay":
        if not args.demo:
            raise ValueError("replay는 --demo에서만 실행할 수 있습니다")
        from .demo import replay_demo

        return replay_demo(settings, args.ticks, args.interval)
    if args.command == "ops-report":
        from .operations import report

        return report(store, settings)
    raise ValueError("Unknown command")


def main():
    args = parser().parse_args()
    try:
        result = execute(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    except KeyboardInterrupt:
        print("Stopped")
    except (ValueError, RuntimeError) as exc:
        # Only our validated errors are displayed; unexpected provider errors never print secrets.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"{type(exc).__name__}: 작업 실패. run_log와 입력 형식을 확인하세요.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
