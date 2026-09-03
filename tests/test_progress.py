from concurrent.futures import ThreadPoolExecutor

from runners.progress import ProgressReporter


def test_progress_reporter_is_thread_safe(capsys) -> None:
    reporter = ProgressReporter(total=100)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(reporter.advance, f"worker={index}")
            for index in range(100)
        ]
        for future in futures:
            future.result()

    assert reporter.completed == 100
    captured = capsys.readouterr()
    assert "100/100" in captured.err
    assert "100.0%" in captured.err
