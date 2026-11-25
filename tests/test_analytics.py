import tempfile
import unittest
from pathlib import Path

from src.analytics.blink_analysis import BlinkAnalyzer
from src.analytics.perclos import PerclosCalculator
from src.analytics.report_generator import generate_session_report_png
from src.analytics.session_storage import SessionStorage


class TestAnalytics(unittest.TestCase):
    def test_perclos_window_calculation(self):
        calc = PerclosCalculator(window_seconds=60.0)
        t = 0.0
        # 20 секунд по 0.5 с: половина кадров глаза закрыты, половина открыты
        for i in range(40):
            t += 0.5
            perclos = calc.update(t, is_closed=(i % 2 == 0))
        self.assertGreater(perclos, 40.0)
        self.assertLess(perclos, 60.0)

    def test_blink_analyzer_counts_valid_blinks(self):
        analyzer = BlinkAnalyzer(window_seconds=60.0)
        # Нормальное моргание длительностью 0.18 с
        analyzer.update(1.00, is_closed=True, elapsed_session_sec=10.0)
        res = analyzer.update(1.18, is_closed=False, elapsed_session_sec=10.2)
        self.assertEqual(res["total_blinks"], 1)
        self.assertAlmostEqual(res["last_blink_duration_ms"], 180.0, delta=5.0)

        # Слишком долгое закрытие (1.5 с) не должно засчитываться как обычное моргание
        analyzer.update(3.00, is_closed=True, elapsed_session_sec=12.0)
        res_long = analyzer.update(4.50, is_closed=False, elapsed_session_sec=13.5)
        self.assertEqual(res_long["total_blinks"], 1)

    def test_session_storage_save_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_path = tmp_path / "test_sessions.db"
            csv_path = tmp_path / "test_sessions.csv"
            storage = SessionStorage(db_path=db_path, csv_path=csv_path)

            storage.save_session(
                {
                    "start_time": "2026-11-17 00:00:00",
                    "duration_sec": 45.0,
                    "total_blinks": 12,
                    "avg_blink_rate": 16.0,
                    "avg_blink_ms": 175.0,
                    "avg_perclos": 8.5,
                    "max_perclos": 14.0,
                    "alerts_count": 0,
                    "time_normal_sec": 45.0,
                    "time_elevated_sec": 0.0,
                    "time_high_sec": 0.0,
                    "model_name": "eye_cnn_finetuned.pth",
                }
            )

            sessions = storage.get_recent_sessions(limit=5)
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["model_name"], "eye_cnn_finetuned.pth")
            self.assertTrue(csv_path.exists())

    def test_report_generator_creates_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample_session.csv"
            rows = [
                "timestamp_ms,score,perclos,left_ear,right_ear,pitch_deg,yaw_deg,roll_deg,mar",
                "0,10.0,2.0,0.29,0.28,1.0,0.0,0.0,0.20",
                "100,15.0,4.0,0.28,0.28,2.0,1.0,0.0,0.22",
                "200,20.0,6.0,0.27,0.26,1.5,-1.0,0.0,0.25",
                "300,45.0,18.0,0.14,0.13,5.0,0.0,1.0,0.60",
                "400,72.0,30.0,0.08,0.08,12.0,0.0,2.0,0.20",
                "500,12.0,10.0,0.29,0.29,0.0,0.0,0.0,0.18",
            ]
            csv_path.write_text("\n".join(rows), encoding="utf-8")
            out_png = generate_session_report_png(csv_path)
            self.assertIsNotNone(out_png)
            self.assertTrue(out_png.exists())
            out_png.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

