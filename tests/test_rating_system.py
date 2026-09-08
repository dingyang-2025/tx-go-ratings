import unittest

import pandas as pd

from rating_system import (
    calculate_ratings,
    classify_event,
    k_factor_for,
    rating_status,
)


def games(rows):
    return pd.DataFrame(
        rows,
        columns=["Date", "Player1", "Player2", "Winner", "Note1", "Note2"],
    ).assign(Date=lambda frame: pd.to_datetime(frame["Date"]))


class RatingRuleTests(unittest.TestCase):
    def test_event_weights_cover_current_event_names(self):
        cases = {
            "第13届腾讯围棋大赛个人赛预选赛A组": 0.8,
            "第12届腾讯围棋大赛团体赛预选赛": 0.8,
            "第12届腾讯围棋大赛个人赛现场赛": 1.0,
            "第10届腾讯围棋大赛个人赛A组": 0.8,
            "捉早杯": 0.5,
            "捉早贺岁杯": 0.5,
            "菜鸡杯": 0.5,
            "练习赛": 0.0,
        }
        for event_name, expected_weight in cases.items():
            with self.subTest(event_name=event_name):
                self.assertEqual(classify_event(event_name)[1], expected_weight)

    def test_dynamic_k_boundaries_and_status(self):
        self.assertEqual(k_factor_for(0), 48)
        self.assertEqual(k_factor_for(9.9), 48)
        self.assertEqual(k_factor_for(10), 40)
        self.assertEqual(k_factor_for(29.9), 40)
        self.assertEqual(k_factor_for(30), 28)
        self.assertEqual(rating_status(9.9), "暂定")
        self.assertEqual(rating_status(10), "校准中")
        self.assertEqual(rating_status(30), "稳定")

    def test_each_player_uses_own_k(self):
        rows = []
        for day in range(1, 14):
            rows.append((f"2026-01-{day:02d}", "老棋手", f"路人{day}", "老棋手", "现场赛", ""))
        rows.append(("2026-02-01", "老棋手", "新人", "新人", "现场赛", ""))
        result = calculate_ratings(games(rows), version="v2")
        final_game = result.history[result.history["Date"] == pd.Timestamp("2026-02-01")]
        k_by_name = final_game.set_index("Name")["K_Factor"].to_dict()
        self.assertEqual(k_by_name["老棋手"], 40)
        self.assertEqual(k_by_name["新人"], 48)

    def test_weight_scales_change_and_effective_games(self):
        frame = games([
            ("2026-01-01", "甲", "乙", "甲", "第13届腾讯围棋大赛个人赛预选赛A组", "第1轮"),
        ])
        result = calculate_ratings(frame, version="v2")
        self.assertAlmostEqual(result.ratings["甲"], 1519.2)
        self.assertAlmostEqual(result.ratings["乙"], 1480.8)
        self.assertAlmostEqual(result.effective_games["甲"], 0.8)
        self.assertAlmostEqual(result.effective_games["乙"], 0.8)

    def test_unknown_event_is_kept_but_not_rated(self):
        frame = games([("2026-01-01", "甲", "乙", "甲", "内部练习", "")])
        result = calculate_ratings(frame, version="v2")
        self.assertEqual(result.ratings, {"甲": 1500.0, "乙": 1500.0})
        self.assertTrue((result.history["Event_Weight"] == 0).all())

    def test_v1_matches_fixed_k_32(self):
        frame = games([("2026-01-01", "甲", "乙", "甲", "任意赛事", "")])
        result = calculate_ratings(frame, version="v1")
        self.assertEqual(result.ratings, {"甲": 1516.0, "乙": 1484.0})
        self.assertEqual(result.effective_games, {"甲": 1.0, "乙": 1.0})


if __name__ == "__main__":
    unittest.main()
