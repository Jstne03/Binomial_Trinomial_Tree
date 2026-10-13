import unittest
from datetime import date

from trinomialTreeV2 import (
    Market,
    Option,
    Model,
    TrinomialTree,
    price_tree,
)


class TestTrinomialTree(unittest.TestCase):

    def test_tree_parameters(self):
        pricing_date = date(2026, 9, 15)

        m = Market(100, 0.03, 0.20)

        o = Option(
            strike=101,
            t=date(2027, 9, 2),
            CP="C",
            exercise="EU"
        )

        tree = TrinomialTree(
            m,
            o,
            Model(pricing_date, 2)
        )

        self.assertGreater(tree.deltaT, 0)
        self.assertTrue(0 < tree.pu < 1)
        self.assertTrue(0 < tree.pd < 1)
        self.assertTrue(0 < tree.pm < 1)

        self.assertAlmostEqual(tree.pu + tree.pd + tree.pm, 1.0)

        self.assertAlmostEqual(
            tree.root.check_expected_value(),
            tree.root.forward()
        )

    def test_tree_recombination(self):
        pricing_date = date(2026, 9, 15)

        m = Market(100, 0.03, 0.20)

        o = Option(
            strike=101,
            t=date(2027, 9, 2),
            CP="C",
            exercise="EU"
        )

        tree = TrinomialTree(
            m,
            o,
            Model(pricing_date, 2)
        )

        tree.build()

        up = tree.root.next_up
        down = tree.root.next_down

        self.assertIsNotNone(up)
        self.assertIsNotNone(down)

        self.assertIs(up.next_down, down.next_up)

    def test_call_strike_zero(self):
        pricing_date = date(2026, 9, 15)

        m = Market(100, 0.03, 0.20)

        o = Option(
            strike=0,
            t=date(2027, 9, 2),
            CP="C",
            exercise="EU"
        )

        price = price_tree(
            m,
            o,
            pricing_date,
            50
        )

        self.assertAlmostEqual(price, 100.0)

    def test_american_call_greater_or_equal_european(self):
        pricing_date = date(2026, 9, 15)

        m = Market(100, 0.03, 0.20)

        maturity = date(2027, 9, 2)

        eu = price_tree(
            m,
            Option(105, maturity, "C", "EU"),
            pricing_date,
            200
        )

        am = price_tree(
            m,
            Option(105, maturity, "C", "AM"),
            pricing_date,
            200
        )

        self.assertGreaterEqual(am, eu)

    def test_american_put_greater_or_equal_european(self):
        pricing_date = date(2026, 9, 15)

        m = Market(100, 0.03, 0.20)

        maturity = date(2027, 9, 2)

        eu = price_tree(
            m,
            Option(105, maturity, "P", "EU"),
            pricing_date,
            200
        )

        am = price_tree(
            m,
            Option(105, maturity, "P", "AM"),
            pricing_date,
            200
        )

        self.assertGreaterEqual(am, eu)


if __name__ == "__main__":
    unittest.main()