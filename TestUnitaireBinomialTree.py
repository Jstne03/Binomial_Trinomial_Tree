import math
import unittest
from datetime import date

from Bin_Tri_treeV4 import (
    Market,
    Option,
    Model,
    BinomialTree,
    BlackScholes,
    price_binomial_tree,
)


class TestBinomialTree(unittest.TestCase):

    def test_tree_parameters(self):
        pricing_date = date(2026, 9, 15)

        m = Market(100, 0.03, 0.20)

        o = Option(
            strike=101,
            t=date(2027, 9, 2),
            CP="C",
            exercise="EU"
        )

        tree = BinomialTree(
            m,
            o,
            Model(pricing_date, 2)
        )

        self.assertGreater(tree.deltaT, 0)
        self.assertTrue(0 < tree.u < 1)
        self.assertTrue(0 < tree.d < 1)

        self.assertAlmostEqual(tree.u + tree.d, 1.0)

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

        tree = BinomialTree(
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

        price = price_binomial_tree(
            m,
            o,
            pricing_date,
            50
        )

        self.assertAlmostEqual(price, 100.0)


    def test_american_put_greater_or_equal_european(self):
        pricing_date = date(2026, 9, 15)

        m = Market(100, 0.03, 0.20)

        maturity = date(2027, 9, 2)

        eu = price_binomial_tree(
            m,
            Option(105, maturity, "P", "EU"),
            pricing_date,
            200
        )

        am = price_binomial_tree(
            m,
            Option(105, maturity, "P", "AM"),
            pricing_date,
            200
        )

        self.assertGreaterEqual(am, eu)

    def test_call_put_parity(self):
        # l'arbre respecte l'espérance à chaque pas : C - P = S0 - K e^{-rT} exactement, pour tout N
        pricing_date, maturity = date(2026, 9, 15), date(2027, 9, 2)
        T = (maturity - pricing_date).days / 365
        m = Market(100, 0.03, 0.20)
        for n in (10, 57, 200):
            c = price_binomial_tree(m, Option(95, maturity, "C", "EU"), pricing_date, n)
            p = price_binomial_tree(m, Option(95, maturity, "P", "EU"), pricing_date, n)
            self.assertAlmostEqual(c - p, 100 - 95 * math.exp(-0.03 * T), places=9)

    def test_expected_value_and_variance_on_every_node(self):
        # espérance = forward et variance = S^2 e^{2r dt} (e^{vol^2 dt} - 1) sur TOUS les nœuds
        pricing_date = date(2026, 9, 15)
        m = Market(100, 0.03, 0.20)
        tree = BinomialTree(m, Option(101, date(2027, 9, 2), "C", "EU"), Model(pricing_date, 40))
        tree.price()
        e = math.exp(m.vol ** 2 * tree.deltaT)
        trunk = tree.last_trunk
        while trunk is not None:
            node = trunk.lowest()
            while node is not None:
                if node.next_up is not None:
                    up, dn = node.next_up.St, node.next_down.St
                    esp = tree.u * up + tree.d * dn
                    var = tree.u * up ** 2 + tree.d * dn ** 2 - esp ** 2
                    self.assertAlmostEqual(esp / node.forward(), 1.0, places=9)
                    self.assertAlmostEqual(
                        var / (node.St ** 2 * math.exp(2 * m.r * tree.deltaT) * (e - 1)), 1.0, places=6)
                node = node.above
            trunk = trunk.prec_trunk


if __name__ == "__main__":
    unittest.main()