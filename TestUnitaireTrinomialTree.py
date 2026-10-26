import unittest
from datetime import date
import math

from Bin_Tri_treeV4 import (
    Market,
    Option,
    Model,
    TrinomialTree,
    BlackScholes,
    price_trinomial_tree,
)

# vérifier la variance
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

        price = price_trinomial_tree(
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

        eu = price_trinomial_tree(
            m,
            Option(105, maturity, "C", "EU"),
            pricing_date,
            200
        )

        am = price_trinomial_tree(
            m,
            Option(105, maturity, "C", "AM"),
            pricing_date,
            200
        )

        self.assertGreaterEqual(am, eu)
        

    def test_div_column_moments(self):
        pricing_date = date(2026, 9, 15)
        D = 4.0
        m = Market(100, 0.03, 0.20, q_ex_div_date=date(2026, 9, 16), q_amount=D)
        o = Option(101, date(2027, 9, 2), "P", "EU")
        tree = TrinomialTree(m, o, Model(pricing_date, 50))
        tree.build()

        self.assertEqual(tree.col_div, 1)

        root = tree.root
        r, dt = m.r, tree.deltaT
        delta = dt - 1 / 365
        frac = delta / dt

        pu, pm, pd = root.probs()
        self.assertAlmostEqual(pu + pm + pd, 1.0)
        self.assertTrue(min(pu, pm, pd) >= 0)

        # contrainte 1 : espérance
        esp = pu * root.next_up.St + pm * root.next_mid.St + pd * root.next_down.St
        self.assertAlmostEqual(esp, root.forward() - D * math.exp(r * delta))

        # contrainte 2 : variance
        var = (pu * root.next_up.St ** 2 + pm * root.next_mid.St ** 2
               + pd * root.next_down.St ** 2) - esp ** 2
        V = (100 - frac * D) ** 2 * math.exp(2 * r * dt) * (tree.e - 1)
        self.assertAlmostEqual(var, V)

    def test_div_next_day_equals_adjusted_spot(self):
        # dividende le lendemain du pricing <=> arbre sans dividende démarrant à S0 - D*exp(-r/365)
        # (écart résiduel ~9e-4 : dividende cash contre spot ajusté, ce n'est pas une erreur)
        pricing_date = date(2026, 9, 15)
        maturity = date(2027, 9, 2)
        D, r, vol = 4.0, 0.03, 0.20

        m_div = Market(100, r, vol, q_ex_div_date=date(2026, 9, 16), q_amount=D)
        m_adj = Market(100 - D * math.exp(-r / 365), r, vol)

        for cp in ("P", "C"):
            o = Option(101, maturity, cp, "EU")
            price_with_div = price_trinomial_tree(m_div, o, pricing_date, 500)
            price_adjusted = price_trinomial_tree(m_adj, o, pricing_date, 500)
            self.assertAlmostEqual(price_with_div, price_adjusted, delta=0.002)

    def test_call_put_parity(self):
        # C - P = e^{-rT} (F - K), F = S0 e^{rT} - D e^{r(T - t_ex)} ; exact pour tout N (EU)
        pricing_date, maturity = date(2026, 9, 15), date(2027, 9, 2)
        T = (maturity - pricing_date).days / 365
        r, D, days = 0.03, 3.0, 120
        no_div = Market(100, r, 0.20)
        with_div = Market(100, r, 0.20, q_ex_div_date=date(2027, 1, 13), q_amount=D)   # 120 jours
        forward = 100 * math.exp(r * T) - D * math.exp(r * (T - days / 365))
        for n in (10, 57, 211):
            c = price_trinomial_tree(no_div, Option(95, maturity, "C", "EU"), pricing_date, n)
            p = price_trinomial_tree(no_div, Option(95, maturity, "P", "EU"), pricing_date, n)
            self.assertAlmostEqual(c - p, 100 - 95 * math.exp(-r * T), places=9)
            c = price_trinomial_tree(with_div, Option(95, maturity, "C", "EU"), pricing_date, n)
            p = price_trinomial_tree(with_div, Option(95, maturity, "P", "EU"), pricing_date, n)
            self.assertAlmostEqual(c - p, math.exp(-r * T) * (forward - 95), places=9)


if __name__ == "__main__":
    unittest.main()