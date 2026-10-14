"""Integration orders take precedence over cut conditioning in both backends."""
from types import SimpleNamespace
from unittest.mock import patch
from hyperell_regulator.experiments import beilinson as b


def test_orders_then_conditioning():
    def triple(p, q, ratio, detour=False):
        return dict(P=(0, 1), Q=(1, 1), R=('infinity', 1),
                    order_P=p, order_Q=q, ratio=ratio, detour=detour)
    high = triple(5, 9, 2)
    larger_second = triple(7, 7, 1)
    worse_gap = triple(5, 7, 1000)
    best = triple(7, 5, 100, True)
    smaller_sum = triple(3, 8, 10000)
    unavailable = triple(3, 3, 1)
    triples = [high, larger_second, worse_gap, unavailable, best, smaller_sum]
    active = {}

    def model(C, t):
        active.clear(); active.update(t)
        if t is unavailable:
            raise ValueError('no usable cover')
        return SimpleNamespace(degree=lambda: 6), [t['P'], t['Q']], None, 'infinity'

    def conditioning(G, point, n, pole):
        return n, 5, active['ratio'], 1, active['detour']

    with patch.object(b, '_model_for_triple', side_effect=model), \
         patch.object(b, '_resolve_triple_orders'), \
         patch.object(b, 'cut_conditioning', side_effect=conditioning):
        for mp in (False, True):
            ranked = b.rank_triples(None, triples, mp=mp, usable_only=True)
            assert ranked == [smaller_sum, best, worse_gap, larger_second, high]
            assert b.triple_conditioning(None, best, mp=mp)[0] == (12, 100)
            assert b.rank_triples(None, triples, mp=mp)[-1] is unavailable
