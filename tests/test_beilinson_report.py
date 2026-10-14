"""The conductor report uses float64 and excludes every conjugate pair."""
import contextlib
import inspect
import io
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from hyperell_regulator.experiments import beilinson as b
from hyperell_regulator.regulator.polynomials import ring


def test_float64_and_conductor_cli_defaults():
    for fn in (b.run,b.analyse,b._analyse_triple):
        assert inspect.signature(fn).parameters['reg_digits'].default is None
    with patch.object(b,'run',return_value=[]) as run:
        b.main([])
        assert run.call_args.args==(1000,)
        assert run.call_args.kwargs['reg_digits'] is None
        assert run.call_args.kwargs['digits']==15
        b.main(['500','--no-lvalue','--verbose'])
        assert run.call_args.args==(500,)
        assert not run.call_args.kwargs['with_lvalue']


def test_triples_exclude_conjugate_pairs_including_infinity():
    x=ring().gen()
    C=SimpleNamespace(F=x**6+x+1,h=ring()(0),genus=lambda:2)
    points=[(0,1),(0,-1),(-1,1),(-1,-1),('infinity',1),('infinity',-1)]
    pairs=[dict(P=p,Q=q,same_component_everywhere=True,order_of_P_minus_Q=5)
           for p,q in itertools.combinations(points,2)]
    with patch.object(b,'odd_degree_model',return_value=object()):
        triples=b.find_triples({'pairs':pairs},C,row={'mw_invs':[5]})
    assert len(triples)==24
    for t in triples:
        assert len({t['P'][0],t['Q'][0],t['R'][0]})==3
        assert t['pairwise_nonconjugate'] and 'sum_nontrivial' not in t
    assert not b.pairwise_nonconjugate(C,points[0],points[1],points[2])
    assert not b.pairwise_nonconjugate(C,points[4],points[2],points[5])


def test_conductor_bound_is_exclusive_and_filters_before_preparation():
    with TemporaryDirectory() as d:
        cache=Path(d)/'curves.json';records=Path(d)/'pairs.json'
        rows=[dict(label=str(n),cond=n,is_simple_geom=True) for n in (999,1000,1001)]
        cache.write_text(json.dumps(rows));records.write_text(json.dumps(rows))
        with patch('hyperell_regulator.components.fetch_lmfdb') as fetch:
            with patch('hyperell_regulator.components.report_curve') as prepare:
                out=b.ensure_records(1000,cache,records)
        assert [r['cond'] for r in out]==[999]
        fetch.assert_not_called();prepare.assert_not_called()


def test_cache_policy_retries_failures_and_invalidates_old_reports():
    records=[dict(label='a',cond=249),dict(label='b',cond=295)]
    calls=[]
    def analyse(record,**kwargs):
        calls.append((record['label'],kwargs))
        return dict(record,status='ok' if record['label']=='a' else 'numerical_failure',
                    det=1.25 if record['label']=='a' else None,comparison={})
    with TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
        out=Path(d)/'report.json'
        out.write_text(json.dumps([dict(records[0],status='ok',det=999)]))
        with patch.object(b,'ensure_records',return_value=records),patch.object(b,'analyse',side_effect=analyse):
            first=b.run(out=out,with_lvalue=False)
            assert len(calls)==2 and first[0]['det']==1.25
            assert all(k['reg_digits'] is None and k['digits']==15 for _,k in calls)
            calls.clear();b.run(out=out,with_lvalue=False)
            assert [label for label,_ in calls]==['b']
            calls.clear();b.run(out=out,with_lvalue=False,force=True)
            assert len(calls)==2
        assert len(json.loads(out.read_text()))==2


def test_explicit_precision_failure_does_not_silently_fall_back():
    x=ring().gen();G=x**5+x+1
    t=dict(P=(0,1),Q=(1,1),R=('infinity',1),order_P=5,order_Q=5)
    with patch.object(b,'_model_for_triple',return_value=(G,[(0,1),(1,1)],None,'infinity')):
        with patch.object(b,'_resolve_triple_orders'):
            with patch.object(b,'beilinson_determinant_stokes',side_effect=RuntimeError('failed')) as calc:
                r=b._analyse_triple({'label':'test','cond':1},None,t,1,verbose=False,reg_digits=30)
    assert r['status']=='numerical_failure' and r['error']=='RuntimeError: failed'
    assert calc.call_count==1 and calc.call_args.kwargs['digits']==30


def test_higher_precision_report_is_reused_without_rounding():
    record = dict(label='a', cond=249)
    def analyse(record, **kwargs):
        return dict(record, status='ok', det='1.2345678901234567890123456789', comparison={})
    with TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        out = Path(d)/'report.json'
        with patch.object(b, 'ensure_records', return_value=[record]), patch.object(b, 'analyse', side_effect=analyse) as calc:
            b.run(out=out, reg_digits=50, digits=50, max_edge_seconds=120)
            saved = out.read_text()
            calc.reset_mock()
            b.run(out=out, reg_digits=30, digits=30)
            b.run(out=out)
            calc.assert_not_called()
            assert out.read_text() == saved
            b.run(out=out, reg_digits=60, digits=50)
            assert calc.call_count == 1
            b.run(out=out, reg_digits=30, digits=30, force=True)
            assert calc.call_count == 2


def test_precision_cache_boundaries():
    settings = dict(report_version=b.REPORT_VERSION, regulator_digits=30,
                    lvalue_digits=30, with_lvalue=True)
    old = dict(status='ok', settings=dict(settings, regulator_digits=50, lvalue_digits=50))
    assert b._stored_result_satisfies(old, settings)
    assert not b._stored_result_satisfies(old, dict(settings, lvalue_digits=60))
    assert not b._stored_result_satisfies(old, dict(settings, report_version=-1))
    old['settings']['regulator_digits'] = None
    assert not b._stored_result_satisfies(old, settings)
    old['settings']['regulator_digits'] = 10
    assert not b._stored_result_satisfies(old, dict(settings, regulator_digits=None))
    old['status'] = 'numerical_failure'
    assert not b._stored_result_satisfies(old, settings)
