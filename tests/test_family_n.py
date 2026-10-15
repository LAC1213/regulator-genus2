"""Family reports share the Beilinson integrator, precision, and cache policy."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sage.all import RealField
from hyperell_regulator.experiments import family_n as f
from hyperell_regulator.regulator import IntegrationMonitor


def test_family_cli_defaults_and_precision_alias():
    with patch.object(f, 'run', return_value=[]) as run:
        f.main([])
        assert run.call_args.args == (None,)
        assert run.call_args.kwargs['reg_digits'] is None
        assert run.call_args.kwargs['digits'] == 15
        assert run.call_args.kwargs['max_edge_seconds'] == 15
        f.main(['2', '--prec', '30', '--digits', '35', '--no-lvalue', '--edge-seconds', '45'])
        assert run.call_args.args == ([2],)
        assert run.call_args.kwargs['reg_digits'] == 30
        assert run.call_args.kwargs['digits'] == 35
        assert not run.call_args.kwargs['with_lvalue']


def test_family_determinant_dispatch_keeps_precision():
    with patch.object(f.b, 'beilinson_determinant_stokes', return_value=('det', 'matrix', None)) as calc:
        assert f.family_determinant('F', dps=30) == ('det', 'matrix', None)
    calc.assert_called_once_with('F', (0, 0), 2, (1, 0), 2, digits=30)
    try:
        f.family_determinant('F', dps=30, reg_digits=40)
    except ValueError:
        pass
    else:
        raise AssertionError('conflicting precision aliases accepted')


def test_family_cache_validates_precision_and_retries_failures():
    calls = []
    def compute(n, **kw):
        calls.append((n, kw))
        return dict(n=n, label='family_n(%s)' % n, cond=123,
                    status='ok' if n == 2 else 'numerical_failure',
                    det='1.2345678901234567890123456789', comparison={})
    with TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        path = Path(d)/'family.json'
        with patch.object(f, 'beilinson_for_n', side_effect=compute), patch.object(f, 'standard_conductor', side_effect=lambda n: 100+n):
            f.run([2, 3], out=path, max_conductor=102)
            assert [n for n, _ in calls] == [2]
            calls.clear()
            f.run([2, 3], out=path)
            assert [n for n, _ in calls] == [3]
            calls.clear()
            f.run([2, 3], out=path)
            assert [n for n, _ in calls] == [3]
            calls.clear()
            f.run([2], out=path, reg_digits=30)
            assert calls[0][1]['reg_digits'] == 30
            assert json.loads(path.read_text())[0]['det'].endswith('6789')
            calls.clear()
            f.run([3, 2], out=None, order='lex', limit=1)
            assert [n for n, _ in calls] == [2]


def test_family_nonreal_member_runs_at_arbitrary_precision():
    assert not f.is_real_slit_member(2)
    high = f.beilinson_for_n(2, reg_digits=25, with_lvalue=False,
                           max_edge_seconds=30)
    assert high['status'] == 'ok', high
    assert isinstance(high['det'], str)
    assert high['regulator_digits'] == 25
    assert high['edges_attempted'] == 10
    assert high['triple']['pairwise_nonconjugate']
    assert 'L_second_derivative' not in high


def test_family_real_slit_member_uses_detoured_cuts():
    assert f.is_real_slit_member(-4)
    model, h = f.family_polynomial(-4)
    G = 4*model+h*h
    with IntegrationMonitor(max_edge_seconds=30, max_edge_steps=100000) as monitor:
        det, matrix, _ = f.family_determinant(G, reg_digits=20)
    assert det != 0 and len(monitor.records) > 10
    assert all(e['status'] == 'ok' for e in monitor.records)
    assert all(v.parent().precision() > 64 for row in matrix for v in row)


def test_family_numerical_failure_is_reported():
    with patch.object(f.b, 'beilinson_determinant_stokes', side_effect=RuntimeError('quadrature failed')):
        result = f.beilinson_for_n(2, with_lvalue=False)
    assert result['status'] == 'numerical_failure'
    assert 'quadrature failed' in result['error']


def test_family_lvalue_comparison_uses_regulator_precision():
    R = RealField(150)
    det = R('1.23456789012345678901234567890123456789')
    with patch.object(f.b, 'beilinson_determinant_stokes', return_value=(det, [[det]*3]*3, None)), \
         patch.object(f.b, 'c_invariants', return_value=(R(2), R(3))) as periods, \
         patch.object(f.b, 'standard_lvalue_cached', return_value={'L_second_derivative': '0.01', 'L_digits': 40}) as lvalue:
        result = f.beilinson_for_n(2, reg_digits=35, digits=40)
    assert result['status'] == 'ok'
    assert periods.call_args.kwargs['prec'] == 35
    assert lvalue.call_args.kwargs['digits'] == 40
    expected = det/(2*R.pi()*2*3*R('0.01'))
    assert abs(R(result['comparison']['ratio'])-expected) < R('1e-33')
    saved = json.loads(json.dumps(result, default=str))
    assert abs(R(saved['det'])-det) < R('1e-33')


def test_family_reuses_higher_precision():
    result = dict(n=2, label='family_n(2)', cond=123, status='ok',
                  det='1.234567890123456789', comparison={})
    with TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        path = Path(d)/'family.json'
        with patch.object(f, 'beilinson_for_n', return_value=result) as calc:
            f.run([2], out=path, reg_digits=50, digits=50, max_edge_seconds=120)
            saved = path.read_text()
            calc.reset_mock()
            f.run([2], out=path, reg_digits=30, digits=30)
            f.run([2], out=path)
            calc.assert_not_called()
            assert path.read_text() == saved
