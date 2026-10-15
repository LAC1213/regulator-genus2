"""Precision growth, exact coefficients, and full-precision report regressions."""
import json
from unittest.mock import patch
from sage.all import QQ, RealField
from hyperell_regulator.regulator import periods_and_regulator_stokes, c_invariants, IntegrationMonitor
from hyperell_regulator.regulator.polynomials import ring
from hyperell_regulator.experiments import beilinson as b


def test_exact_chart_center_does_not_round_through_float64():
    from hyperell_regulator.regulator._mp_continuation import Arithmetic,Chart
    from hyperell_regulator.regulator.cover import Cover
    x=ring().gen();q=QQ(1)/3;F=(x-q)**5+(1+x-q)**2
    ar=Arithmetic(35);ch=Chart(Cover(F,x-q+1,1),ar,center=q)
    assert ch.F[0]==1 and ch.F[1]==2 and ch.F[5]==1
    assert ch.a[0]==1
    inverse=Chart(Cover(F,x-q+1,1),ar,kind='inv',center=q)
    assert inverse.F[0]==0 and inverse.F[1]==1
    assert inverse.F[4]==1 and inverse.F[5]==2 and inverse.F[6]==1


def test_report_preserves_decimal_digits_and_cli_precision():
    x=ring().gen();F=x**5+x+1;R=RealField(150)
    value=R('1.23456789012345678901234567890123456789')
    triple=dict(P=(0,1),Q=(1,1),R=('infinity',1),order_P=5,order_Q=5)
    with patch.object(b,'_model_for_triple',return_value=(F,[(0,1),(1,1)],None,'infinity')),patch.object(b,'_resolve_triple_orders'),patch.object(b,'beilinson_determinant_stokes',return_value=(value,[[value]*3]*3,None)):
        out=b._analyse_triple({'label':'test','cond':1},None,triple,1,
            verbose=False,reg_digits=30,with_periods=False,with_lvalue=False)
    saved=json.loads(json.dumps(out,default=str))
    assert isinstance(saved['det'],str)
    assert abs(R(saved['det'])-value)<R('1e-30')
    assert abs(R(saved['matrix'][0][0])-value)<R('1e-30')
    with patch.object(b,'run',return_value=[]) as run:
        b.main(['250','--prec','30','--edge-seconds','45'])
    assert run.call_args.kwargs['reg_digits']==30
    assert run.call_args.kwargs['max_edge_seconds']==45


def test_lvalue_cache_retains_precision_and_rejects_old_float_records():
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from unittest.mock import Mock
    R=RealField(140);value=R('2.345678901234567890123456789012345')
    lf=Mock()
    lf._coeffs_needed.return_value=100
    lf.conductor.return_value=249
    lf.check_functional_equation.return_value=R('1e-40')
    lf.value.return_value=value
    with TemporaryDirectory() as d:
        path=Path(d)/'L.json'
        path.write_text(json.dumps({'curve':dict(L_second_derivative=float(value),
            L_pari_precision=True,L_digits=30,L_num_coeffs=100)}))
        with patch.object(b,'StandardLFunction',return_value=lf):
            result=b.standard_lvalue_cached('curve',None,digits=30,path=path,progress=False)
            assert lf.value.call_count==1
            assert abs(R(result['L_second_derivative'])-value)<R('1e-35')
            cached=b.standard_lvalue_cached('curve',None,digits=30,path=path,progress=False)
            assert cached['L_cached'] and lf.value.call_count==1


def test_regular_inverse_coordinate_zero_is_crossed():
    from hyperell_regulator.regulator import function_with_divisor
    from hyperell_regulator.regulator.cover import Cover
    from hyperell_regulator.regulator._mp_continuation import Arithmetic
    from hyperell_regulator.regulator._mp_stokes import Integrator
    x=ring().gen();F=x**6+4*x**5+4*x**4+2*x**3+1
    cov=Cover(F,*function_with_divisor(F,(0,-1),'infinity',14))
    ar=Arithmetic(20);engine=Integrator(cov,ar)
    a,b=ar.C('-2.05'),ar.C('-1.95')
    points=engine.fibre(a);lam=a.log()
    with IntegrationMonitor(max_edge_seconds=20,max_edge_steps=100000) as monitor:
        with monitor.edge('cross finite-value infinity',degree=14):
            ends,ll,I,L=engine.finite(a,b,points,lam)
        with monitor.edge('reverse finite-value infinity',degree=14):
            back,lb,Ib,Lb=engine.finite(b,a,ends,ll)
    for p,q in zip(points,back):
        for v,w in zip(p,q):assert abs(v-w)<ar.R('1e-25')*max(1,abs(v))
    assert abs(lb-lam)<ar.R('1e-25')
    for forward,reverse in ((I,Ib),(L,Lb)):
        for row,rev in zip(forward,reverse):
            assert abs(sum(row))<ar.R('1e-23')
            for v,w in zip(row,rev):assert abs(v+w)<ar.R('1e-23')


def test_709_periods_continue_across_principal_sqrt_cut():
    x=ring().gen();F=4*x**5-7*x**4+4*x
    cp,cm=c_invariants(F,prec=35)
    assert abs(cp-cp.parent()('-2.2951452040570181850466376396996508'))<cp.parent()('1e-32')
    for lo,hi in zip(c_invariants(F),(cp,cm)):
        assert abs(lo-float(hi))<1e-10
