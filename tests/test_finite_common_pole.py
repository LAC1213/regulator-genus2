"""Regulator weights with arbitrary finite common poles."""
import numpy as np
from hyperell_regulator.experiments import beilinson as b
from hyperell_regulator.regulator import IntegrationMonitor
from hyperell_regulator.regulator.polynomials import ring


# Exact LMFDB equations y^2 + h(x)y = f(x), with coefficients in
# ascending degree. Keep the regression inputs independent of local caches.
CURVE_249 = {"label": "249.a.6723.1",
             "eqn": [[2, 3, 1, 1, 0, -1], [1, 0, 0, 1]]}
CURVE_353 = {"label": "353.a.353.1",
             "eqn": [[0, 0, 1], [1, 1, 0, 1]]}


def test_finite_pole_column_equals_difference_of_infinity_columns():
    x=ring().gen();F=x*(x-1)*(x**3+x+1)
    with IntegrationMonitor(max_edge_seconds=15) as monitor:
        finite=b.regulator_column_stokes(F,(0,0),2,pole=(1,0))
        p=b.regulator_column_stokes(F,(0,0),2)
        r=b.regulator_column_stokes(F,(1,0),2)
    np.testing.assert_allclose(np.asarray(finite,float),np.asarray(p,float)-np.asarray(r,float),atol=1e-8,rtol=0)
    assert all(e['status']=='ok' for e in monitor.records)


def test_model_keeps_conjugate_infinities_distinct_and_affine():
    C=b.curve_from_row(CURVE_249)
    points=[(-1,0),('infinity',0),('infinity',-1)]
    F,images,shift=b.model_with_finite_points(C,points)
    assert all(p!='infinity' for p in images)
    assert images[1][0]==images[2][0]
    assert images[1][1]==-images[2][1] and images[1]!=images[2]
    assert all(y*y==F(x) for x,y in images)


def test_finite_pole_fibre_has_no_fixed_denominator_roots():
    from hyperell_regulator.regulator import function_with_divisor
    from hyperell_regulator.regulator.cover import Cover
    C=b.curve_from_row(CURVE_249)
    F,images,_=b.model_with_finite_points(C,[(-1,0),('infinity',0),('infinity',-1)])
    phi=function_with_divisor(F,images[0],images[2],7)
    cov=Cover(F,*phi)
    assert cov.N==7
    x,y=cov.fibre(1+2j)
    assert len(x)==7
    A,B=phi
    values=np.array([complex(A(z))+complex(B(z))*v for z,v in zip(x,y)])
    np.testing.assert_allclose(values,1+2j,atol=1e-7,rtol=0)
    reciprocal=cov.reciprocal(2)
    assert reciprocal.N==7
    u,v=reciprocal.fibre(1+2j)
    original_x=2+1/u
    original_y=v/u**3
    values=np.array([complex(A(z))+complex(B(z))*yv for z,yv in zip(original_x,original_y)])
    np.testing.assert_allclose(values,1+2j,atol=1e-6,rtol=0)


def test_rank_zero_orders_and_conjugate_zero_weight():
    from hyperell_regulator.regulator import function_with_divisor
    from hyperell_regulator.regulator.cover import Cover
    C=b.curve_from_row(CURVE_353)
    t={'P':(0,0),'Q':('infinity',0),'R':(0,-1),
       'order_P':'rank 0','order_Q':'rank 0','torsion_exponent':11}
    F,images,_,pole=b._model_for_triple(C,t)
    assert pole!='infinity'
    b._resolve_triple_orders(F,images,pole,t)
    assert (t['order_P'],t['order_Q'])==(11,11)
    cov=Cover(F,*function_with_divisor(F,images[0],pole,11))
    assert cov.N==11
    # P and R are conjugate, hence the normalized weight is identically 1.
    assert images[0][0]==pole[0] and images[0][1]==-pole[1]
    num,den=cov.norm_exact()
    assert num.degree()==0 and den.degree()==0
