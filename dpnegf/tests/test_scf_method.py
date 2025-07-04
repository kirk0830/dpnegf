import numpy as np
import pytest
from dpnegf.negf.scf_method import AndersonMixer
from dpnegf.negf.scf_method import BroydenSecondMixer
from dpnegf.negf.scf_method import BroydenFirstMixer
from dpnegf.negf.scf_method import DIISMixer
from dpnegf.negf.scf_method import PDIISMixer

def test_anderson_mixer_linear_mixing_behavior():
    mixer = AndersonMixer(m=3, alpha=0.5, num_linear_warmup=2)
    x0 = np.array([1.0, 2.0])
    f0 = np.array([2.0, 4.0])
    # First update: should use linear mixing
    x1 = mixer.update(f0, x0)
    expected_x1 = x0 + 0.5 * (f0 - x0)
    np.testing.assert_allclose(x1, expected_x1)
    # Second update: still linear mixing
    f1 = np.array([3.0, 6.0])
    x2 = mixer.update(f1, x1)
    expected_x2 = x1 + 0.5 * (f1 - x1)
    np.testing.assert_allclose(x2, expected_x2)

def test_anderson_mixer_switches_to_anderson():
    mixer = AndersonMixer(m=2, alpha=0.1, num_linear_warmup=1)
    x0 = np.array([0.0, 0.0])
    f0 = np.array([1.0, 1.0])
    # First update: linear mixing
    x1 = mixer.update(f0, x0)
    np.testing.assert_allclose(x1, x0 + 0.1 * (f0 - x0))
    # Second update: should switch to Anderson mixing
    f1 = np.array([0.6, 0.4])
    x2 = mixer.update(f1, x1)
    # Anderson mixing should not raise and should return a numpy array of correct shape
    assert isinstance(x2, np.ndarray)
    assert x2.shape == x1.shape
    x2_ = np.array([0.35135135, 0.02702703])
    assert  abs(x2 - x2_).max() < 1e-8

def test_anderson_mixer_reset():
    mixer = AndersonMixer(m=2, alpha=0.3, num_linear_warmup=1)
    x0 = np.array([1.0, 1.0])
    f0 = np.array([2.0, 2.0])
    mixer.update(f0, x0)
    mixer.reset()
    assert mixer.iter == 0
    assert mixer.xkm1 is None
    assert mixer.fkm1 is None
    assert mixer.dx_hist == []
    assert mixer.df_hist == []
    assert mixer.first_linear is True

def test_anderson_mixer_shape_assertion():
    mixer = AndersonMixer()
    x = np.array([1.0, 2.0])
    f = np.array([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError):
        mixer.update(f, x)

def test_broyden_second_mixer_linear_first_step():
    # Test that the first update is equivalent to linear mixing with -alpha
    shape = (2,)
    alpha = 0.3
    mixer = BroydenSecondMixer(shape, alpha=alpha)
    x0 = np.array([1.0, 2.0])
    f0 = np.array([0.5, -1.0])
    x1 = mixer.update(x0, f0)
    expected = x0 + alpha * f0
    np.testing.assert_allclose(x1, expected)

def test_broyden_second_mixer_second_step_update():
    # Test that the second update uses the Broyden formula and returns correct shape
    shape = (2,)
    alpha = 0.2
    mixer = BroydenSecondMixer(shape, alpha=alpha)
    x0 = np.array([0.0, 0.0])
    f0 = np.array([1.0, -1.0])
    x1 = mixer.update(x0, f0)
    x2 = mixer.update(x1, np.array([0.5, -0.5]))
    x2_ = np.array([0.4, -0.4])  # Expected value after second update
    assert isinstance(x2, np.ndarray)
    assert x2.shape == x0.shape
    np.testing.assert_allclose(x2, x2_)

def test_broyden_second_mixer_reset():
    shape = (3,)
    mixer = BroydenSecondMixer(shape, alpha=0.1)
    x0 = np.array([1.0, 2.0, 3.0])
    f0 = np.array([0.1, 0.2, 0.3])
    mixer.update(x0, f0)
    mixer.reset(shape)
    assert mixer.iter == 0
    np.testing.assert_array_equal(mixer.x_last, np.zeros(shape))
    np.testing.assert_array_equal(mixer.f_last, np.zeros(shape))
    assert isinstance(mixer.H0, np.ndarray)
    assert mixer.df_hist == []

def test_broyden_second_mixer_multiple_iterations():
    # Test that the mixer can run for several iterations without error
    shape = (2,)
    mixer = BroydenSecondMixer(shape, alpha=0.15)
    x = np.array([0.0, 0.0])
    xlist_ = np.array([[0.15,-0.15],
                       [1.5,-1.5],
                       [12.3,-12.3],
                       [87.8999999,-87.8999999],
                       [541.4999999,-541.4999999]])
    for i in range(5):
        f = np.array([1.0 - 0.1 * i, -1.0 + 0.1 * i])
        x = mixer.update(x, f)
        assert x.shape == (2,)
        assert np.allclose(x, xlist_[i])


def test_broyden_first_mixer_linear_warmup():
    # Test that the first three updates use linear mixing
    init_x = np.array([1.0, 2.0])
    alpha = 0.2
    mixer = BroydenFirstMixer(init_x, alpha=alpha)
    f0 = np.array([0.5, -1.0])
    # First update: should use init_x + alpha * f
    x1 = mixer.update(f0)
    np.testing.assert_allclose(x1, init_x + alpha * f0)
    # Second update: should use x_n + alpha * f
    f1 = np.array([1.0, 1.0])
    x2 = mixer.update(f1)
    np.testing.assert_allclose(x2, x1 + alpha * f1)
    # Third update: should use x_n + alpha * f
    f2 = np.array([-0.5, 0.5])
    x3 = mixer.update(f2)
    np.testing.assert_allclose(x3, x2 + alpha * f2)

def test_broyden_first_mixer_switches_to_broyden():
    # After three iterations, should use Broyden's update
    init_x = np.array([0.0, 0.0])
    alpha = 0.1
    mixer = BroydenFirstMixer(init_x, alpha=alpha)
    f0 = np.array([1.0, 2.0])
    x1 = mixer.update(f0)
    np.testing.assert_allclose(x1, init_x + alpha * f0)
    f1 = np.array([0.5, 1.5])
    x2 = mixer.update(f1)
    np.testing.assert_allclose(x2, x1 + alpha * f1)
    f2 = np.array([0.2, 1.0])
    x3 = mixer.update(f2)
    np.testing.assert_allclose(x3, x2 + alpha * f2)
    # Now, Broyden's update should be used
    f3 = np.array([0.1, 0.5])
    x4 = mixer.update(f3)
    x4_ = np.array([0.19, 0.55])
    # Check that the shape is correct and no error is raised
    assert isinstance(x4, np.ndarray)
    assert x4.shape == init_x.shape
    np.testing.assert_allclose(x4, x4_)

def test_broyden_first_mixer_reset():
    init_x = np.array([1.0, 2.0, 3.0])
    mixer = BroydenFirstMixer(init_x, alpha=0.3)
    f = np.array([0.1, 0.2, 0.3])
    mixer.update(f)
    mixer.reset(init_x.shape)
    assert mixer.iter == 0
    np.testing.assert_array_equal(mixer.x_n, np.zeros(init_x.shape))
    np.testing.assert_array_equal(mixer.x_nm1, np.zeros(init_x.shape))
    assert mixer.J0.shape == (3, 3)
    assert mixer.J_inv.shape == (3, 3)

def test_broyden_first_mixer_multiple_iterations():
    # Run several iterations and check for shape and no errors
    init_x = np.array([0.0, 0.0])
    mixer = BroydenFirstMixer(init_x, alpha=0.15)
    x = init_x.copy()
    x_last = np.array([ 1644.28499959, -1644.28499959])
    for i in range(10):
        f = np.array([1.0 - 0.1 * i, -1.0 + 0.1 * i])
        x = mixer.update(f)
        assert x.shape == (2,)
        if i == 9:
            np.testing.assert_allclose(x, x_last)
 

def test_broyden_first_mixer_handles_zero_residual():
    # Should not crash if residual is zero
    init_x = np.array([1.0, 2.0])
    mixer = BroydenFirstMixer(init_x, alpha=0.2)
    f = np.zeros_like(init_x)
    x1 = mixer.update(f)
    x2 = mixer.update(f)
    x3 = mixer.update(f)
    x4 = mixer.update(f)
    np.testing.assert_allclose(x4, x3)


def test_diis_mixer_linear_mixing_warmup():
    mixer = DIISMixer(max_hist=3, alpha=0.5, linear_warmup=2)
    x0 = np.array([1.0, 2.0])
    r0 = np.array([0.5, 1.0])
    # First update: should use linear mixing
    x1 = mixer.update(x0, r0)
    expected_x1 = (x0 - r0) + 0.5 * r0
    np.testing.assert_allclose(x1, expected_x1)
    # Second update: still linear mixing
    x2 = mixer.update(x1, r0)
    expected_x2 = (x1 - r0) + 0.5 * r0
    np.testing.assert_allclose(x2, expected_x2)

def test_diis_mixer_switches_to_diis():
    mixer = DIISMixer(max_hist=3, alpha=0.2, linear_warmup=1)
    x0 = np.array([1.0, 2.0])
    r0 = np.array([0.5, 1.0])
    x1 = mixer.update(x0, r0)  # linear mixing
    x2 = mixer.update(x1, r0)  # should switch to DIIS
    # Should return a numpy array of correct shape
    assert isinstance(x2, np.ndarray)
    assert x2.shape == x0.shape
    x2_ = np.array([0.2, 0.4])  # Expected value after DIIS mixing
    assert abs(x2 - x2_).max() < 1e-8

def test_diis_mixer_history_limit():
    mixer = DIISMixer(max_hist=2, alpha=0.3, linear_warmup=1)
    x = np.array([1.0, 2.0])
    r = np.array([0.5, 1.0])
    # Fill up history
    mixer.update(x, r)
    mixer.update(x, r)
    mixer.update(x, r)
    # Should not exceed max_hist
    assert len(mixer.x_hist) <= 2
    assert len(mixer.r_hist) <= 2

def test_diis_mixer_reset():
    mixer = DIISMixer(max_hist=2, alpha=0.3, linear_warmup=1)
    x = np.array([1.0, 2.0])
    r = np.array([0.5, 1.0])
    mixer.update(x, r)
    mixer.reset()
    assert mixer.iter == 0
    assert mixer.x_hist == []
    assert mixer.r_hist == []

def test_diis_mixer_fallback_on_singular_matrix(monkeypatch):
    mixer = DIISMixer(max_hist=2, alpha=0.1, linear_warmup=0)
    x = np.array([1.0, 2.0])
    r = np.array([0.5, 1.0])
    # Fill up history so that B will be singular
    mixer.update(x, r)
    mixer.update(x, r)
    # Monkeypatch np.linalg.solve to raise LinAlgError
    monkeypatch.setattr(np.linalg, "solve", lambda *a, **k: (_ for _ in ()).throw(np.linalg.LinAlgError))
    x_new = mixer.update(x, r)
    # Should fall back to linear mixing
    expected = (x - r) + 0.1 * r
    np.testing.assert_allclose(x_new, expected)

def test_diis_mixer_shape_and_type():
    mixer = DIISMixer(max_hist=3, alpha=0.2, linear_warmup=1)
    x = np.array([1.0, 2.0, 3.0])
    r = np.array([0.1, 0.2, 0.3])
    out = mixer.update(x, r)
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape

def test_pdiis_mixer_linear_warmup_behavior():
    init_x = np.array([1.0, 2.0])
    mixer = PDIISMixer(init_x, mix_rate=0.5, max_hist=3, mixing_period=2)
    g0 = np.array([2.0, 4.0])
    # First update: linear mixing
    x1 = mixer.update(g0)
    expected_x1 = init_x + 0.5 * (g0 - init_x)
    np.testing.assert_allclose(x1, expected_x1)
    # Second update: still linear mixing
    g1 = np.array([3.0, 6.0])
    x2 = mixer.update(g1)
    expected_x2 = x1 + 0.5 * (g1 - x1)
    np.testing.assert_allclose(x2, expected_x2)

def test_pdiis_mixer_switches_to_pdiis_and_periodic():
    init_x = np.array([0.0, 0.0])
    mixer = PDIISMixer(init_x, mix_rate=0.2, max_hist=2, mixing_period=2)
    g0 = np.array([1.0, 1.0])
    x1 = mixer.update(g0)  # linear mixing
    g1 = np.array([0.5, 0.5])
    x2 = mixer.update(g1)  # linear mixing
    g2 = np.array([0.2, 0.4])
    x3 = mixer.update(g2)  # should use PDIIS (since iter_count == 3)
    # Should return a numpy array of correct shape
    assert isinstance(x3, np.ndarray)
    assert x3.shape == init_x.shape

def test_pdiis_mixer_history_limit():
    init_x = np.array([1.0, 2.0])
    mixer = PDIISMixer(init_x, mix_rate=0.3, max_hist=2, mixing_period=2)
    g = np.array([2.0, 3.0])
    # Fill up history
    mixer.update(g)
    mixer.update(g)
    mixer.update(g)
    mixer.update(g)
    # Should not exceed max_hist
    assert len(mixer.F) <= 2
    assert len(mixer.R) <= 2

def test_pdiis_mixer_reset():
    init_x = np.array([1.0, 2.0])
    mixer = PDIISMixer(init_x, mix_rate=0.3, max_hist=2, mixing_period=2)
    g = np.array([2.0, 3.0])
    mixer.update(g)
    mixer.reset()
    assert mixer.iter_count == 1
    assert mixer.x is None or isinstance(mixer.x, np.ndarray)
    assert mixer.f is None
    assert mixer.R == []
    assert mixer.F == []

def test_pdiis_mixer_shape_assertion():
    init_x = np.array([1.0, 2.0])
    mixer = PDIISMixer(init_x)
    g = np.array([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError):
        mixer.update(g)

def test_pdiis_mixer_fallback_on_singular_matrix(monkeypatch):
    init_x = np.array([1.0, 2.0])
    mixer = PDIISMixer(init_x, mix_rate=0.1, max_hist=2, mixing_period=3)
    g = np.array([2.0, 3.0])
    x1 = mixer.update(g)
    x2 = mixer.update(g)
    # Monkeypatch np.linalg.solve to raise LinAlgError
    monkeypatch.setattr(np.linalg, "solve", lambda *a, **k: (_ for _ in ()).throw(np.linalg.LinAlgError))
    # Third update triggers PDIIS and should fallback to linear mixing
    x_new = mixer.update(g)
    expected = mixer.x_last.ravel() + 0.1 * (g - mixer.x_last.ravel())
    np.testing.assert_allclose(x_new, expected.ravel())

def test_pdiis_mixer_output_shape_and_type():
    init_x = np.array([1.0, 2.0, 3.0])
    mixer = PDIISMixer(init_x, mix_rate=0.2, max_hist=2, mixing_period=2)
    g = np.array([2.0, 3.0, 4.0])
    out = mixer.update(g)
    assert isinstance(out, np.ndarray)
    assert out.shape == init_x.shape






