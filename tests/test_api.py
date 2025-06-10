from windtunnel.api import Simulator

def test_simulator_init():
    sim = Simulator({'diffusion': {'input_dim': 1, 'timesteps': 5}})
    assert hasattr(sim, 'engine')
    assert sim.engine.timesteps == 5