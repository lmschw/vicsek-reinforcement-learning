from vicsek_environment import VicsekEnvironment

from pettingzoo.test import parallel_api_test
from pettingzoo.test import render_test

if __name__ == "__main__":
    env = VicsekEnvironment()
    parallel_api_test(env, num_cycles = 1_000_000)

    #render_test(env)

