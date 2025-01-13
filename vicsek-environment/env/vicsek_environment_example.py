from vicsek_environment import VicsekEnvironment

env = VicsekEnvironment(render_mode='human')
observations, infos = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render()

env.close()