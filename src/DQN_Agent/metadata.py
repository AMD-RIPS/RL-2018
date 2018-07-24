import sys
sys.dont_write_bytecode = True


class Training_Metadata:

    def __init__(self, frame=0, frame_limit=1000000, episode=0, num_episodes=10000):
        self.frame = frame
        self.frame_limit = frame_limit
        self.episode = episode
        self.num_episodes = num_episodes

    def increment_frame(self):
        self.frame += 1

    def increment_episode(self):
        self.episode += 1
