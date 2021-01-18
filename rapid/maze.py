import numpy as np
import math
from gym import spaces
from gym_miniworld.miniworld import MiniWorldEnv, Room
from gym_miniworld.entity import Box, ImageFrame
from gym_miniworld.params import DEFAULT_PARAMS

class MazeS5(MiniWorldEnv):
    """
    Maze environment in which the agent has to reach a red box
    """

    def __init__(
        self,
        num_rows=5,
        num_cols=5,
        room_size=3,
        max_episode_steps=None,
        **kwargs
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25

        params = DEFAULT_PARAMS
        params.set('forward_step', 0.4)
        params.set('turn_step', 22.5)

        super().__init__(
            max_episode_steps = max_episode_steps or num_rows * num_cols * 24, params=params,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):

                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex='brick_wall',
                    #floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            neighbors = self.rand.subset([(0,1), (0,-1), (-1,0), (1,0)], 4)

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(room, neighbor, min_x=room.min_x, max_x=room.max_x)
                elif dj == 0:
                    self.connect_rooms(room, neighbor, min_z=room.min_z, max_z=room.max_z)

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0)

        X = (self.num_cols - 0.5) * self.room_size + (self.num_cols - 1) * self.gap_size
        Z = (self.num_rows - 0.5) * self.room_size + (self.num_rows - 1) * self.gap_size
        self.box = self.place_entity(Box(color='red'), pos=np.array([X, 0, Z]))

        X = 0.5 * self.room_size
        Z = 0.5 * self.room_size
        self.place_entity(self.agent, pos=np.array([X, 0, Z]), dir=0)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info
