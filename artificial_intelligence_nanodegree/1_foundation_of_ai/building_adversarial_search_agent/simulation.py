from tournament import Agent, play_matches
from sample_players import improved_score
from game_agent import (AlphaBetaPlayer, custom_score,
                        custom_score_2, custom_score_3)


# Define two agents to compare -- these agents will play from the same
# starting position against the same adversaries in the tournament

NUM_MATCHES = 100

benchmark_agent = [
    Agent(AlphaBetaPlayer(score_fn=improved_score), "AB_Improved")
]

test_agents = [
    Agent(AlphaBetaPlayer(score_fn=custom_score), "AB_Custom"),
    Agent(AlphaBetaPlayer(score_fn=custom_score_2), "AB_Custom_2"),
    Agent(AlphaBetaPlayer(score_fn=custom_score_3), "AB_Custom_3")
]


play_matches(benchmark_agent, test_agents, NUM_MATCHES)
