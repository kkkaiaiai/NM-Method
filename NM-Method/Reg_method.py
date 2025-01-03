import LiteEFG as leg
import pyspiel
import argparse
from tqdm import tqdm

from absl import app
from absl import flags
from open_spiel.python.algorithms import exploitability

import wandb
import sequence_form_algo.mmd_dilated as mmd_dilated
import sequence_form_algo.omwu_dilated as omwu_dilated
import sequence_form_algo.ogda_dilated as ogda_dilated
import sequence_form_algo.gda_dilated as gda_dilated
import sequence_form_algo.mmd_dilated_moving as mmd_dilated_moving
import sequence_form_algo.gda_dilated_moving as gda_dilated_moving
import sequence_form_algo.mommwu_dilated as mommwu_dilated
import sequence_form_algo.MoGDA_dilated as MoGDA_dilated
# import sequence_form_algo.reg_ogda.graph as MoGDA_dilated
import pyspiel
import sys

flags.DEFINE_integer("iterations", 10000, "Number of iterations")
flags.DEFINE_float(
    "alpha", 0.0, "QRE parameter, larger value amounts to more regularization")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("print_freq", 10, "How often to print the gap")
flags.DEFINE_bool("use_wandb",False,
                  "use the policy finetune trick when evaluating.")
flags.DEFINE_string("project_name", "your_project_name", "project name of wandb")


FLAGS = flags.FLAGS
FLAGS(sys.argv)


if (FLAGS.use_wandb):
    wandb.init(
        project=FLAGS.project_name,
        config=FLAGS,
        name="Reg_CFR"
    )

normalize = lambda x: (x - x.min()) / (x.max() - x.min())
# game_rps = np.array([
#     [0, 1, -1, 0, 0], 
#     [-1, 0, 1, 0, 0], 
#     [1, -1, 0, 0, 0], 
#     [1, -1, 0, -2, 1], 
#     [1, -1, 0, 1, -2], 
#     ])
def main(_):
  #
  game = pyspiel.load_game("kuhn_poker")
  # game = pyspiel.load_game("liars_dice(dice_sides=4)")
#   game = pyspiel.load_game("turn_based_simultaneous_game(game=goofspiel(imp_info=True,num_cards=5,players=2,points_order=descending))")
  env =  leg.OpenSpielEnv(game, regenerate=False , traverse_type = "Enumerate") 
  # solver = leg.baselines.Reg_CFR.graph(tau = 1e-7, kappa = 1 ) 
  # solver = leg.baselines.CFRplus.graph()
  solver = leg.baselines.Reg_DOMD.graph(eta = 0.5, tau = 1e-8) 

  env.set_graph(solver)
  for i in range(FLAGS.iterations):
    solver.update_graph(env)
    env.update_strategy(solver.current_strategy())
    policy, _ = env.get_strategy(solver.current_strategy(), "last-iterate")

    if i % FLAGS.print_freq == 0:
      conv = exploitability.nash_conv(game, policy)
      if (FLAGS.use_wandb):
            wandb.log({
                "conv": conv,
                "step": i + 1
            })
      print(conv,i)
      # print(conv , i)

if __name__ == "__main__":
  app.run(main)
