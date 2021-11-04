"""
Plot steady-state solutions for certain example systems with pre-defined parameters using analytic and recurrence
method, as well as models with extrinsic as described in:

L. Ham, D. Schnoerr, R. D. Brackston & M. P. H. Stumpf, Exactly solvable models of stochastic gene expression, J. Chem. Phys. 152, 144106 (2020).

The following models are available for the different methods:

- analytic method ('analytic'):
  * leaky Telegraph model ('leaky')
  * 2^2 model ('twotwo')
  * 2^3 model ('twothree')
- recurrence method ('recurrence'):
    * leaky Telegraph model ('leaky')
    * three switch model ('three_switch')
    * feedback model ('feedback')
- models with extrinsic noise ('extrinsic'):
    * leaky Telegraph model ('leaky')
    * three switch model ('three_switch')
    * 2^2 model ('twotwo')

The string in parentheses can be used to call one of the methods to solve and plot one of the available models from
the command line (as an example the analytic method gets called here for the leaky Telegraph model):

>> python main.py --method='analytic' --model='leaky'

"""



import click

import examples.plot_analytic_solutions as plot_analytic
import examples.plot_extrinsic_solutions as plot_extrinsic
import examples.plot_recurrence_solutions as plot_recurrence

lookup_dict = {
    "analytic": {
        "leaky": plot_analytic.plot_leaky_telegraph,
        "twotwo": plot_analytic.plot_twotwo_multistate,
        "twothree": plot_analytic.plot_twothree_multistate,
    },
    "recurrence": {
        "leaky": plot_recurrence.plot_leaky_telegraph_recurrence,
        "three_switch": plot_recurrence.plot_three_switch_recurrence,
        "feedback": plot_recurrence.plot_feedback_model_recurrence,
    },
    "extrinsic": {
        "leaky": plot_extrinsic.plot_leaky_telegraph_extrinsic,
        "three_switch": plot_extrinsic.plot_three_switch_extrinsic,
        "twotwo": plot_extrinsic.plot_twotwo_multistate_extrinsic,
    },
}


@click.command()
@click.option("--method", default="analytic")
@click.option("--model", default="leaky")
def main(method: str, model: str):
    function_to_run = lookup_dict.get(method).get(model)
    function_to_run()


if __name__ == "__main__":
    main()
