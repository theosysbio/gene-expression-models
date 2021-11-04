import click

import examples.plot_analytic_solutions as plot_analytic
import examples.plot_extrinsic_solutions as plot_extrinsic
import examples.plot_recurrence_solutions as plot_recurrence

lookup_dict = {
    "a": {
        "leaky": plot_analytic.plot_leaky_telegraph,
        "twotwo": plot_analytic.plot_twotwo_multistate,
        "twothree": plot_analytic.plot_twothree_multistate,
    },
    "r": {
        "leaky": plot_recurrence.plot_leaky_telegraph_recurrence,
        "three_switch": plot_recurrence.plot_three_switch_recurrence,
        "feedback": plot_recurrence.plot_feedback_model_recurrence,
    },
    "e": {
        "leaky": plot_extrinsic.plot_leaky_telegraph_extrinsic,
        "three_state": plot_extrinsic.plot_three_state_extrinsic,
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
