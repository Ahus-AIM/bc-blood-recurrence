import matplotlib.pyplot as plt
import numpy as np

from src.model.ebm_visualizations import apply_univariate_x_axis_scale




def test_static_change_plot_keeps_linear_axis_even_when_all_values_are_positive() -> None:
    figure, axis = plt.subplots()
    values = np.asarray([1.0, 10.0, 100.0])
    try:
        apply_univariate_x_axis_scale(axis, "A|||u change", values, None, np.ones(3, dtype=bool), "log")
        assert axis.get_xscale() == "linear"
    finally:
        plt.close(figure)
