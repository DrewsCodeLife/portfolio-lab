"""
This is the main script file for portfolio lab.

|-----------| Ez access |-----------|
| STRING WIDGET |  |  INPUT WIDGET  |
us_equities_main     , us_equities_input     , us_equities_participation
int_equities_main    , int_equities_input    , int_equities_participation
fixed_income_main    , fixed_income_input    , fixed_income_participation
real_assets_main     , real_assets_input     , real_assets_participation
cash_equivalents_main, cash_equivalents_input, cash_equivalents_participation
"""


import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import tkinter as tk
import pandas as pd
import numpy as np
import cvxpy as cp
import json


from sklearn.covariance import LedoitWolf
from contextlib import contextmanager
from scipy.linalg import solve
from pathlib import Path
from math import floor



np.random.seed(42)

@contextmanager
def use_parent(tag):
    dpg.push_container_stack(tag)
    try:
        yield
    finally:
        dpg.pop_container_stack()

# |--------------------------------- Dev Toggles ----------------------------------|
# Used for dev, should be False in releases
SHOW_DEMO = False

DEBUG_OUTPUT = True
VERBOSE_DEBUG_OUTPUT = True
# |--------------------------------- Dev Toggles ----------------------------------|


# |---------------------------------- Constants ----------------------------------|
# Card padding constants
CARD_OUT_X_PAD = 0.02   # padding L / R
CARD_OUT_Y_PAD = 0.15   # Padding top / bottom
CARD_GAP_X     = 0.015  # Between card padding L / R
CARD_GAP_Y     = 0.08   # Between card padding top / bottom

CARD_ROUNDING     = 0.06  # rounding relative to card size
CARD_INTERNAL_PAD = 5     # Padding between card content and edge
CARD_SIZE         = 0.25  # Relative to window size

SIM_WINDOW_SIZE = 0.75
WINDOW_GAP      = 5

# Text size macros
SMALL_TEXT_SIZE  = 12.0
MEDIUM_TEXT_SIZE = 20.0
LARGE_TEXT_SIZE  = SMALL_TEXT_SIZE * 2
HEADER_TEXT_SIZE = SMALL_TEXT_SIZE * 3

SMALL_TEXT_PAD  = SMALL_TEXT_SIZE * 2
MEDIUM_TEXT_PAD = MEDIUM_TEXT_SIZE * 2
LARGE_TEXT_PAD  = LARGE_TEXT_SIZE * 2

TRADING_DAYS = 252

USER = (66, 135, 245)      # bright blue
REC  = (46, 204, 113)      # bright green
MIN  = (120, 140, 170)     # muted slate
MAX  = (230, 160, 60)      # muted orange
# |---------------------------------- Constants ----------------------------------|


# |---------------------------------- App Class ----------------------------------|
class PortfolioLab:
    # |------------------------------ Internal Model ---------------------------------|
    def __init__(self):
        # |------------------------------ Internal Model ---------------------------------|
        self.CWD = Path(__file__).parent.parent
        self.FONT_FOLDER = self.CWD / "font"

        self.data    = pd.read_csv(self.CWD / "data" / "cleaned" / "cleaned.csv")
        self.returns = pd.read_csv(self.CWD / "data" / "cleaned" / "asset_daily_returns.csv")

        self.dates = self.data["Date"]

        self.data = self.data.loc[:, self.data.columns != "Date"]
        self.returns = self.returns.loc[:, self.returns.columns != "Date"]
        self.returns = self.returns.iloc[:, 1:]  # first column is unnamed idx

        # First day has no 'daily return', so we drop it
        self.data = self.data.iloc[1:].reset_index(drop=True)

        self.T = len(self.returns)
        self.N = self.returns.shape[1]

        self.asset_names = self.returns.columns
        portfolio_dicts = self._build_portfolio_dicts()

        # Initialize to 0, should be updated by _compute_portfolio_layout()
        self.screen_width  = 0
        self.screen_height = 0

        self.asset_map = {}
        for i, asset_name in enumerate(self.asset_names):
            self.asset_map[f"asset_{i}"] = asset_name

        self.starting_value = 100.0

        self.state = {
            **portfolio_dicts,
            "simulated_portfolios": {
                "user_portfolio": 100,
                "rec_portfolio" : 100,
                "min_portfolio" : 100,
                "max_portfolio" : 100,
            },
            'portfolios_over_time'  : None,
            "min_risk"              : [],
            "max_risk"              : [],
            "portfolio_value"       : self.starting_value,
            "lookback"              : 252,
            "mu"                    : None, # daily
            "R_max"                 : None, # daily
            "sigma"                 : None, # daily
            "req_return_daily"      : None, # Converted to daily from UI annualized request
            "eps"                   : 1e-12,
            "w_min"                 : 0.01,
            "sim_period"            : 252 - self.T
        }
        # |------------------------------ Internal Model ---------------------------------|


        # |----------------------------- Window Initialization -----------------------------|
        # Drawing the window means we need fonts
        with dpg.font_registry():
            title_font = dpg.add_font(self.FONT_FOLDER / "League_Gothic" / "static" / "LeagueGothic-Regular.ttf", LARGE_TEXT_SIZE)
            body_font = dpg.add_font(self.FONT_FOLDER / "Open_Sans" / "static" / "OpenSans_SemiCondensed-Regular.ttf", MEDIUM_TEXT_SIZE)
            sim_body_font = dpg.add_font(self.FONT_FOLDER / "Open_Sans" / "static" / "OpenSans_Condensed-Regular.ttf", MEDIUM_TEXT_SIZE)
            numbers_font = dpg.add_font(self.FONT_FOLDER / "Space_Mono" / "SpaceMono-Regular.ttf", MEDIUM_TEXT_SIZE)

        dpg.bind_font(body_font)

        self.fonts = {
            "title"     : title_font,
            "body"      : body_font,
            "sim_body"  : sim_body_font,
            "numbers"   : numbers_font,
            }

        with dpg.theme() as self.transparent_child_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                # Set background to transparent
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                # Remove border
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 0, category=dpg.mvThemeCat_Core)

        # We've loaded the data and lookback defaults to 252, so this should be safe.
        #   We're going to want the initial values stored so we can clamp annualized returns
        self._estimate_mu_sigma()

        # Initialize the desired return to 50% of theoretical max
        self._update_desired_ret(sender=None, app_data=0.5)

        self._try_load_state()
        # |----------------------------- Window Initialization -----------------------------|


    # |---------------------------------- Initialization ----------------------------------|
    def _build_portfolio_dicts(self, starting_value=100.0):
        equal_weight = 1.0 / self.N
        equal_percent = 100.0 / self.N

        return {
            "rec_portfolio": {name: equal_weight for name in self.asset_names},
            "user_portfolio": {name: equal_weight * starting_value for name in self.asset_names},
            "user_portfolio_fractional": {name: equal_weight for name in self.asset_names},
        }

    def _try_load_state(self):
        """
        Attempt to load the autosaved app_state.json, which may have been automatically
            saved from a prior optimization
        """
        try:
            with open(self.CWD / "app_state.json", "r") as f:
                loaded_data = json.load(f)

                # |------------------------------- Load Lookback -------------------------------|
                self.state['lookback'] = lookback = loaded_data['lookback']
                # sim_max = floor((self.T - lookback) / 252)
                # dpg.set_value("sim_period", sim_max)
                # dpg.configure_item("sim_period", max_value=sim_max, label=f"Simulation Period [1, {sim_max}] (years)")
                self._estimate_mu_sigma()
                # |------------------------------- Load Lookback -------------------------------|


                # |-------------------------- Redo Prior Optimization --------------------------|
                self.state['user_portfolio_fractional'] = loaded_data['user_portfolio_fractional']
                self.state['user_portfolio'] = loaded_data['user_portfolio']
                self.state['portfolio_value'] = loaded_data['portfolio_value']

                self.state['req_return_daily'] = loaded_data['req_return_daily']
                self.state['min_risk'] = list(loaded_data['min_risk'])
                # |-------------------------- Redo Prior Optimization --------------------------|

                self.loaded_successfully = True
        except Exception as e:
            self.loaded_successfully = False
            if VERBOSE_DEBUG_OUTPUT:
                print(f"Failed to load CWD/app_state.json: {e}")
    # |---------------------------------- Initialization ----------------------------------|


    # |----------------------------- Window Initialization -----------------------------|
    def _build_error_modal(self):
        with dpg.window(
            label="Load Error",
            tag="error_modal",
            modal=True,
            show=False,
            no_resize=True,
            no_collapse=True,
            width=420,
            height=140,
        ):
            dpg.add_text("Something went wrong.", tag="error_modal_text", wrap=380)
            dpg.add_spacer(height=10)
            dpg.add_button(label="OK", callback=lambda: dpg.hide_item("error_modal"))

    def _show_error(self, message):
        dpg.set_value("error_modal_text", message)
        dpg.show_item("error_modal")

    def _compute_portfolio_layout(self):
        width, height = dpg.get_item_rect_size("portfolio_window")

        self.grid_cols = 5
        self.grid_rows = 2

        self.gap_x       = int(width * CARD_GAP_X)
        self.gap_y       = int(height * CARD_GAP_Y)
        self.outer_pad_x = int(width * CARD_OUT_X_PAD)
        self.outer_pad_y = int(height * CARD_OUT_Y_PAD)

        # self.gap_x between each card, self.outer_pad_x between cards and window edge
        available_width = width - (self.gap_x * 4) - (self.outer_pad_x * 2)

        self.card_w = self.screen_width * (1 / 3) - WINDOW_GAP - 5
        self.card_h = int(height / 11.0) + 3

        self.rounding = int(min(self.card_w, self.card_h) * CARD_ROUNDING)

    def _get_card_pos(self, idx):
        idx = idx[6]
        row = int(idx) // self.grid_cols
        col = int(idx) % self.grid_cols
        x = self.outer_pad_x + col * (self.card_w + self.gap_x)
        y = self.outer_pad_y + row * (self.card_h + self.gap_y)
        return x, y

    def _build_portfolio_window(self):
        self._compute_portfolio_layout()

        with use_parent("portfolio_window"):
            with dpg.group(horizontal=False):
                for tag, asset in self.asset_map.items():
                    pf_val = self.state['portfolio_value']

                    in_val = self.state['user_portfolio'][asset]
                    rec_val = pf_val * self.state['rec_portfolio'][asset]

                    frac_in_val = self.state['user_portfolio_fractional'][asset]
                    frac_rec_val = self.state['rec_portfolio'][asset]

                    drift, high = self._overunder(frac_in_val, frac_rec_val)

                    if high is None:
                        # Balanced (+/- 5%)
                        participation = f"Asset Class Balanced!"
                        pass
                    elif high:
                        # More than 5% over contributed
                        participation = f"Overparticipating: {drift:.3f}%"
                        pass
                    else:
                        # Less then 95% under contributed
                        participation = f"Underparticipating: {drift:.3f}%"

                    with dpg.child_window(
                        tag=f"{tag}_window",
                        width=self.screen_width * (1 / 3) - WINDOW_GAP - 5,
                        height=self.card_h,
                        no_scrollbar=True,
                        no_scroll_with_mouse=True,
                    ) as child_window:
                        child_height = self.card_h - 15

                        with dpg.group(horizontal=True):
                            y_offset = ((self.card_h - LARGE_TEXT_SIZE) / 3)

                            with dpg.child_window(
                                width=50,
                                height=child_height,
                                no_scrollbar=True,
                                no_scroll_with_mouse=True,
                            ) as box_1:
                                # 6 Derived experimentally, centers "VBMFX"
                                dpg.add_text(
                                    pos=(6, y_offset),
                                    default_value=asset,
                                    tag=f"{tag}_title"
                                )

                            # These guys are already centered 'cuz god is just.
                            with dpg.child_window(
                                width=(self.card_w - 50) / 3,
                                height=child_height,
                                no_scrollbar=True,
                                no_scroll_with_mouse=True,
                            ) as box_2:
                                with dpg.group(horizontal=False):
                                    with dpg.group(horizontal=True) as in_group:
                                        dpg.add_text(
                                            default_value=f"IN: $"
                                        )
                                        dpg.add_text(
                                            default_value=f"{in_val:.2f}",
                                            tag=f"{tag}_main_1"
                                        )
                                        dpg.add_text(
                                            default_value=" / ",
                                        )
                                        dpg.add_text(
                                            default_value=f"{(frac_in_val * 100):.2f}",
                                            tag=f"{tag}_main_2"
                                        )
                                        dpg.add_text(
                                            default_value="%"
                                        )

                                    with dpg.group(horizontal=True) as rec_group:
                                        dpg.add_text(
                                            default_value=f"REC: $"
                                        )
                                        dpg.add_text(
                                            default_value=f"{rec_val:.2f}",
                                            tag=f"{tag}_rec_1",
                                        )
                                        dpg.add_text(
                                            default_value=" / "
                                        )
                                        dpg.add_text(
                                            default_value=f"{(frac_rec_val * 100):.2f}",
                                            tag=f"{tag}_rec_2"
                                        )
                                        dpg.add_text(
                                            default_value="%"
                                        )

                            y_offset = (self.card_h - MEDIUM_TEXT_SIZE) / 3
                            with dpg.child_window(
                                width=(self.card_w - 125) / 2,
                                height=child_height,
                                no_scrollbar=True,
                                no_scroll_with_mouse=True,
                            ) as box_3:
                                dpg.add_text(
                                    pos=(5, y_offset),
                                    default_value=participation,
                                    tag=f"{tag}_participation"
                                )

                            y_offset = y_offset - 1  # It's a little too low otherwise
                            with dpg.child_window(
                                height=child_height,
                                no_scrollbar=True,
                                no_scroll_with_mouse=True,
                            ) as box_4:
                                dpg.add_input_double(
                                    pos=(12.5, y_offset),
                                    tag=f"{tag}_input",
                                    label="",
                                    # width=int(self.card_w * 0.04),
                                    step=0,
                                    step_fast=0,
                                    default_value=in_val,
                                    min_value=0.0,
                                    callback=self._update_card_states
                                )

                            dpg.bind_item_theme(box_1, self.fg_theme)
                            dpg.bind_item_theme(box_2, self.fg_theme)
                            dpg.bind_item_theme(box_3, self.fg_theme)
                            dpg.bind_item_theme(box_4, self.fg_theme)

                            with dpg.theme() as tight_text_theme:
                                with dpg.theme_component(dpg.mvAll):
                                    dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 2, 0)

                            dpg.bind_item_theme(in_group, tight_text_theme)
                            dpg.bind_item_theme(rec_group, tight_text_theme)

                            dpg.bind_item_font(f"{tag}_title", self.fonts["title"])
                            dpg.bind_item_font(f"{tag}_main_1", self.fonts["body"])
                            dpg.bind_item_font(f"{tag}_main_2", self.fonts["body"])
                            dpg.bind_item_font(f"{tag}_rec_1", self.fonts["body"])
                            dpg.bind_item_font(f"{tag}_rec_2", self.fonts["body"])
                            dpg.bind_item_font(f"{tag}_participation", self.fonts["body"])

                    # with dpg.theme() as child_theme:
                    #     with dpg.theme_component(dpg.mvChildWindow):
                    #         # Change background color (RGBA)
                    #         dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (35, 35, 35, 255))
                    dpg.bind_item_theme(child_window, self.mid_theme)


    def _build_sim_window(self):
        with use_parent('simulation_window'):
            with dpg.group(tag="sim_group"):
                with dpg.child_window(
                    width=-1,
                    height=self.screen_height * SIM_WINDOW_SIZE,
                    no_scroll_with_mouse=True,
                    no_scrollbar=True,
                    border=True,
                    tag='sim_plot_window'
                    ):
                    with dpg.plot(
                        label="Simulation Plot",
                        width=-1,
                        height=self.screen_height * SIM_WINDOW_SIZE - 60,
                        tag="sim_plot"
                        ): # width=-1 auto-expands
                        dpg.add_plot_legend()

                        dpg.add_plot_axis(dpg.mvXAxis, label="Trading Days", tag="sim_plot_x_axis", auto_fit=True)
                        dpg.add_plot_axis(dpg.mvYAxis, label="Value ($)", tag="sim_plot_y_axis", auto_fit=True)

                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        # dpg.add_spacer(width=35)

                        with dpg.file_dialog(
                            directory_selector=False,
                            show=False,
                            callback=self._load_portfolio,
                            tag="load_dialog",
                            width=700,
                            height=400
                        ):
                            dpg.add_file_extension(".json", color=(0, 255, 0, 255))

                        with dpg.file_dialog(
                            directory_selector=False,
                            show=False,
                            callback=self._save_portfolio,
                            tag="save_dialog",
                            width=700,
                            height=400
                        ):
                            dpg.add_file_extension(".json", color=(0, 255, 0, 255))

                        width = dpg.get_item_width('simulation_window')
                        left_pad = ((width - 4 * 125 - 4*50) / 2) - 15

                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=left_pad)
                            dpg.add_button(
                                width=125,
                                label="Save Portfolio",
                                callback=lambda: dpg.show_item("save_dialog")
                            )
                            dpg.add_spacer(width=50)
                            dpg.add_button(
                                width=125,
                                label="Load Portfolio",
                                callback=lambda: dpg.show_item("load_dialog"),
                            )

                            dpg.add_spacer(width=50)

                            dpg.add_button(
                                width=125,
                                label="Clear Portfolio",
                                callback=self._clear_portfolio
                            )
                            dpg.add_spacer(width=50)
                            dpg.add_button(
                                width=125,
                                label="Optimize Portfolio",
                                callback=self._construct_portfolio
                            )

                with dpg.child_window(
                    tag='sim_slider_window',
                    no_scroll_with_mouse=True,
                    no_scrollbar=True,
                ):
                    dpg.add_spacer(height=10)
                    # Here we can define the manual data entry
                    dpg.add_slider_int(
                        label="Lookback period (days)",
                        tag="lookback",
                        min_value=252,
                        max_value=(self.T - 252),
                        callback=self._lookback_update,
                        user_data=self.T,
                        default_value=252,
                        # indent=100
                        )
                    dpg.add_slider_int(
                        label=f"Simulation Period (years)",
                        tag="sim_period",
                        min_value=1,
                        max_value=floor(self.T / 252),
                        user_data=self.T,
                        default_value=1,
                        callback=self._sim_period_update,
                        # indent=100
                        )

                    dpg.add_slider_float(
                        label="Low -> High Risk",
                        tag="desired_ret",
                        min_value=0.05,
                        max_value=1.0,
                        default_value=0.5,
                        callback=self._update_desired_ret,
                        # indent=100
                        )
                    dpg.add_slider_double(
                        label="Model Minimum Investment",
                        tag="model_min_investment",
                        min_value=0.00,
                        max_value=0.2,
                        default_value=0.01,
                        callback=self._update_min_investment,
                        # indent=100
                    )
                    dpg.add_slider_double(
                        label="Model Return Margin",
                        tag="model_return_margin",
                        min_value=1e-12,
                        max_value=1e-3,
                        default_value=1e-12,
                        clamped=True,
                        format="%.12f",
                        callback=self._update_return_margin,
                        # indent=100
                    )
                # dpg.add_text(default_value=f"Portfolio Value: {self.state['portfolio_value']}", tag='pf_value')
            dpg.bind_item_theme("sim_slider_window", self.mid_theme)
            dpg.bind_item_font("sim_group", self.fonts['sim_body'])

        self._build_error_modal()
    # |----------------------------- Window Initialization -----------------------------|


    # |----------------------------- Callback Functions --------------------------------|
    def _save_portfolio(self, sender, app_data):
        path = str(app_data["file_path_name"])

        if not path.lower().endswith(".json"):
            path += ".json"

        with open(path, "w") as f:
            json.dump(self.state, f, indent=4)

    def _load_portfolio(self, sender, app_data):
        try:
            with open(app_data["file_path_name"], "r") as f:
                loaded_data = json.load(f)

                # Unlike the initial automatic load, now the UI exists so we should
                #   try to update the UI to reflect this loaded data immediately.
                # |------------------------------- Load Lookback -------------------------------|
                self.state['lookback'] = lookback = loaded_data['lookback']
                sim_max = floor((self.T - lookback) / 252)
                dpg.set_value("sim_period", sim_max)
                dpg.configure_item("sim_period", max_value=sim_max, label=f"Simulation Period [1, {sim_max}] (years)")
                self._estimate_mu_sigma()
                # |------------------------------- Load Lookback -------------------------------|


                # |-------------------------- Redo Prior Optimization --------------------------|
                self.state['user_portfolio_fractional'] = loaded_data['user_portfolio_fractional']
                self.state['user_portfolio'] = loaded_data['user_portfolio']
                self.state['portfolio_value'] = loaded_data['portfolio_value']

                self.state['req_return_daily'] = loaded_data['req_return_daily']
                self.state['min_risk'] = list(loaded_data['min_risk'])

                # No sender, no app data, it's a routine.
                self._construct_portfolio(None, None)
                # |-------------------------- Redo Prior Optimization --------------------------|
                self._update_all_other_cards()
        except json.JSONDecodeError:
            self._show_error("That file is not valid JSON.")
        except FileNotFoundError:
            self._show_error("The selected file could not be found.")
        except KeyError as e:
            self._show_error(f"The file is missing required portfolio data: {e}")
        except Exception as e:
            self._show_error(f"Failed to load portfolio: {e}")

    def _clear_portfolio(self):
        equal_weight = 1.0 / self.N

        self.state['rec_portfolio'] = {name: equal_weight for name in self.asset_names}
        self.state['user_portfolio'] = {name: equal_weight * self.starting_value for name in self.asset_names}
        self.state['user_portfolio_fractional'] = {name: equal_weight for name in self.asset_names}
        self.state['portfolio_value'] = self.starting_value

        dpg.delete_item('sim_plot_y_axis', children_only=True, slot=1)

        self._update_all_other_cards()

    def _lookback_update(self, sender, app_data, user_data):
        """
        User requested different lookback length, so we need to update
            the max sim period and recalculate mu and sigma.

        user_data = self.T = total number of days in the data set
        app_data = the lookback length that the user requested
        """
        self.state['lookback'] = lookback = app_data
        sim_max = floor((user_data - lookback) / 252)

        if (dpg.get_value("sim_period") > sim_max):
            dpg.set_value("sim_period", sim_max)

        dpg.configure_item("sim_period", max_value=sim_max, label=f"Simulation Period (years)")

        # Need to recalculate, lookback changed
        self._estimate_mu_sigma()


    def _sim_period_update(self, sender, app_data, user_data):
        cur_lookback = dpg.get_value("lookback")
        sim_max = floor((user_data - cur_lookback) / 252)
        if (app_data > sim_max):
            with dpg.popup(dpg.get_item_parent(sender)):
                dpg.add_text("ERROR: Requested sim period exceeds available days - lookback period")
                dpg.set_value(sender, 1)
        else:
            dpg.configure_item(sender, max_value=sim_max, label=f"Simulation Period(years)")
            self.state['sim_period'] = app_data*252


    def _update_desired_ret(self, sender, app_data):
        # Convert abstracted risk level to real return value
        mu = np.asarray(self.state['mu'])
        mu_max = float(mu.max())

        # Max daily return should be mu_max * (max single contribution) - (3 * offset) / 2
        #   - max single contribution is the most that a single asset class can participate
        #       accounting for the minimum investment requested of the model
        #   - Offset is a small offset because numbers get weird near the margin
        #       2e-12 just seems to be feasible from trial and error.
        R_max = self.state['R_max'] = ((mu_max * (1.0 - 4 * self.state['w_min'])) - self.state['eps'] * (2))

        self.state['req_return_daily'] = app_data * R_max


    def _construct_portfolio(self, sender, app_data):
        if DEBUG_OUTPUT:
            print("\n\n\n\n\n\nNEW OPTIMIZATION BEGINNING")

        pf = self._build_long_only_portfolio(w_max=None)

        if pf is None:
            # Optimization failed
            # Collapse further
            return

        if pf['status'] == cp.OPTIMAL:
            print("Portfolio Solved:", pf['exp_return_ann'], pf['vol_ann'])
            print(f"Solved weights:\n"
                    + "\n".join(f"{name}: {pf['weights'][i]}" for i, name in enumerate(self.asset_names))
            )
        else:
            print("Target return unsolvable. Or potentially invalid R_target. Future pull up popup. For now crash")
            exit(-1)

        self._simulate_portfolio()
        self._update_sim_plot()
        self._save_portfolio(0, {"file_path_name": self.CWD / "app_state.json"})

    def _update_return_margin(self, sender, app_data):
        self.state["eps"] = app_data

    def _update_min_investment(self, sender, app_data):
        self.state["w_min"] = app_data
    # |----------------------------- Callback Functions --------------------------------|


    # |------------------------------ Helper Functions ---------------------------------|
    # |------------------------------- INTERNALIZED ----------------------------------|
    def _overunder(self, frac_in_val, frac_rec_val):
        if frac_in_val is None:
            frac_in_val = frac_rec_val

        drift = frac_in_val - frac_rec_val
        high = None
        if drift > 0.05:
            high = True
        elif drift < -0.05:
            high = False
        return drift * 100, high  # Convert drift to percent


    def _update_card_states(self, sender, app_data):
        pf_ref = self.state['user_portfolio']
        asset_name = self.asset_map[sender[0:7]]
        asset_idx = sender[6]

        pf_ref[asset_name] = app_data
        frac_pf_ref = self.state['user_portfolio_fractional']
        rec_ref = self.state['rec_portfolio']
        pf_val = self.state['portfolio_value'] = sum(self.state['user_portfolio'].values())

        if pf_val <= 0.00:
            # Definitely need to warn the user somehow and abort the update,
            #   they need to input a portfolio larger than $0...
            # For now just freak out in the console
            if DEBUG_OUTPUT:
                print("WHY WOULD YOU OPEN A PORTFOLIO OF $0 YOU ABSOLUTE MONGOLOID")

        # dpg.configure_item('pf_value', default_value=pf_val)

        # in_val = app_data
        if app_data > 0.00:
            frac_pf_ref[asset_name] = app_data / pf_val
        else:
            frac_pf_ref[asset_name] = 0.00

        if rec_ref[asset_name] == 0.00:
            if app_data > 0.00:
                dpg.configure_item(f"asset_{asset_idx}_participation", default_value=f"Overparticipating")
            else:
                dpg.configure_item(f"asset_{asset_idx}_participation", default_value=f"Asset Class Balanced!")

            rec_val = 0.00
            dpg.configure_item(f"asset_{asset_idx}_main_1", default_value=f"{pf_ref[asset_name]:.2f}")
            dpg.configure_item(f"asset_{asset_idx}_main_2", default_value=f"{(frac_in_val * 100):.2f}")
            dpg.configure_item(f"asset_{asset_idx}_rec_1", default_value=f"{rec_val:.2f}")
            dpg.configure_item(f"asset_{asset_idx}_rec_2", default_value=f"0.00")
        else:
            rec_val = pf_val * rec_ref[asset_name]
            frac_rec_val = rec_ref[asset_name]
            frac_in_val = frac_pf_ref[asset_name]

            change, high = self._overunder(frac_in_val, frac_rec_val)
            if high is None:
                dpg.configure_item(f"asset_{asset_idx}_participation", default_value=f"Asset Class Balanced!")
            elif high:
                dpg.configure_item(f"asset_{asset_idx}_participation", default_value=f"Overparticipating: {change:.2f}%")
            else:
                dpg.configure_item(f"asset_{asset_idx}_participation", default_value=f"Underparticipating: {change:.2f}%")

            dpg.configure_item(f"asset_{asset_idx}_main_1", default_value=f"{pf_ref[asset_name]:.2f}")
            dpg.configure_item(f"asset_{asset_idx}_main_2", default_value=f"{(frac_in_val * 100):.2f}")
            dpg.configure_item(f"asset_{asset_idx}_rec_1", default_value=f"{rec_val:.2f}")
            dpg.configure_item(f"asset_{asset_idx}_rec_2", default_value=f"{(frac_rec_val * 100):.2f}")

        self._update_all_other_cards(sender[0:7])


    def _update_all_other_cards(self, skip=None):
        pf_ref = self.state['user_portfolio']

        frac_pf_ref = self.state['user_portfolio_fractional']
        rec_ref = self.state['rec_portfolio']
        pf_val = self.state['portfolio_value'] = sum(self.state['user_portfolio'].values())

        # For asset_{i}, ticker_name
        for internal_name, real_name in self.asset_map.items():
            if internal_name != skip:
                frac_pf_ref[real_name] = pf_ref[real_name] / pf_val

                if rec_ref[real_name] == 0.00:
                    frac_in_val = frac_pf_ref[real_name]
                    if pf_ref[real_name] > 0.00:
                        dpg.configure_item(f"{internal_name}_participation", default_value=f"Overparticipating")
                    else:
                        dpg.configure_item(f"{internal_name}_participation", default_value=f"Asset Class Balanced!")

                    rec_val = 0.00
                    dpg.configure_item(f"{internal_name}_main_1", default_value=f"{pf_ref[real_name]:.2f}")
                    dpg.configure_item(f"{internal_name}_main_2", default_value=f"{(frac_in_val * 100):.2f}")
                    dpg.configure_item(f"{internal_name}_rec_1", default_value=f"{rec_val:.2f}")
                    dpg.configure_item(f"{internal_name}_rec_2", default_value=f"0.00")
                else:
                    rec_val = pf_val * rec_ref[real_name]
                    frac_rec_val = rec_ref[real_name]
                    frac_in_val = frac_pf_ref[real_name]

                    change, high = self._overunder(frac_in_val, frac_rec_val)
                    if high is None:
                        dpg.configure_item(f"{internal_name}_participation", default_value=f"Asset Class Balanced!")
                    elif high:
                        dpg.configure_item(f"{internal_name}_participation", default_value=f"Overparticipating: {change:.2f}%")
                    else:
                        dpg.configure_item(f"{internal_name}_participation", default_value=f"Underparticipating: {change:.2f}%")

                    dpg.configure_item(f"{internal_name}_main_1", default_value=f"{pf_ref[real_name]:.2f}")
                    dpg.configure_item(f"{internal_name}_main_2", default_value=f"{(frac_in_val * 100):.2f}")
                    dpg.configure_item(f"{internal_name}_rec_1", default_value=f"{rec_val:.2f}")
                    dpg.configure_item(f"{internal_name}_rec_2", default_value=f"{(frac_rec_val * 100):.2f}")


    def _estimate_mu_sigma(self):
        """
        Estimate mean values and covariance matrix of daily returns
        """
        lookback = self.state['lookback']

        if lookback is not None:
          local_data = self.returns.iloc[-lookback:]
        else:
          # Shouldn't be possible, for now just output
          if DEBUG_OUTPUT:
            print("Error: Lookback was None")
          pass

        mu = local_data.mean().values.flatten()

        # The original script allowed shrinking or not. My understanding is we pretty much always
        #   want to, so I've collapsed the conditional to a single path.
        lw = LedoitWolf().fit(local_data.values)
        sigma = lw.covariance_

        # This is a covariance matrix, it'd better already be symmetric.
        # Nonetheless, we explicitly make it symmetric for safety.
        # - If it weren't symmetric, there could be multiple local optima,
        #     And I'm pretty sure the solver would do this anyway.
        sigma = (sigma + sigma.T) / 2
        self.state['mu'] = mu.tolist()
        self.state['sigma'] = sigma.tolist()


    def _build_long_only_portfolio(self, w_max=None):
        """
        Given a target return (daily) that the user wants, provide an 'optimized' portfolio

        Important detail:
        - With long-only constraints, not every target return is feasible.
        - The feasible expected-return range is roughly [min(mu), max(mu)] under long-only,
        but in practice the covariance + sum-to-1 can still make some exact equality targets
        numerically finicky.
        - So we generate targets inside a "safe" range and skip infeasible solves.

        Solves:
            minimize      : w^T Sigma w
            Constrained by: sum(w) = 1
                            w >= 0
                            mu^T w = R_target
                            (optional) w <= w_max

        Parameters
        ----------
        w_max : float or (N,) array, optional
            Weight caps.
        solver : str
            Solver to use. OSQP is just as good here as it was for min var

        Returns
        -------
        portfolio : dict
            Contains weights, target return, realized return/vol, status.
        """
        mu = np.asarray(self.state['mu'])
        Sigma = np.asarray(self.state['sigma'])

        # Commented out for now. Probably warn the user (with explanation) and err out
        #   if it exceeds like 1e5 or something
        if DEBUG_OUTPUT:
            cond_number = np.linalg.cond(Sigma)
            eigs = np.linalg.eigvals(Sigma)
            print(f"Condition number: {cond_number}")
            print(f"Min Eigenvalue: {min(eigs)}\nMax Eigenvalue: {max(eigs)}")

        # End User Requested Portfolio
        N = Sigma.shape[0]

        w = cp.Variable(N)

        objective = cp.Minimize(cp.quad_form(w, Sigma))

        R_target = self.state['req_return_daily']

        if R_target is None:
            # The user hasn't selected a value yet.
            #   - Probably notify via popup and abort optimization
            print("ERROR: R_target uninitialized")
            self._show_error("Target return unspecified, wiggle the risk slider.")
            return None

        # Trying to minimize the work for OSQP to get it to work hopefully
        mu = np.asarray(mu, dtype=float).reshape(N,)
        R_target = float(R_target)

        constraints = [
            cp.sum(w) == 1,            # fully invested
            w >= self.state['w_min'],  # long-only, minimum {W_min*100}% participation
            (-mu) @ w <= -R_target     # target return or larger
        ]

        if DEBUG_OUTPUT:
            print("We're asking cvxpy to solve the following problem:")
            print(f"mu: {mu}\nsigma: {Sigma}\nTarget Daily Return: {R_target}")

        if w_max is not None:
            constraints.append(w <= w_max)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver="SCS", verbose=False)

        # Fail early if optimization was unsuccessful
        if w.value is None:
            if DEBUG_OUTPUT:
                print(f"Optimization was {problem.status}")
            if VERBOSE_DEBUG_OUTPUT:
                print(f"Solver statistics: {problem.solver_stats}")

            self._show_error("The optimizer was unable to solve for the requested return. Please Request a lower risk level or increase return margin.")
            return None

        w = np.asarray(w.value).ravel()
        status = problem.status

        # 'asset_names' ensures columns remain aligned
        # I could probably use the dict itself, it's in the right order,
        #   but this is a better design pattern (1 source of truth)
        self.state['rec_portfolio'] = dict(zip(self.asset_names, w))

        # We also want to coerce it to 0 to prevent odd stuff from happening
        for asset in self.state['rec_portfolio']:
            if self.state['rec_portfolio'][asset] < 0.00:
                self.state['rec_portfolio'][asset] = 0.00

        stats = self._compute_portfolio_stats(annualize=True)

        # Call update functions, new rec values
        self._update_all_other_cards()

        self.state['markowitz_status'] = status

        ret_val = {
            "target_R": float(R_target),
            "status": status,
            "weights": w,
            **stats
        }
        # End User Requested Portfolio

        # Min Risk Portfolio
        w = cp.Variable(N)

        constraints = []
        constraints.append(cp.sum(w) == 1)  # fully invested
        constraints.append(w >= 0)          # long-only (no shorting)

        objective = cp.Minimize(cp.quad_form(w, Sigma))

        problem = cp.Problem(objective, constraints)
        problem.solve(solver="SCS", verbose=False)

        self.state['min_risk'] = np.asarray(w.value).ravel().tolist()
        # End Min Risk Portfolio

        return ret_val


    def _compute_portfolio_stats(self, annualize=True):
        """
        Utility helper: compute expected return and volatility for weights w.

        mu: expected daily returns
        Sigma: daily covariance
        w: weights

        Returns dict with daily and (optionally) annualized metrics.
        """
        mu = np.asarray(self.state['mu'])
        Sigma = np.asarray(self.state['sigma'])
        w = np.asarray(list(self.state['rec_portfolio'].values()), dtype=float).reshape(-1)

        exp_ret_daily = float(mu @ w)
        var_daily = float(w.T @ Sigma @ w)
        vol_daily = float(np.sqrt(max(var_daily, 0.0)))

        out = {
            "exp_return_daily": exp_ret_daily,
            "vol_daily": vol_daily,
        }

        if annualize:
            out["exp_return_ann"] = exp_ret_daily * TRADING_DAYS
            out["vol_ann"] = vol_daily * np.sqrt(TRADING_DAYS)

        return out

    def _simulate_portfolio(self):
        pf_val = self.state['portfolio_value']

        sim_rets = self.returns.iloc[-self.state['sim_period']:]

        user_portfolio = np.array(list(self.state['user_portfolio'].values()))

        rec_risk = np.array(list(self.state['rec_portfolio'].values()))
        rec_portfolio = rec_risk * pf_val

        min_risk = np.asarray(self.state['min_risk'])
        min_portfolio = min_risk * pf_val

        Sigma = np.asarray(self.state['sigma'])
        max_asset = np.argmax(np.diag(Sigma))
        max_risk = np.zeros(len(self.asset_names))
        max_risk[max_asset] = 1
        max_portfolio = max_risk * pf_val

        print(max_portfolio)

        portfolios = np.vstack([user_portfolio, rec_portfolio, min_portfolio, max_portfolio])

        pfs_over_time = []

        for _, row in sim_rets.iterrows():
            # Assumes that data is aligned, matching mu in markowitz.py
            portfolios = portfolios * (1.0 + row[self.asset_names].values)
            pfs_over_time.append(portfolios.sum(axis=1).tolist())

        final_values = portfolios.sum(axis=1)

        self.state['simulated_portfolios'] = {
            "user_portfolio": float(final_values[0]),
            "rec_portfolio" : float(final_values[1]),
            "min_portfolio" : float(final_values[2]),
            "max_portfolio" : float(final_values[3])
        }

        self.state['portfolios_over_time'] = pfs_over_time

        pfs_save = pd.DataFrame(pfs_over_time)
        pfs_save.to_csv("simulated_portfolios.csv")


    def _update_sim_plot(self):
        """
            "user_portfolio"
            "rec_portfolio"
            "min_portfolio"
            "max_portfolio"
        """
        dpg.delete_item('sim_plot_y_axis', children_only=True, slot=1)

        pfs_over_time = self.state['portfolios_over_time']
        x = list(range(len(pfs_over_time)))

        cols = list(zip(*pfs_over_time))

        user_series = dpg.add_line_series(x=x, y=cols[0], parent="sim_plot_y_axis", label="User Portfolio")
        rec_series  = dpg.add_line_series(x=x, y=cols[1], parent="sim_plot_y_axis", label="Recommended Portfolio")
        min_series  = dpg.add_line_series(x=x, y=cols[2], parent="sim_plot_y_axis", label="Min Portfolio")
        max_series  = dpg.add_line_series(x=x, y=cols[3], parent="sim_plot_y_axis", label="Max Portfolio")

        with dpg.theme() as user_series_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, USER, category=dpg.mvThemeCat_Plots)

        with dpg.theme() as rec_series_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, REC, category=dpg.mvThemeCat_Plots)

        with dpg.theme() as min_series_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, MIN, category=dpg.mvThemeCat_Plots)

        with dpg.theme() as max_series_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, MAX, category=dpg.mvThemeCat_Plots)

        dpg.bind_item_theme(user_series, user_series_theme)
        dpg.bind_item_theme(rec_series, rec_series_theme)
        dpg.bind_item_theme(min_series, min_series_theme)
        dpg.bind_item_theme(max_series, max_series_theme)
    # |------------------------------- INTERNALIZED ----------------------------------|


    # |------------------------------- EXTERNALIZED ----------------------------------|
    def init_window_shells(self):
        # Externalized because we want to call it after calling maximize()
        #   if we call this too early, these two get_viewport calls
        #       reflect an early, incorrect value
        self.screen_width = dpg.get_viewport_client_width()
        self.screen_height = dpg.get_viewport_client_height()

        portfolio_window = dpg.add_window(
                                        tag="portfolio_window",
                                        pos=(0, 0),
                                        width=self.screen_width * (1 / 3) + 5,
                                        height=self.screen_height,
                                        no_move=True,
                                        no_resize=True,
                                        no_title_bar=True,
                                        no_scrollbar=True,
                                        no_scroll_with_mouse=True,
                                        )

        simulation_window = dpg.add_window(
                                        tag='simulation_window',
                                        pos=(self.screen_width * (1 / 3) + WINDOW_GAP, 0),
                                        width=(self.screen_width * (2 / 3) - WINDOW_GAP),
                                        height=self.screen_height,
                                        no_move=True,
                                        no_resize=True,
                                        no_title_bar=True,
                                        no_scrollbar=True,
                                        no_scroll_with_mouse=True,
                                        )


    def build_windows(self):
        # Background (main windows)
        with dpg.theme() as self.bg_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (15, 15, 18, 255))


        # Midground (rows / main child windows)
        with dpg.theme() as self.mid_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (32, 32, 36, 255))
                # dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 0)


        # Foreground (inner semantic boxes)
        with dpg.theme() as self.fg_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (35, 35, 39, 255))
                # dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 0)

        self._build_portfolio_window()
        self._build_sim_window()

        dpg.bind_item_theme("portfolio_window", self.bg_theme)
        dpg.bind_item_theme("simulation_window", self.bg_theme)
    # |------------------------------- EXTERNALIZED ----------------------------------|
    # |----------------------------- Helper Functions --------------------------------|
# |---------------------------------- App Class ----------------------------------|


# |-------------------------------- Main Function ---------------------------------|
def main():
    # |---------------------------- Screen Initialization -----------------------------|
    # We load tkinter literally just to fetch the screen specs
    root = tk.Tk()
    root.withdraw()
    height = root.winfo_screenheight()
    width = root.winfo_screenwidth()
    root.destroy()

    dpg.create_context()
    dpg.create_viewport(
        title='Portfolio Lab',
        width=width,
        height=height,
        x_pos=0, y_pos=0,
        )
    # |---------------------------- Screen Initialization -----------------------------|

    pl = PortfolioLab()

    # Anything defined after these calls simply won't appear
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.maximize_viewport()

    # Let one frame render, so the width/height match maximized size.
    dpg.render_dearpygui_frame()

    # Setup windows after the viewport is setup
    pl.init_window_shells()

    # Let another frame render so the windows are initialized properly
    dpg.render_dearpygui_frame()

    pl.build_windows()

    if pl.loaded_successfully:
        pl._construct_portfolio(None, None)

    if SHOW_DEMO:
        # Demo only shows up *inside* of the viewport we defined for the main window
        demo.show_demo()

    # This is our last chance to do anything, actions after this point won't populate the UI before it shows

    # This replaces dpg.start_dearpygui()
    while dpg.is_dearpygui_running():
        # This will run every frame
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
