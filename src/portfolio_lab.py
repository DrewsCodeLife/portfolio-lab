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
CARD_OUT_X_PAD = 0.05   # padding L / R
CARD_OUT_Y_PAD = 0.06   # Padding top / bottom
CARD_GAP_X     = 0.03   # Between card padding L / R
CARD_GAP_Y     = 0.06   # Between card padding top / bottom

CARD_ROUNDING     = 0.06  # rounding relative to card size
CARD_INTERNAL_PAD = 5     # Padding between card content and edge
CARD_SIZE         = 0.25  # Relative to window size

SIM_WINDOW_SIZE = 0.5
WINDOW_GAP      = 5

# Text size macros
SMALL_TEXT_SIZE  = 24.0
MEDIUM_TEXT_SIZE = 30.0
LARGE_TEXT_SIZE  = SMALL_TEXT_SIZE * 2
HEADER_TEXT_SIZE = SMALL_TEXT_SIZE * 3

SMALL_TEXT_PAD  = SMALL_TEXT_SIZE * 2
MEDIUM_TEXT_PAD = MEDIUM_TEXT_SIZE * 2
LARGE_TEXT_PAD  = LARGE_TEXT_SIZE * 2

TRADING_DAYS = 252
# |---------------------------------- Constants ----------------------------------|


# |---------------------------------- App Class ----------------------------------|
class PortfolioLab:
    def __init__(self):
        # |------------------------------ Internal Model ---------------------------------|
        self.state = {
            "rec_portfolio": {  # default portfolio assumes flat distribution
                "U.S. Equities"             : 0.2,
                "International Equities"    : 0.2,
                "Fixed Income"              : 0.2,
                "Real Assets"               : 0.2,
                "Cash Equivalents"          : 0.2
                },
            "user_portfolio": {
                "U.S. Equities"             : 20.0,
                "International Equities"    : 20.0,
                "Fixed Income"              : 20.0,
                "Real Assets"               : 20.0,
                "Cash Equivalents"          : 20.0
            },
            "user_portfolio_fractional": {
                "U.S. Equities"             : 0.2,
                "International Equities"    : 0.2,
                "Fixed Income"              : 0.2,
                "Real Assets"               : 0.2,
                "Cash Equivalents"          : 0.2
            },
            # 1 Source of truth for column order (from daily_returns.csv)
            "asset_names": [
                "U.S. Equities",
                "International Equities",
                "Fixed Income",
                "Real Assets",
                "Cash Equivalents"
            ],
            "simulated_portfolios": {
                "user_portfolio": 100,
                "rec_portfolio" : 100,
                "min_portfolio" : 100,
                "max_portfolio" : 100,
            },
            'portfolios_over_time'  : None,
            "min_risk"              : [],
            "max_risk"              : [],
            "portfolio_value"       : 100.0,
            "lookback"              : 252,
            "mu"                    : None, # daily
            "R_max"                 : None, # daily
            "sigma"                 : None, # daily
            "req_return_daily"      : None, # Converted to daily from UI annualized request
            "eps"                   : 1e-12,
            "w_min"                 : 0.01,
        }
        # |------------------------------ Internal Model ---------------------------------|


        # |---------------------------------- Data Model ----------------------------------|
        self.CWD = Path(__file__).parent.parent
        self.FONT_FOLDER = self.CWD / "font" / "League_Gothic" / "static"

        self.data    = pd.read_csv("data/cleaned/cleaned.csv")
        self.returns = pd.read_csv("data/cleaned/daily_returns.csv")

        self.dates = self.data["Date"]

        self.data = self.data.loc[:, self.data.columns != "Date"]
        self.returns = self.returns.loc[:, self.returns.columns != "Date"]

        # First day has no 'daily return', so we drop it
        self.data = self.data.iloc[1:].reset_index(drop=True)
        self.returns = self.returns.iloc[1:].reset_index(drop=True)

        self.T = len(self.returns)
        self.N = self.returns.shape[1]

        # Initialize to 0, should be updated by _compute_portfolio_layout()
        self.screen_width  = 0
        self.screen_height = 0
        # |---------------------------------- Data Model ----------------------------------|


        # |----------------------------- Window Initialization -----------------------------|
        # Drawing the window means we need fonts
        with dpg.font_registry():
            default_font = dpg.add_font(self.FONT_FOLDER / "LeagueGothic-Regular.ttf", SMALL_TEXT_SIZE)
            semicondensed_font = dpg.add_font(self.FONT_FOLDER / "LeagueGothic_SemiCondensed-Regular.ttf", MEDIUM_TEXT_SIZE)
            condensed_font = dpg.add_font(self.FONT_FOLDER / "LeagueGothic_Condensed-Regular.ttf", LARGE_TEXT_SIZE)

        dpg.bind_font(default_font)

        self.fonts = {
            "small": default_font,
            "medium": semicondensed_font,
            "large": condensed_font
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
        # |----------------------------- Window Initialization -----------------------------|


    # |----------------------------- Window Initialization -----------------------------|
    def _compute_portfolio_layout(self):
        width, height = dpg.get_item_rect_size("portfolio_window")

        self.gap_x       = int(width * CARD_GAP_X)  # First row 2, second row 1
        self.gap_y       = int(height * CARD_GAP_Y)  # One between row 1 and row 2
        self.outer_pad_x = int(width * CARD_OUT_X_PAD) # Two for row 1 and row 2
        self.outer_pad_y = int(height * CARD_OUT_Y_PAD)  # One above, one below

        self.card_w = int((width - (self.gap_x * 2) - (self.outer_pad_x * 2)) / 3)
        # Card height from 2 rows
        self.card_h = int((height - 2 * self.outer_pad_y - self.gap_y) / 2.0)

        # Enforce card slightly wider than tall
        self.card_h = min(self.card_h, self.card_w * 0.85)

        self.rounding = int(min(self.card_w, self.card_h) * CARD_ROUNDING)


    def _draw_card(self, tag, label, drift, high, callback):
        with dpg.child_window(
            tag=tag + '_window_1',
            width=self.card_w,
            height=self.card_h,
            no_scrollbar=True,
            no_scroll_with_mouse=True,
            ):
            with dpg.drawlist(
                width=self.card_w,
                height=self.card_h,
                ):
                dpg.draw_rectangle(
                    (0, 0),
                    (self.card_w, self.card_h),
                    color=(70, 70, 70, 255),
                    fill=(35, 35, 35, 255),
                    rounding=self.rounding,
                    thickness=1,
                )


            with dpg.child_window(
                pos=(0, 0),  # Draw text over rectangle
                tag=tag + '_window_2',
                width=self.card_w,
                height=self.card_h,
                border=False,
                ):

                with dpg.group():
                    dpg.add_text(
                        default_value=label,
                        pos=(CARD_INTERNAL_PAD, CARD_INTERNAL_PAD),
                        tag=tag + "_main",
                        )

                    bl_label = ''
                    if high is None:
                        # Balanced (+/- 5%)
                        bl_label = f"ASSET CLASS BALANCED!"
                        pass
                    elif high:
                        # More than 5% over contributed
                        bl_label = f"OVERPARTICIPATING: {drift:.3f}%"
                        pass
                    else:
                        # Less then 95% under contributed
                        bl_label = f"UNDERPARTICIPATING: {drift:.3f}%"

                    dpg.add_text(
                        default_value=bl_label,
                        pos=(CARD_INTERNAL_PAD, SMALL_TEXT_PAD + CARD_INTERNAL_PAD),
                        tag=tag + '_participation'
                    )

                    # step & step_fast = 0 disables +/- buttons
                    dpg.add_input_double(
                        pos=(CARD_INTERNAL_PAD, (SMALL_TEXT_PAD * 2) + CARD_INTERNAL_PAD),
                        tag=tag + "_input",
                        label="Current Portfolio Value",
                        width=self.card_w / 3,
                        step=0,
                        step_fast=0,
                        default_value=0.00,
                        min_value=0.00,
                        callback=callback,
                        )

                dpg.bind_item_font(tag + "_main", font=self.fonts["small"])
                dpg.bind_item_font(tag + '_participation', font=self.fonts["small"])

            dpg.bind_item_theme(tag + '_window_1', self.transparent_child_theme)
            dpg.bind_item_theme(tag + '_window_2', self.transparent_child_theme)

    def _build_portfolio_window(self):
        with use_parent('portfolio_window'):
            # Make sure we compute the inner size and card widths, etc. before using them
            self._compute_portfolio_layout()

            width, height = dpg.get_item_rect_size("portfolio_window")

            dpg.add_spacer(height=self.outer_pad_y)

            # Row 1 (3 cards), with left padding via spacer
            with dpg.group(horizontal=True):
                # Subtracting a magic number to make it centered
                #   - This is horrible design
                #   - Oh well. I've spent too many hours trying to align things.
                #   - Measured by a ruler, this is damn close, off by about 1/10 of an inch
                dpg.add_spacer(width=self.outer_pad_x - 5)

                pf_val = self.state['portfolio_value']

                in_val = self.state['user_portfolio']['U.S. Equities']
                rec_val = pf_val * self.state['rec_portfolio']['U.S. Equities']

                frac_in_val = self.state['user_portfolio_fractional']['U.S. Equities']
                frac_rec_val = self.state['rec_portfolio']['U.S. Equities']
                drift, high = self._overunder(frac_in_val, frac_rec_val)

                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=100)
                        us_equity_header = dpg.add_text("U.S. Equities")

                    self._draw_card(
                        tag="us_equities",
                        label=f"IN: ${in_val} / REC: ${rec_val:.2f}",
                        drift=drift,
                        high=high,
                        callback=self._us_class_update
                        )
                dpg.add_spacer(width=self.gap_x)

                in_val = self.state['user_portfolio']['International Equities']
                rec_val = pf_val * self.state['rec_portfolio']['International Equities']

                frac_in_val = self.state['user_portfolio_fractional']['International Equities']
                frac_rec_val = self.state['rec_portfolio']['International Equities']
                drift, high = self._overunder(frac_in_val, frac_rec_val)

                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=60)
                        int_equity_header = dpg.add_text("International Equities")

                    self._draw_card(
                        tag="int_equities",
                        label=f"IN: ${in_val} / REC: ${rec_val:.2f}",
                        drift=drift,
                        high=high,
                        callback=self._inter_class_update
                        )

                dpg.add_spacer(width=self.gap_x)
                in_val = self.state['user_portfolio']['Fixed Income']
                rec_val = pf_val * self.state['rec_portfolio']['Fixed Income']

                frac_in_val = self.state['user_portfolio_fractional']['Fixed Income']
                frac_rec_val = self.state['rec_portfolio']['Fixed Income']
                drift, high = self._overunder(frac_in_val, frac_rec_val)

                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=95)
                        fi_header = dpg.add_text("Fixed Income")

                    self._draw_card(
                        tag="fixed_income",
                        label=f"IN: ${in_val} / REC: ${rec_val:.2f}",
                        drift=drift,
                        high=high,
                        callback=self._fixed_class_update
                        )

                dpg.add_spacer(width=self.outer_pad_x)

            dpg.add_spacer(height=self.gap_y)

            # Row 2 (2 cards)
            with dpg.group(horizontal=True):
                # Similar to the first group, we use a magic number to center things.
                # @TODO: Fix this.
                left_offset = int((width - 2 * self.card_w - self.gap_x - 10) / 2)
                dpg.add_spacer(width=left_offset)

                in_val = self.state['user_portfolio']['Real Assets']
                rec_val = pf_val * self.state['rec_portfolio']['Real Assets']

                frac_in_val = self.state['user_portfolio_fractional']['Real Assets']
                frac_rec_val = self.state['rec_portfolio']['Real Assets']
                drift, high = self._overunder(frac_in_val, frac_rec_val)

                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=75)
                        reit_header = dpg.add_text("Real Assets / REITs")

                    self._draw_card(
                        tag="real_assets",
                        label=f"IN: ${in_val} / REC: ${rec_val:.2f}",
                        drift=drift,
                        high=high,
                        callback=self._reits_class_update
                        )

                dpg.add_spacer(width=self.gap_x)
                in_val = self.state['user_portfolio']['Cash Equivalents']
                rec_val = pf_val * self.state['rec_portfolio']['Cash Equivalents']

                frac_in_val = self.state['user_portfolio_fractional']['Cash Equivalents']
                frac_rec_val = self.state['rec_portfolio']['Cash Equivalents']
                drift, high = self._overunder(frac_in_val, frac_rec_val)

                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=50)
                        cash_header = dpg.add_text("Cash / Cash Equivalents")

                    self._draw_card(
                        tag="cash_equivalents",
                        label=f"IN: ${in_val} / REC: ${rec_val:.2f}",
                        drift=drift,
                        high=high,
                        callback=self._cash_class_update
                        )

        # Bind all the headers in one pass to keep things consistent.
        for item in (us_equity_header, int_equity_header, fi_header, reit_header, cash_header):
            dpg.bind_item_font(item, font=self.fonts["large"])


    def _build_sim_window(self):
        with use_parent('simulation_window'):
            with dpg.group():
                with dpg.child_window(
                    width=-1,
                    height=self.screen_height * SIM_WINDOW_SIZE,
                    border=True,
                    tag='sim_plot_window'
                    ):
                    with dpg.plot(
                        label="Simulation Plot",
                        width=-1,
                        height=-1,
                        tag="sim_plot"
                        ): # width=-1 auto-expands
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Trading Days", tag="sim_plot_x_axis", auto_fit=True)
                        dpg.add_plot_axis(dpg.mvYAxis, label="Value ($)", tag="sim_plot_y_axis", auto_fit=True)

                dpg.add_spacer(height=20)

                # Here we can define the manual data entry
                dpg.add_slider_int(
                    label="Lookback period (days)",
                    tag="lookback",
                    min_value=252,
                    max_value=(self.T - 252),
                    callback=self._lookback_update,
                    user_data=self.T,
                    default_value=252
                    )
                dpg.add_slider_int(
                    label=f"Simulation Period [1, {floor(self.T / 252)}] (years)",
                    tag="sim_period",
                    min_value=1,
                    max_value=floor(self.T / 252),
                    user_data=self.T,
                    default_value=1,
                    callback=self._sim_period_update,
                    )

                dpg.add_slider_float(
                    label="Low -> High Risk",
                    tag="desired_ret",
                    min_value=0.05,
                    max_value=1.0,
                    default_value=0.5,
                    callback=self._update_desired_ret,
                    )
                dpg.add_slider_double(
                    label="Model Minimum Investment",
                    tag="model_min_investment",
                    min_value=0.00,
                    max_value=0.2,
                    default_value=0.01,
                    callback=self._update_min_investment
                )
                dpg.add_slider_double(
                    label="Model Return Margin",
                    tag="model_return_margin",
                    min_value=1e-12,
                    max_value=1e-3,
                    default_value=1e-12,
                    clamped=True,
                    format="%.12f",
                    callback=self._update_return_margin
                )
                dpg.add_text(default_value=self.state['portfolio_value'], tag='pv_value')
                dpg.add_button(label="Optimize Portfolio", callback=self._construct_portfolio)

    # |----------------------------- Window Initialization -----------------------------|


    # |----------------------------- Callback Functions --------------------------------|
    def _lookback_update(self, sender, app_data, user_data):
        self.state['lookback'] = lookback = app_data
        sim_max = floor((user_data - lookback) / 252)

        if (dpg.get_value("sim_period") > sim_max):
            dpg.set_value("sim_period", sim_max)

        dpg.configure_item("sim_period", max_value=sim_max, label=f"Simulation Period [1, {sim_max}] (years)")

        # Need to recalculate, lookback changed
        self._estimate_mu_sigma()


    def _sim_period_update(self, sender, app_data, user_data):
        cur_lookback = dpg.get_value("lookback")
        sim_max = floor((user_data - cur_lookback) / 252)
        if (app_data > sim_max):
            with dpg.popup(dpg.get_item_parent(sender)):
                dpg.add_text("ERROR: Requested sim period exceeds available days - lookback period")
                dpg.set_value(sender, 1)

        dpg.configure_item(sender, max_value=sim_max, label=f"Simulation Period [1, {sim_max}] (years)")


    def _us_class_update(self, sender, app_data):
        self.state['user_portfolio']['U.S. Equities'] = app_data
        self._update_all_card_states()


    def _inter_class_update(self, sender, app_data):
        self.state['user_portfolio']['International Equities'] = app_data
        self._update_all_card_states()


    def _fixed_class_update(self, sender, app_data):
        self.state['user_portfolio']['Fixed Income'] = app_data
        self._update_all_card_states()


    def _reits_class_update(self, sender, app_data):
        self.state['user_portfolio']['Real Assets'] = app_data
        self._update_all_card_states()


    def _cash_class_update(self, sender, app_data):
        self.state['user_portfolio']['Cash Equivalents'] = app_data
        self._update_all_card_states()


    def _update_desired_ret(self, sender, app_data):
        # Convert abstracted risk level to real return value
        mu = self.state['mu']
        mu_max = float(mu.max())

        # Max daily return should be mu_max * (max single contribution) - (3 * offset) / 2
        #   - max single contribution is the most that a single asset class can participate
        #       accounting for the minimum investment requested of the model
        #   - Offset is a small offset because numbers get weird near the margin
        #       3*1e-12/2 just seems to be feasible from trial and error.
        R_max = self.state['R_max'] = ((mu_max * (1.0 - 4 * self.state['w_min'])) - self.state['eps'] * (3/2))

        self.state['req_return_daily'] = app_data * R_max


    def _construct_portfolio(self, sender, app_data):
        if DEBUG_OUTPUT:
            print("\n\n\n\n\n\nNEW OPTIMIZATION BEGINNING")

        pf = self._build_long_only_portfolio(w_max=None)

        asset_names = self.state['asset_names']

        if pf is None:
            # Optimization failed
            # Collapse further
            return

        if pf['status'] == cp.OPTIMAL:
            print("Portfolio Solved:", pf['exp_return_ann'], pf['vol_ann'])
            print(f"Solved weights:\n"
                    + "\n".join(f"{name}: {pf['weights'][i]}" for i, name in enumerate(asset_names))
            )
        else:
            print("Target return unsolvable. Or potentially invalid R_target. Future pull up popup. For now crash")
            exit(-1)

        self._simulate_portfolio()
        self._update_sim_plot()

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


    def _update_all_card_states(self):
        pf_ref = self.state['user_portfolio']
        frac_pf_ref = self.state['user_portfolio_fractional']
        rec_ref = self.state['rec_portfolio']
        pf_val = self.state['portfolio_value'] = sum(self.state['user_portfolio'].values())

        if pf_val <= 0.00:
            # Definitely need to warn the user somehow and abort the update,
            #   they need to input a portfolio larger than $0...
            # For now just freak out in the console
            if DEBUG_OUTPUT:
                print("WHY WOULD YOU OPEN A PORTFOLIO OF $0 YOU ABSOLUTE MONGOLOID")

        dpg.configure_item('pv_value', default_value=pf_val)

        # |------------------------------ U.S. Equities ------------------------------|
        in_val = pf_ref['U.S. Equities']

        if in_val > 0.00:
            frac_pf_ref['U.S. Equities'] = in_val / pf_val
        else:
            frac_pf_ref['U.S. Equities'] = 0.00

        if rec_ref['U.S. Equities'] == 0.00:
            if in_val > 0.00:
                dpg.configure_item('us_equities_participation', default_value=f"OVERPARTICIPATING")
            else:
                dpg.configure_item('us_equities_participation', default_value=f"ASSET CLASS BALANCED!")
            rec_val = 0.00
            dpg.configure_item('us_equities_main', default_value=f"IN: ${in_val} / REC: $0")
        else:
            rec_val = pf_val * rec_ref['U.S. Equities']
            frac_rec_val = rec_ref['U.S. Equities']
            frac_in_val = frac_pf_ref['U.S. Equities']

            change, high = self._overunder(frac_in_val, frac_rec_val)
            if high is None:
                dpg.configure_item('us_equities_participation', default_value=f"ASSET CLASS BALANCED!")
            elif high:
                dpg.configure_item('us_equities_participation', default_value=f"OVERPARTICIPATING: {change:.3f}%")
            else:
                dpg.configure_item('us_equities_participation', default_value=f"UNDERPARTICIPATING: {change:.3f}%")

            dpg.configure_item('us_equities_main', default_value=f"IN: ${in_val} / REC: ${rec_val:.2f}")
        # |------------------------------ U.S. Equities ------------------------------|

        # |------------------------- International Equities -------------------------|
        in_val = pf_ref['International Equities']

        if in_val > 0.00:
            frac_pf_ref['International Equities'] = in_val / pf_val
        else:
            frac_pf_ref['International Equities'] = 0.00

        if rec_ref['International Equities'] == 0.00:
            if in_val > 0.00:
                dpg.configure_item('int_equities_participation', default_value=f"OVERPARTICIPATING")
            else:
                dpg.configure_item('int_equities_participation', default_value=f"ASSET CLASS BALANCED!")
            rec_val = 0.00
            dpg.configure_item('int_equities_main', default_value=f"IN: ${in_val} / REC: $0")
        else:
            rec_val = pf_val * rec_ref['International Equities']
            frac_rec_val = rec_ref['International Equities']
            frac_in_val = frac_pf_ref['International Equities']

            change, high = self._overunder(frac_rec_val, frac_in_val)
            if high is None:
                dpg.configure_item('int_equities_participation', default_value=f"ASSET CLASS BALANCED!")
            elif high:
                dpg.configure_item('int_equities_participation', default_value=f"OVERPARTICIPATING: {change:.3f}%")
            else:
                dpg.configure_item('int_equities_participation', default_value=f"UNDERPARTICIPATING: {change:.3f}%")

            dpg.configure_item('int_equities_main', default_value=f"IN: ${in_val} / REC: ${rec_val:.2f}")
        # |------------------------- International Equities -------------------------|

        # |------------------------------ Fixed Income ------------------------------|
        in_val = pf_ref['Fixed Income']

        if in_val > 0.00:
            frac_pf_ref['Fixed Income'] = in_val / pf_val
        else:
            frac_pf_ref['Fixed Income'] = 0.00

        if rec_ref['Fixed Income'] == 0.00:
            if in_val > 0.00:
                dpg.configure_item('fixed_income_participation', default_value=f"OVERPARTICIPATING")
            else:
                dpg.configure_item('fixed_income_participation', default_value=f"ASSET CLASS BALANCED!")
            rec_val = 0.00
            dpg.configure_item('fixed_income_main', default_value=f"IN: ${in_val} / REC: $0")
        else:
            rec_val = pf_val * rec_ref['Fixed Income']
            frac_rec_val = rec_ref['Fixed Income']
            frac_in_val = frac_pf_ref["Fixed Income"]

            change, high = self._overunder(frac_in_val, frac_rec_val)
            if high is None:
                dpg.configure_item('fixed_income_participation', default_value=f"ASSET CLASS BALANCED!")
            elif high:
                dpg.configure_item('fixed_income_participation', default_value=f"OVERPARTICIPATING: {change:.3f}%")
            else:
                dpg.configure_item('fixed_income_participation', default_value=f"UNDERPARTICIPATING: {change:.3f}%")

            dpg.configure_item('fixed_income_main', default_value=f"IN: ${in_val} / REC: ${rec_val:.2f}")
        # |------------------------------ Fixed Income ------------------------------|

        # |------------------------------ Real Assets ------------------------------|
        in_val = pf_ref['Real Assets']

        if in_val > 0.00:
            frac_pf_ref['Real Assets'] = in_val / pf_val
        else:
            frac_pf_ref['Real Assets'] = 0.00

        if rec_ref['Real Assets'] == 0.00:
            if in_val > 0.00:
                dpg.configure_item('real_assets_participation', default_value=f"OVERPARTICIPATING")
            else:
                dpg.configure_item('real_assets_participation', default_value=f"ASSET CLASS BALANCED!")
            rec_val = 0.00
            dpg.configure_item('real_assets_main', default_value=f"IN: ${in_val} / REC: $0")
        else:
            rec_val = pf_val * rec_ref['Real Assets']
            frac_rec_val = rec_ref['Real Assets']
            frac_in_val = frac_pf_ref['Real Assets']

            change, high = self._overunder(frac_in_val, frac_rec_val)
            if high is None:
                dpg.configure_item('real_assets_participation', default_value=f"ASSET CLASS BALANCED!")
            elif high:
                dpg.configure_item('real_assets_participation', default_value=f"OVERPARTICIPATING: {change:.3f}%")
            else:
                dpg.configure_item('real_assets_participation', default_value=f"UNDERPARTICIPATING: {change:.3f}%")

            dpg.configure_item('real_assets_main', default_value=f"IN: ${in_val} / REC: ${rec_val:.2f}")
        # |------------------------------ Real Assets ------------------------------|

        # |--------------------------- Cash Equivalents ---------------------------|
        in_val = pf_ref['Cash Equivalents']

        if in_val > 0.00:
            frac_pf_ref['Cash Equivalents'] = in_val / pf_val
        else:
            frac_pf_ref['Cash Equivalents'] = 0.00

        if rec_ref['Cash Equivalents'] == 0.00:
            if in_val > 0.00:
                dpg.configure_item('cash_equivalents_participation', default_value=f"OVERPARTICIPATING")
            else:
                dpg.configure_item('cash_equivalents_participation', default_value=f"ASSET CLASS BALANCED!")
            rec_val = 0.00
            dpg.configure_item('cash_equivalents_main', default_value=f"IN: ${in_val} / REC: $0")
        else:
            rec_val = pf_val * rec_ref['Cash Equivalents']
            frac_rec_val = rec_ref['Cash Equivalents']
            frac_in_val = frac_pf_ref['Cash Equivalents']

            change, high = self._overunder(frac_in_val, frac_rec_val)

            if high is None:
                dpg.configure_item('cash_equivalents_participation', default_value=f"ASSET CLASS BALANCED!")
            elif high:
                dpg.configure_item('cash_equivalents_participation', default_value=f"OVERPARTICIPATING: {change:.3f}%")
            else:
                dpg.configure_item('cash_equivalents_participation', default_value=f"UNDERPARTICIPATING: {change:.3f}%")

            dpg.configure_item('cash_equivalents_main', default_value=f"IN: ${in_val} / REC: ${rec_val:.2f}")
        # |--------------------------- Cash Equivalents ---------------------------|


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

        mu = local_data.mean().values.reshape(-1, 1) # (5x1 vector, reshaped for matrix mult)

        # The original script allowed shrinking or not. My understanding is we pretty much always
        #   want to, so I've collapsed the conditional to a single path.
        lw = LedoitWolf().fit(local_data.values)
        sigma = lw.covariance_

        # This is a covariance matrix, it'd better already be symmetric.
        # Nonetheless, we explicitly make it symmetric for safety.
        # - If it weren't symmetric, there could be multiple local optima,
        #     And I'm pretty sure the solver would do this anyway.
        sigma = (sigma + sigma.T) / 2
        self.state['mu'] = np.asarray(mu).reshape(-1) # We most often use it in this form
        self.state['sigma'] = sigma


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
        mu = self.state['mu']
        Sigma = self.state['sigma']

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
            return None

        # Trying to minimize the work for OSQP to get it to work hopefully
        mu = np.asarray(mu, dtype=float).reshape(N,)
        R_target = float(R_target)

        constraints = [
            cp.sum(w) == 1,          # fully invested
            w >= self.state['w_min'],              # long-only, minimum {W_min*100}% participation
            (-mu) @ w <= -R_target   # target return or larger
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
            return None

        w = np.asarray(w.value).ravel()
        status = problem.status

        # 'asset_names' ensures columns remain aligned
        # I could probably use the dict itself, it's in the right order,
        #   but this is a better design pattern (1 source of truth)
        self.state['rec_portfolio'] = dict(zip(self.state['asset_names'], w))

        # We also want to coerce it to 0 to prevent odd stuff from happening
        for asset in self.state['rec_portfolio']:
            if self.state['rec_portfolio'][asset] < 0.00:
                self.state['rec_portfolio'][asset] = 0.00

        stats = self._compute_portfolio_stats(annualize=True)

        # Call update functions, new rec values
        self._update_all_card_states()

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

        self.state['min_risk'] = np.asarray(w.value).ravel()
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
        mu = self.state['mu']
        Sigma = self.state['sigma']
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
        classes = ["US_Equity", "International", "Bonds", "REITs", "Cash"]
        pf_val = self.state['portfolio_value']

        # We don't want to simulate over the same period we trained the model
        # - Model uses the last 'lookback' rows, so we drop them
        sim_rets = self.returns.iloc[:-self.state['lookback']]

        user_portfolio = np.array(list(self.state['user_portfolio'].values()))

        rec_risk = np.array(list(self.state['rec_portfolio'].values()))
        rec_portfolio = rec_risk * pf_val

        min_risk = self.state['min_risk']
        min_portfolio = min_risk * pf_val

        Sigma = self.state['sigma']
        max_asset = np.argmax(np.diag(Sigma))
        max_risk = np.zeros(5)
        max_risk[max_asset] = 1
        max_portfolio = max_risk * pf_val

        print(max_portfolio)

        portfolios = np.vstack([user_portfolio, rec_portfolio, min_portfolio, max_portfolio])

        pfs_over_time = []

        for _, row in sim_rets.iterrows():
            # Assumes that data is aligned, matching mu in markowitz.py
            portfolios = portfolios * (1.0 + row[classes].values)
            pfs_over_time.append(portfolios.sum(axis=1).copy())

        final_values = portfolios.sum(axis=1)

        self.state['simulated_portfolios'] = {
            "user_portfolio": final_values[0],
            "rec_portfolio" : final_values[1],
            "min_portfolio" : final_values[2],
            "max_portfolio" : final_values[3]
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

        dpg.add_line_series(x=x, y=cols[0], parent="sim_plot_y_axis", label="User Portfolio")
        dpg.add_line_series(x=x, y=cols[1], parent="sim_plot_y_axis", label="Recommended Portfolio")
        dpg.add_line_series(x=x, y=cols[2], parent="sim_plot_y_axis", label="Min Portfolio")
        dpg.add_line_series(x=x, y=cols[3], parent="sim_plot_y_axis", label="Max Portfolio")
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
                                        width=self.screen_width * (2 / 3),
                                        height=self.screen_height,
                                        no_move=True,
                                        no_title_bar=True
                                        )

        simulation_window = dpg.add_window(
                                        tag='simulation_window',
                                        pos=(self.screen_width * (2 / 3) + WINDOW_GAP, 0),
                                        width=(self.screen_width * (1 / 3) - WINDOW_GAP),
                                        height=self.screen_height,
                                        no_move=True,
                                        no_title_bar=True,
                                        )

        with dpg.theme() as portfolio_window_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme(portfolio_window, portfolio_window_theme)


    def build_windows(self):
        self._build_portfolio_window()
        self._build_sim_window()
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
